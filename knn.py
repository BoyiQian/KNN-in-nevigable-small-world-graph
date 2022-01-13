
from collections import OrderedDict
import numpy as np
import random
from scipy.spatial import distance
import datetime


class Node:
    """
    node in graph,location is the vector and connect is the neigbors of this node
    """

    def __init__(self, location, connect,index):
        self.location = location
        self.connect = connect
        self.index = index

    def __repr__(self):
        return ", ".join(map(str,self.location))


class ModifiedANN:
    """
    train is a list containing current node in a graph
    """
    def __init__(self, train):
        self.train = train

    def euc(self, a, b):
        return distance.euclidean(a, b)

    def knnsearch(self, q, m, k):
        """
        knn search
        :param q: list or iterable of vector location information
        :param m: range m
        :param k: length k
        :return: bestk
        """
        start = datetime.datetime.now()
        tempRes = OrderedDict()
        visitedSet = set()
        result = OrderedDict()
        for i in range(m):
            candidates = random.sample(self.train, len(self.train)//100+1)
            while True:
                if not candidates:
                    break
                mindist = self.euc(candidates[0].location, q)
                c = candidates[0]
                minindex = 0
                for element in range(1, len(candidates)):
                    if self.euc(candidates[element].location, q) < mindist:
                        mindist = self.euc(candidates[element].location, q)
                        c = candidates[element]
                        minindex = element
                del candidates[minindex]
                if len(result) >= k:
                    if mindist > list(result.values())[:k][-1]:
                        break
                flag = 0
                for e in c.connect:
                    if e not in visitedSet:
                        visitedSet.add(e)
                        candidates.append(e)
                        tempRes[e] = self.euc(e.location, q)
                        flag = 1
                if flag == 0:
                    break
            for key, value in tempRes.items():
                result[key] = value
            result = OrderedDict(sorted(result.items(), key=lambda t: t[1]))
        end = datetime.datetime.now()
        # print(f"the time consume in this search is {end-start}")
        return list(result)[:k]

    def insert(self, new_object, f, w):
        """
        add new node into graph
        :param new_object: list or iterable of vector location information
        :param f: length
        :param w: range
        :return: new node inserted
        """
        neighbors = self.knnsearch(new_object, w, f)
        new_node = Node(new_object, [], 0)

        for i in range(f):
            neighbors[i].connect.append(new_node)
            new_node.connect.append(neighbors[i])
        return new_node


# a = Node([1,54,32],[])
# b = Node([3,6,23],[])
# c = Node([12,12,1],[a])
# d = Node([12,12,3],[b])
# a.connect.append(c)
# b.connect.append(d)
#
# train = [a,b,c,d]
# k = ModifiedANN(train)
# print(k.knnsearch([44,12,15],3,2))
# print(k.insert([44,12,15],2,3))


def fvecs_read(filename,flag, c_contiguous=True):
    if flag == 1:
        fv = np.fromfile(filename, dtype=np.float32)
    else:
        fv = np.fromfile(filename, dtype=np.int32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    # print(len(fv))
    return fv


base = fvecs_read('siftsmall_base.fvecs',1)
truth = fvecs_read('siftsmall_groundtruth.ivecs',2)
query = fvecs_read('siftsmall_query.fvecs',1)
print("\nbase:")
print(base)
print("\ntruth:")
print(truth)
print("\nquery:")
print(query)
####### parameters ########

"""
insert -- add new node into graph
:param f: f local closest element
:param w: affect the accuracy of recall
:return: new node inserted
"""
w = 2
f = 10


"""
knn search
:param m: serious of m search, return the best result of them
:param k: closest kth neighbor
:return: bestk
"""
m = 30
k0 = 10


start1 = datetime.datetime.now()
# build graph
train = []
for i in range(100):
    train.append(Node(base[i],[],i))
for i in range(100):
    for j in range(100):
        if i != j:
            train[i].connect.append(train[j])

# insert train
k = ModifiedANN(train)
for i in range(100,len(base)):
    new_node = k.insert(base[i],f,w)
    new_node.index = i
    k.train.append(new_node)
end1 = datetime.datetime.now()
print(f"the time consume in build is {end1 - start1}")

# insert query
start = datetime.datetime.now()
cnt = 0
total = 0
for i in range(len(query)):
    result = k.knnsearch(query[i],m,k0)
    for j in range(len(result)):
        total+=1
        if result[j].index in truth[i][:k0]:
            cnt += 1
recall = cnt/total
print(result)
print(recall)
end = datetime.datetime.now()
print(f"the time consume in this insert is {end - start}")