from ogb.nodeproppred import NodePropPredDataset
from initG import *
from numba import njit, objmode, cuda, jit, vectorize
import numba
import numpy as np
import scipy.sparse as sp
import sklearn.preprocessing
import time
from tqdm import tqdm as tqdm


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px != py:
            self.parent[px] = py

    def connected(self, x, y):
        return self.find(x) == self.find(y)


def kruskal(edges, n):
    uf = UnionFind(n)
    T = []
    for edge in tqdm(edges):
        if not uf.connected(edge[0], edge[1]):
            uf.union(edge[0], edge[1])
            T.append(edge)
            if len(T) == n - 1:
                break
    return T


np.random.seed(17)
dataset = NodePropPredDataset(name='ogbn-products')
graph_info, labels = dataset[0]
node_feature = graph_info['node_feat']
split_idx = dataset.get_idx_split()
n = graph_info['num_nodes']
edge_info = graph_info['edge_index']
raw_row = edge_info[0]
raw_col = edge_info[1]
adic = {}
self_loop = {}
num = 0
same_num = 0
for i in tqdm(range(len(raw_row))):
    u = raw_row[i]
    v = raw_col[i]
    if u == v:
        self_loop[(u,v)] = 0
        continue
    if (u,v) not in adic:
        adic[(u,v)] = False
    else:
        num += 1
    if (v,u) not in adic:
        adic[(v,u)] = False
    else:
        num += 1
raw_row = []
raw_col = []
edge_info = list(adic.keys())
loop_edge = list(self_loop.keys())
'''for i in range(n):
    self_loop[(i,i)] = 0
loop_edge = list(self_loop.keys())'''

for i in tqdm(range(len(edge_info))):
    raw_row.append(edge_info[i][0])
    raw_col.append(edge_info[i][1])
for i in range(len(loop_edge)):
    raw_row.append(loop_edge[i][0])
    raw_col.append(loop_edge[i][1])
data = np.array([1] * len(raw_row), dtype=np.int64)
adj_matrix = sp.csr_matrix((data, (raw_row, raw_col)), shape=(n, n))
indices = adj_matrix.indices
indptr = adj_matrix.indptr
degree = np.zeros(n, dtype=np.int64)

edges = list(adic.keys())[0::2]
np.random.shuffle(edges)
stedges = kruskal(edges, n)

print(degree.sum())
for edge in tqdm(loop_edge):
    node = edge[0]
    indices[indptr[node] + degree[node]] = node
    degree[node] += 1
print(degree.sum())
for edge in tqdm(stedges):
    node = edge[0]
    to_node = edge[1]
    indices[indptr[node] + degree[node]] = to_node
    indices[indptr[to_node] + degree[to_node]] = node
    adic[(node, to_node)] = 1
    adic[(to_node, node)] = 1
    degree[node] += 1
    degree[to_node] += 1
print(degree.sum())
for node in tqdm(range(n)):
    for i in range(indptr[node] + degree[node], indptr[node + 1]):
        indices[i] = n
print(degree.sum())

raw_row = []
raw_col = []
for key, val in tqdm(adic.items()):
    if val:
        continue
    raw_row.append(key[0])
    raw_col.append(key[1])
raw_row = raw_row[0::2]
raw_col = raw_col[0::2]
m = len(raw_row)

edge_stream = np.array(list(range(m)))
np.random.shuffle(edge_stream)
delnum = 30000000
new_dict = {}
for i in tqdm(range(m)):
    new_dict[(raw_row[i], raw_col[i])] = 0
edges = list(new_dict.keys())
for i in tqdm(range(m - delnum)):
    edge = edges[edge_stream[i]]
    new_dict[edge] = 1
    node = edge[0]
    to_node = edge[1]
    indices[indptr[node] + degree[node]] = to_node
    indices[indptr[to_node] + degree[to_node]] = node
    degree[node] += 1
    degree[to_node] += 1
print(degree.sum())

raw_row = []
raw_col = []
for key, val in tqdm(new_dict.items()):
    if val:
        continue
    raw_row.append(key[0])
    raw_col.append(key[1])
m = len(raw_row)
raw_row = np.array(raw_row)
raw_col = np.array(raw_col)
snapshot = 15
edge_stream = np.array(list(range(m)))
np.random.shuffle(edge_stream)

batchsize = int(m / snapshot)
Us = np.zeros((snapshot, batchsize), dtype = np.int32)
Vs = np.zeros((snapshot, batchsize), dtype = np.int32)
for nowsnapshot in range(snapshot):
    Us[nowsnapshot] = raw_row[edge_stream[nowsnapshot * batchsize: (nowsnapshot + 1) * batchsize]]
    Vs[nowsnapshot] = raw_col[edge_stream[nowsnapshot * batchsize: (nowsnapshot + 1) * batchsize]]

np.savez('./dataset/ogbn-products-exp.npz', indices=indices, indptr=indptr, degree=degree, Us = Us, Vs = Vs,
         node_feature=node_feature, labels=labels, split_idx=split_idx)


