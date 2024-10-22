import numpy as np
import scipy.sparse as sp
import numba
from katz_solver import *

graph_names = ['Cora', 'Citeseer', 'ogbn-arxiv', 'as-skitter', 'ogbn-proteins',
               'com-orkut', 'cit-patent', 'ogbl-ppa', 'ogbn-products', 'wiki-talk', 'com-youtube', 'ogbn-mag',
               'soc-lj1', 'reddit', 'pubmed', 'wiki-en21', 'com-friendster', 'ogbn-papers100M']
path = './dataset/'
all_result = {}

for graph_name in graph_names:
    graph_path = path + graph_name + '/'
    adj_matrix = sp.load_npz(graph_path + graph_name + '_csr-mat.npz').astype(np.float32)
    indices = adj_matrix.indices
    indptr = adj_matrix.indptr
    n = len(indptr) - 1
    m = len(indices)
    print(graph_name, end=' ')
    d_max = sp.linalg.eigsh(adj_matrix, 1, which='SA')[0][0]
    print(d_max)
    all_result[graph_name] = d_max
    np.save('./results/katz_exp_lambda_min.npy', all_result)
