import numpy as np
import torch
import scipy.sparse as sp
import numba
from katz_solver import *
import warnings

warnings.filterwarnings("ignore")
large_graph = ['com-friendster', 'ogbn-papers100M']
graph_names = ['Cora', 'Citeseer', 'ogbn-arxiv', 'as-skitter', 'ogbn-proteins', 'com-orkut', 'cit-patent', 'ogbl-ppa',
               'ogbn-products', 'wiki-talk', 'com-youtube', 'ogbn-mag', 'soc-lj1', 'reddit', 'pubmed', 'wiki-en21',
               'com-friendster', 'ogbn-papers100M']
path = './dataset/'

all_result = {}
ppr_result = np.load('./results/ppr_exp_pk_result_new.npy', allow_pickle=True)
ppr_result = ppr_result.reshape(1, -1)[0][0]
lambda_result = np.load('./results/katz_exp_lambda.npy', allow_pickle=True)
lambda_result = lambda_result.reshape(1, -1)[0][0]
for graph_name in graph_names:
    graph_path = path + graph_name + '/'
    adj_matrix = sp.load_npz(graph_path + graph_name + '_csr-mat.npz')
    indices = adj_matrix.indices
    indptr = adj_matrix.indptr
    n = len(indptr) - 1
    m = len(indices)
    lambda_1 = lambda_result[graph_name]
    lambda_result[graph_name] = lambda_1
    alpha = 1. / (lambda_1 + 1)
    omega = 2. / (1 + np.sqrt(1 - (alpha * lambda_1) ** 2))
    degree = np.array(adj_matrix.sum(1)).flatten()
    if graph_name == 'ogbn-papers100M' or graph_name == 'com-friendster':
        eps = 1e-4 / (m + n)
    else:
        eps = 1e-10 / (m + n)
    np.random.seed(17)
    s_nodes = ppr_result[graph_name][2][0]
    print(graph_name, len(s_nodes), end=' ')
    result = solve_a_graph_pk(n, indptr, indices, degree, alpha, eps, omega, graph_name, s_nodes)
    all_result[graph_name] = (n, degree.mean(), result)
    print(all_result[graph_name][0], all_result[graph_name][1], all_result[graph_name][2][1].mean())
    np.save('./results/katz_exp_pk_result_new.npy', all_result)