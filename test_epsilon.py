import numpy as np
import scipy.sparse as sp
import numba
import argparse
from ppr_solver import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=17, help='random seed.')
    parser.add_argument('--dataset', default='wiki-talk', help='dateset.')
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha.')
    parser.add_argument('--omega', type=float, default=1.2, help='omega.')
    parser.add_argument('--opt_omega', type=bool, default=True, help='choose optimal or not.')
    parser.add_argument('--test_num', type=int, default=50, help='the number of test nodes.')
    args = parser.parse_args()

    np.random.seed(args.seed)
    all_result = {}
    path = './dataset/'
    test_num = args.test_num
    graph_name = args.dataset
    alpha = args.alpha
    if args.opt_omega:
        alpha = alpha / (2 - alpha)
        mu = (1. - alpha) / (1. + alpha)
        omega = 1. + (mu / (1. + np.sqrt(1. - mu ** 2.))) ** 2.
        alpha = 2 * alpha / (1 + alpha)
    else:
        omega = args.omega
    graph_path = path + graph_name + '/'
    adj_matrix = sp.load_npz(graph_path + graph_name + '_csr-mat.npz')
    indices = adj_matrix.indices
    indptr = adj_matrix.indptr
    n = len(indptr) - 1
    m = len(indices)
    eps = np.power(2.0, 8) / n
    mineps = np.power(2.0, -32) / n
    degree = np.array(adj_matrix.sum(1)).flatten()
    s_nodes = np.random.randint(n, size=test_num)
    s_nodes = np.array(list(np.unique(s_nodes).astype(np.int32)))
    while True:
        if eps >= 1 / degree.max():
            eps /= 2
            continue
        graph_result = solve_a_graph_all(n, indptr, indices, degree, alpha, eps, omega, graph_name, s_nodes)
        all_result[eps] = graph_result
        np.save('./results/' + graph_name + '_ppr_exp_ratio_result.npy', all_result)
        eps /= 2
        if eps < mineps or (graph_result[4] / graph_result[2]).mean() < 0.8:
            break

if __name__ == '__main__':
    main()
