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
    parser.add_argument('--eps', type=float, default=1e-7, help='threshold.')
    parser.add_argument('--omega', type=float, default=1.2, help='omega.')
    parser.add_argument('--opt_omega', type=bool, default=True, help='choose optimal or not.')
    parser.add_argument('--test_num', type=int, default=50, help='the number of test nodes.')
    args = parser.parse_args()

    np.random.seed(args.seed)
    result_names = ['local_sor_opers', 'local_sor_algo_times', 'global_sor_opers',
                    'global_sor_algo_times', 'local_gs_opers', 'local_gs_algo_times',
                    'global_gs_opers', 'global_gs_algo_times', 'local_gd_opers',
                    'local_gd_algo_times', 'global_gd_opers', 'global_gd_algo_times']
    result_cheby_names = ['local_ch_opers', 'local_ch_algo_times', 'global_ch_opers', 'global_ch_algo_times']
    result_gpu_names = ['local_gd_gpu_algo_times', 'global_gd_gpu_algo_times']
    path = './dataset/'
    test_num = args.test_num
    graph_name = args.dataset
    alpha = args.alpha
    eps = args.eps
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
    degree = np.array(adj_matrix.sum(1)).flatten()
    s_nodes = np.random.randint(n, size=test_num)
    s_nodes = np.array(list(np.unique(s_nodes).astype(np.int32)))
    print(graph_name, end=':\n')
    graph_result = solve_a_graph_all(n, indptr, indices, degree, alpha, eps, omega, graph_name, s_nodes)
    for i in range(len(result_names)):
        print(result_names[i], ':', graph_result[i + 1].mean())
    cheby_result = solve_a_graph_cheby(n, indptr, indices, degree, alpha, eps, omega, graph_name, s_nodes)
    for i in range(len(result_cheby_names)):
        print(result_cheby_names[i], ':', cheby_result[i + 1].mean())
    gpu_result = solve_a_graph_gpu(n, indptr, indices, degree, alpha, eps, s_nodes)
    for i in range(len(result_gpu_names)):
        print(result_gpu_names[i], ':', gpu_result[i + 1].mean())

if __name__ == '__main__':
    main()
