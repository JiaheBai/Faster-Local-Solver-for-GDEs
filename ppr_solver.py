import time
import numpy as np
import cupy as cp
import numba
import os
import torch
import torch_sparse
from numpy import bool_
from numpy import int64
from numpy import float64
from numba import njit
from numba import objmode
from numba.typed import List, Dict
import gc

list_type = numba.types.ListType(numba.types.int64)

"""
This module is for personalized PageRank diffusion equation:
        Qx=b, where
            Q = I - (1-alpha)*A D^{-1} 
            b = alpha * e_s
"""


def ppr_gd_standard_gpu(n, indptr, indices, degree, s_node, alpha, eps):
    """
    Local/Global Gradient Descent for Qx=e_s
    """
    # Kernel code
    kernel_code = r'''
            extern "C" __global__
            void csr_mat_sparse_vec(const int* A_indices, const long* A_indptr,
                                    const float* b_data, const int* b_indices, int b_nnz,
                                    float* result, int n) {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                if (i < b_nnz) {
                    int b_index = b_indices[i];
                    float b_value = b_data[i];
                    for (long j = A_indptr[b_index]; j < A_indptr[b_index + 1]; j++) {
                        atomicAdd(&result[A_indices[j]], b_value);
                    }
                }
            }'''
    start = time.perf_counter()
    # sending to GPU
    indptr = cp.asarray(indptr, dtype=cp.int64)
    indices = cp.asarray(indices, dtype=cp.int32)
    degree = cp.asarray(degree, dtype=cp.float32)
    module = cp.RawModule(code=kernel_code)  # Load the kernel
    csr_sparsevec_dot_kernel = module.get_function('csr_mat_sparse_vec')
    oper = 0
    m = cp.sum(degree)  # total number of directed edges
    xt = cp.zeros(n, dtype=cp.float32)
    rt = cp.zeros(n, dtype=cp.float32)
    rt[s_node] = alpha
    eps_vec = eps * alpha * degree
    tmp_rt = cp.zeros(n, dtype=cp.float32)
    vec_ind = cp.arange(0, n, dtype=cp.int32)

    while True:
        # ------ debug time ------
        debug_start = time.perf_counter()
        # ------------------------
        if cp.sum(eps_vec <= cp.abs(rt)) <= 0.:
            break
        oper += (m + n)
        # standard gradient updates
        xt += rt
        tmp_rt[:] = 0.0
        threads_per_block = 1024
        blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
        csr_sparsevec_dot_kernel((blocks_per_grid,), (threads_per_block,),
                                 (indices, indptr, (1. - alpha) * rt / degree, vec_ind, n, tmp_rt, n))
        rt[:] = tmp_rt
    algo_time = time.perf_counter() - start
    xt = cp.asnumpy(xt)
    rt = cp.asnumpy(rt)
    return xt, rt, oper, algo_time


def ppr_gd_local_gpu(n, indptr, indices, degree, s_node, alpha, eps):
    """
    Local/Global Gradient Descent for Qx=e_s
    """
    # Kernel code
    kernel_code = r'''
        extern "C" __global__
        void csr_mat_sparse_vec(const int* A_indices, const long* A_indptr,
                                const float* b_data, const int* b_indices, int b_nnz,
                                float* result, int n) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i < b_nnz) {
                int b_index = b_indices[i];
                float b_value = b_data[i];
                for (long j = A_indptr[b_index]; j < A_indptr[b_index + 1]; j++) {
                    atomicAdd(&result[A_indices[j]], b_value);
                }
            }
        }'''
    start = time.perf_counter()
    # sending to GPU
    indptr = cp.asarray(indptr, dtype=cp.int64)
    indices = cp.asarray(indices, dtype=cp.int32)
    degree = cp.asarray(degree, dtype=cp.float32)
    module = cp.RawModule(code=kernel_code)  # Load the kernel
    csr_sparsevec_dot_kernel = module.get_function('csr_mat_sparse_vec')

    xt = cp.zeros(n, dtype=cp.float32)
    rt = cp.zeros(n, dtype=cp.float32)
    rt[s_node] = alpha
    eps_vec = eps * alpha * degree
    tmp_rt = cp.zeros(n, dtype=cp.float32)

    # queue data structure
    queue = cp.zeros(n, dtype=cp.int32)
    queue[0] = s_node
    rear = 1
    oper = 0

    while True:
        # ------ debug time ------
        oper += cp.sum(degree[queue[:rear]])
        # ------------------------
        # queue is empty now, quit
        if rear == 0:
            break
        # local gradient updates
        # updates for current iteration from queue
        sp_vec_data = (1. - alpha) * rt[queue[:rear]] / degree[queue[:rear]]
        xt[queue[:rear]] += rt[queue[:rear]]
        rt[queue[:rear]] = 0.
        tmp_rt[:] = 0.
        threads_per_block = 1024
        blocks_per_grid = (rear + threads_per_block - 1) // threads_per_block
        csr_sparsevec_dot_kernel((blocks_per_grid,), (threads_per_block,),
                                 (indices, indptr, sp_vec_data, queue[:rear], rear, tmp_rt, n))
        rt[:] += tmp_rt
        non_zeros = cp.nonzero(eps_vec <= cp.abs(rt))[0]
        rear = len(non_zeros)
        if rear != 0:
            queue[:rear] = non_zeros
    algo_time = time.perf_counter() - start
    xt = cp.asnumpy(xt)
    rt = cp.asnumpy(rt)
    return xt, rt, oper, algo_time


def solve_a_graph_gpu(n, indptr, indices, degree, alpha, eps, s_nodes):
    test_num = len(s_nodes)
    local_algo_times = np.zeros(test_num)
    global_algo_times = np.zeros(test_num)
    # pk = np.zeros(test_num)
    for ii in range(test_num):
        s_node = s_nodes[ii]
        xt, rt, oper, algo_time = ppr_gd_standard_gpu(n, indptr, indices, degree, s_node, alpha, eps)
        global_algo_times[ii] = algo_time
        # pk[ii] = ((np.sum(np.power(xt * n, 2))) ** 2) / (np.sum(np.power(xt * n, 4)))
    for ii in range(test_num):
        s_node = s_nodes[ii]
        xt, rt, oper, algo_time = ppr_gd_local_gpu(n, indptr, indices, degree, s_node, alpha, eps)
        local_algo_times[ii] = algo_time
        # pk[ii] = ((np.sum(np.power(xt * n, 2))) ** 2) / (np.sum(np.power(xt * n, 4)))
    return s_nodes, local_algo_times, global_algo_times


def ppr_gd_torch(adj_mat, n, degree, deg_mat, alpha, eps, s_node, device):
    """
    Global Gradient Descent for Qx=e_s
    """
    torch.cuda.empty_cache()
    st = time.time()
    eps_vec = (eps * alpha * degree).reshape(n, 1)
    eps_vec = eps_vec.to(device)
    xt = torch.zeros((n, 1), dtype=torch.float64).to(device)
    rt = torch.zeros((n, 1), dtype=torch.float64).to(device)
    rt[s_node] = alpha
    adj_mat = adj_mat.to(device)
    while True:
        if torch.sum(eps_vec <= torch.abs(rt)) <= 0.:  # or np.abs(errs[-1]) <= 0.:
            break
        xt += rt
        rt = deg_mat @ rt
        rt = adj_mat @ rt
    xt = xt.cpu().numpy()
    rt = rt.cpu().numpy()
    algo_time = time.time() - st
    return xt, rt, algo_time


def solve_a_graph_torch(n, indptr, indices, degree, alpha, eps, s_nodes, device):
    test_num = len(s_nodes)
    algo_times = np.zeros(test_num)
    deg_data = (1 - alpha) / np.array(degree, dtype=np.float64)
    deg_indptr = np.array(list(range(n + 1)), dtype=np.int64)
    deg_indices = np.array(list(range(n)), dtype=np.int64)
    data = np.array([1, ] * len(indices), dtype=np.int8)
    adj_mat = torch.sparse_csr_tensor(indptr, indices, data, dtype=torch.float64).to(device)
    deg_mat = torch.sparse_csr_tensor(deg_indptr, deg_indices, deg_data, dtype=torch.float64).to(device)
    degree = torch.tensor(degree)
    for ii in range(test_num):
        s_node = s_nodes[ii]
        xt, rt, algo_time = ppr_gd_torch(adj_mat, n, degree, deg_mat, alpha, eps, s_node, device)
        algo_times[ii] = algo_time
    return algo_times.mean()


@njit(cache=True)
def ppr_gd(n, indptr, indices, degree, s_node, alpha, eps, local, opt_x, debug=False):
    """
    Local/Global Gradient Descent for Qx=e_s
    """
    assert eps * degree[s_node] <= 1.
    with objmode(start='f8'):
        start = time.perf_counter()
    m = np.sum(degree)  # total number of directed edges
    xt = np.zeros(n, dtype=float64)
    rt = np.zeros(n, dtype=float64)
    rt[s_node] = alpha
    tmp_rt = np.zeros(n, dtype=float64)
    eps_vec = eps * alpha * degree
    if debug:
        # debug info
        errs = []
        opers = []
        ct_xt = []
        ct_rt = []
        vol_st = []
        vol_it = []
        gamma_t = []
        op_time = np.float64(0.)
    else:
        oper = 0

    # LocGD method
    if local:
        # queue data structure
        queue = np.zeros(n, dtype=int64)
        queue[0] = s_node
        q_mark = np.zeros(n, dtype=bool_)
        q_mark[s_node] = True
        rear = 1
        st = np.zeros(n, dtype=int64)
        st[0] = s_node

        while True:
            if debug:
                # ------ debug time ------
                with objmode(debug_start='f8'):
                    debug_start = time.perf_counter()
                if opt_x is not None:
                    errs.append(np.sum(np.abs(xt - opt_x)))
                else:
                    errs.append(np.infty)
                opers.append(np.sum(degree[queue[:rear]]))
                ct_xt.append(np.count_nonzero(xt))
                ct_rt.append(np.count_nonzero(rt))
                vol_st.append(np.sum(degree[queue[:rear]]))
                vol_it.append(np.sum(degree[np.nonzero(rt)]))
                gamma_t.append(np.sum(np.abs(rt[queue[:rear]])) / np.sum(np.abs(rt)))
                with objmode(op_time='f8'):
                    op_time += (time.perf_counter() - debug_start)
            else:
                oper += np.sum(degree[queue[:rear]])
            # ------------------------
            if rear == 0:
                break
            # local gradient updates
            if rear < n / 4:
                st[:rear] = queue[:rear]
            else:  # use continuous memory
                st[:rear] = np.nonzero(q_mark)[0]
            tmp_rt[:rear] = rt[st[:rear]]
            q_mark[st[:rear]] = False

            xt[st[:rear]] += tmp_rt[:rear]
            rt[st[:rear]] -= tmp_rt[:rear]
            tmp_rear = rear
            rear = 0
            for ind in range(tmp_rear):
                u = st[ind]
                val = (1. - alpha) * tmp_rt[ind] / degree[u]
                for v in indices[indptr[u]:indptr[u + 1]]:
                    rt[v] += val
                    if not q_mark[v] and eps_vec[v] <= np.abs(rt[v]):
                        queue[rear] = v
                        q_mark[v] = True
                        rear += 1
    # standard GD method
    else:
        while True:
            if np.sum(eps_vec <= np.abs(rt)) <= 0.:  # or np.abs(errs[-1]) <= 0.:
                break
            # standard gradient updates
            xt += rt
            tmp_rt[:] = 0.
            for u in np.arange(n):
                val = (1. - alpha) * rt[u] / degree[u]
                for v in indices[indptr[u]:indptr[u + 1]]:
                    tmp_rt[v] += val
            rt[:] = tmp_rt
            if debug:
                # ------ debug time ------
                with objmode(debug_start='f8'):
                    debug_start = time.perf_counter()
                if opt_x is not None:
                    errs.append(np.sum(np.abs(xt - opt_x)))
                else:
                    errs.append(np.infty)
                opers.append(n + m)
                ct_xt.append(np.count_nonzero(xt))
                ct_rt.append(np.count_nonzero(rt))
                vol_st.append(np.sum(degree[np.nonzero(rt)]))
                vol_it.append(np.sum(degree[np.nonzero(rt)]))
                gamma_t.append(1.)  # gamma_t is always unit.
                with objmode(op_time='f8'):
                    op_time += (time.perf_counter() - debug_start)
                # ------------------------
            else:
                oper += n + m
    with objmode(run_time='f8'):
        run_time = time.perf_counter() - start
    if debug:
        algo_time = run_time - op_time
    else:
        algo_time = run_time
    # return xt, rt, errs, opers, ct_xt, ct_rt, vol_st, vol_it, gamma_t, run_time, algo_time
    return xt, rt, oper, algo_time


@njit(cache=True)
def ppr_sor(n, indptr, indices, degree, s_node, alpha, eps, omega, local, opt_x, debug=False):
    assert eps * degree[s_node] <= 1.
    with objmode(start='f8'):
        start = time.perf_counter()
    m = np.sum(degree)  # total number of directed edges
    xt = np.zeros(n, dtype=float64)
    rt = np.zeros(n, dtype=float64)
    rt[s_node] = alpha
    eps_vec = eps * alpha * degree
    if debug:
        # debug info
        errs = []
        opers = []
        ct_xt = []
        ct_rt = []
        vol_st = []
        vol_it = []
        beta_t = []
        op_time = np.float64(0.)
    oper = 0.
    if local:
        front = int64(0)
        rear = int64(0)
        q_max = n + 2
        queue = np.zeros(q_max, dtype=int64)
        q_mark = np.zeros(q_max, dtype=bool_)
        # put iteration flag into queue
        queue[rear] = n
        rear = (rear + 1) % q_max
        q_mark[n] = True
        # put s_node into queue.
        queue[rear] = s_node
        rear = (rear + 1) % q_max
        q_mark[s_node] = True
        if debug:
            beta_num = rt[s_node]
            beta_t.append(rt[s_node])
        while True:
            # next candidate node, it could be either active, inactive, or flag
            u = queue[front]
            front = (front + 1) % q_max
            q_mark[u] = False
            # case 1: u is local iteration flag
            if u == n:  # one local iteration
                if debug:
                    # ------ debug time ------
                    with objmode(debug_start='f8'):
                        debug_start = time.perf_counter()
                    if opt_x is not None:
                        errs.append(np.sum(np.abs(xt - opt_x)))
                    else:
                        errs.append(np.infty)  # fakes
                    opers.append(oper)
                    ct_xt.append(np.count_nonzero(xt))
                    ct_rt.append(np.count_nonzero(rt))
                    vol_st.append(oper)
                    vol_it.append(np.sum(degree[np.nonzero(rt)]))
                    beta_t[-1] = beta_num / beta_t[-1]
                    oper = 0.
                    beta_num = 0.
                    if (rear - front) != 0:
                        beta_t.append(np.sum(np.abs(rt)))
                    with objmode(op_time='f8'):
                        op_time += (time.perf_counter() - debug_start)
                # ------------------------
                if (rear - front) == 0:
                    break
                queue[rear] = n
                rear = (rear + 1) % q_max
                q_mark[u] = True
                continue
            # case 2: u is inactive
            if eps_vec[u] > np.abs(rt[u]):
                continue
            # case 3: u is active
            oper += degree[u]
            if debug:
                beta_num += np.abs(rt[u])
            res = omega * rt[u]
            xt[u] += res
            rt[u] -= res
            push_res = (1. - alpha) * res / degree[u]
            for v in indices[indptr[u]:indptr[u] + degree[u]]:
                rt[v] += push_res
                if not q_mark[v] and eps_vec[v] <= np.abs(rt[v]):
                    queue[rear] = v
                    rear = (rear + 1) % q_max
                    q_mark[v] = True
            # put active u back again if r[u] is large enough
            if not q_mark[u] and eps_vec[u] <= np.abs(rt[u]):
                queue[rear] = u
                rear = (rear + 1) % q_max
                q_mark[u] = True
    else:
        while True:
            for u in range(n):
                res = omega * rt[u]
                xt[u] += res
                rt[u] -= res
                push_res = (1. - alpha) * res / degree[u]
                for v in indices[indptr[u]:indptr[u] + degree[u]]:
                    rt[v] += push_res
            if debug:
                with objmode(debug_start='f8'):
                    debug_start = time.perf_counter()
                if opt_x is not None:
                    errs.append(np.sum(np.abs(xt - opt_x)))
                else:
                    errs.append(np.infty)
                opers.append(n + m)
                ct_xt.append(np.count_nonzero(xt))
                ct_rt.append(np.count_nonzero(rt))
                vol_st.append(np.sum(degree[np.nonzero(rt)]))
                vol_it.append(np.sum(degree[np.nonzero(rt)]))
                beta_t.append(1.)  # beta_t is always unit.
                with objmode(op_time='f8'):
                    op_time += (time.perf_counter() - debug_start)
                # ------------------------
            else:
                oper += n + m
            if np.sum(eps_vec <= np.abs(rt)) <= 0.:  # or np.abs(errs[-1]) <= 0.:
                break
    with objmode(run_time='f8'):
        run_time = time.perf_counter() - start
    if debug:
        algo_time = run_time - op_time
    else:
        algo_time = run_time
    # return xt, rt, errs, opers, ct_xt, ct_rt, vol_st, vol_it, beta_t, run_time, algo_time
    return xt, rt, oper, algo_time


@njit(cache=True, parallel=True)
def solve_a_graph_omega(n, indptr, indices, degree, alpha, eps, omega, graph_name, s_nodes):
    test_num = len(s_nodes)
    if graph_name == 'ogbn-papers100M' or graph_name == 'com-friendster':
        batch_num = 17
    else:
        batch_num = 50

    local_sor_opers = np.zeros(test_num, dtype=np.int64)
    local_sor_algo_times = np.zeros(test_num)
    global_sor_opers = np.zeros(test_num, dtype=np.int64)
    global_sor_algo_times = np.zeros(test_num)

    local_appr_opers = np.zeros(test_num, dtype=np.int64)
    local_appr_algo_times = np.zeros(test_num)
    global_appr_opers = np.zeros(test_num, dtype=np.int64)
    global_appr_algo_times = np.zeros(test_num)

    local_gd_opers = np.zeros(test_num, dtype=np.int64)
    local_gd_algo_times = np.zeros(test_num)
    global_gd_opers = np.zeros(test_num, dtype=np.int64)
    global_gd_algo_times = np.zeros(test_num)

    now_batch = 0
    while True:
        for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            xt, rt, oper, algo_time = ppr_sor(n, indptr, indices, degree, s_node, alpha, eps, omega, local=True,
                                              opt_x=None)
            local_sor_opers[ii] = oper
            local_sor_algo_times[ii] = algo_time

        for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            xt, rt, oper, algo_time = ppr_sor(n, indptr, indices, degree, s_node, alpha, eps, omega=1.0, local=True,
                                              opt_x=None)
            local_appr_opers[ii] = oper
            local_appr_algo_times[ii] = algo_time
        now_batch += 1
        if batch_num * now_batch >= test_num:
            break
    return (s_nodes, local_sor_opers, local_sor_algo_times, global_sor_opers,
            global_sor_algo_times, local_appr_opers, local_appr_algo_times,
            global_appr_opers, global_appr_algo_times, local_gd_opers,
            local_gd_algo_times, global_gd_opers, global_gd_algo_times)


@njit(cache=True, parallel=True)
def solve_a_graph_all(n, indptr, indices, degree, alpha, eps, omega, graph_name, s_nodes):
    test_num = len(s_nodes)
    if graph_name == 'ogbn-papers100M' or graph_name == 'com-friendster':
        batch_num = 25
    else:
        batch_num = 50

    local_sor_opers = np.zeros(test_num, dtype=np.int64)
    local_sor_algo_times = np.zeros(test_num)
    global_sor_opers = np.zeros(test_num, dtype=np.int64)
    global_sor_algo_times = np.zeros(test_num)

    local_appr_opers = np.zeros(test_num, dtype=np.int64)
    local_appr_algo_times = np.zeros(test_num)
    global_appr_opers = np.zeros(test_num, dtype=np.int64)
    global_appr_algo_times = np.zeros(test_num)

    local_gd_opers = np.zeros(test_num, dtype=np.int64)
    local_gd_algo_times = np.zeros(test_num)
    global_gd_opers = np.zeros(test_num, dtype=np.int64)
    global_gd_algo_times = np.zeros(test_num)

    now_batch = 0
    while True:

        for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            xt, rt, oper, algo_time = ppr_gd(n, indptr, indices, degree, s_node, alpha, eps, local=False, opt_x=None)
            global_gd_opers[ii] = oper
            global_gd_algo_times[ii] = algo_time

        for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            xt, rt, oper, algo_time = ppr_sor(n, indptr, indices, degree, s_node, alpha, eps, omega, local=True,
                                              opt_x=None)
            local_sor_opers[ii] = oper
            local_sor_algo_times[ii] = algo_time

        for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            xt, rt, oper, algo_time = ppr_sor(n, indptr, indices, degree, s_node, alpha, eps, omega, local=False,
                                              opt_x=None)
            global_sor_opers[ii] = oper
            global_sor_algo_times[ii] = algo_time
        for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            xt, rt, oper, algo_time = ppr_sor(n, indptr, indices, degree, s_node, alpha, eps, omega=1.0, local=True,
                                              opt_x=None)
            local_appr_opers[ii] = oper
            local_appr_algo_times[ii] = algo_time
        for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            xt, rt, oper, algo_time = ppr_sor(n, indptr, indices, degree, s_node, alpha, eps, omega=1.0, local=False,
                                              opt_x=None)
            global_appr_opers[ii] = oper
            global_appr_algo_times[ii] = algo_time
        for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            xt, rt, oper, algo_time = ppr_gd(n, indptr, indices, degree, s_node, alpha, eps, local=True, opt_x=None)
            local_gd_opers[ii] = oper
            local_gd_algo_times[ii] = algo_time
        now_batch += 1
        if batch_num * now_batch >= test_num:
            break
    return (s_nodes, local_sor_opers, local_sor_algo_times, global_sor_opers,
            global_sor_algo_times, local_appr_opers, local_appr_algo_times,
            global_appr_opers, global_appr_algo_times, local_gd_opers,
            local_gd_algo_times, global_gd_opers, global_gd_algo_times)


@njit(cache=True)
def ppr_cheby(n, indptr, indices, degree, s_node, alpha, eps, local):
    with objmode(start='f8'):
        start = time.perf_counter()
    m = np.sum(degree)  # total number of directed edges
    delta_xt = np.zeros(n, dtype=np.float64)
    xt = np.zeros(n, dtype=np.float64)
    rt = np.zeros(n, dtype=np.float64)
    rt[s_node] = alpha
    xt[:] = rt
    delta_xt[:] = xt
    eps_vec = eps * alpha * degree
    delta_t = (1. - alpha)
    tmp_vec = np.zeros(n, dtype=np.float64)
    tmp_vec[:] = 0.
    for u in [s_node]:
        val = (1. - alpha) * rt[u] / degree[u]
        for v in indices[indptr[u]:indptr[u + 1]]:
            tmp_vec[v] += val
    rt[:] = tmp_vec
    oper = 0
    if not local:
        while True:
            if np.sum(eps_vec <= np.abs(rt)) <= 0.:
                break
            oper += n + m
            delta_t = 1. / (2. / (1. - alpha) - delta_t)
            beta = 2. * delta_t / (1. - alpha)
            tmp_vec[:] = beta * rt + (beta - 1.) * delta_xt
            xt[:] += tmp_vec
            delta_xt[:] = tmp_vec
            tmp_vec[:] = 0.
            for u in np.arange(n):
                val = (1. - alpha) * delta_xt[u] / degree[u]
                nei_u = indices[indptr[u]:indptr[u + 1]]
                tmp_vec[nei_u] += val
            rt[:] += tmp_vec - delta_xt
    else:
        # ----------------------
        # queue data structure
        queue = np.zeros(n, dtype=np.int64)
        q_mark = np.zeros(n, dtype=np.bool_)
        queue_pre = np.zeros(n, dtype=np.int64)
        rear = 0
        for v in indices[indptr[u]:indptr[u + 1]]:
            if not q_mark[v] and eps_vec[v] <= np.abs(rt[v]):
                queue[rear] = v
                q_mark[v] = True
                rear += 1
        # ----------------------
        while True:
            if rear == 0:
                break
            oper += np.sum(degree[queue[:rear]])
            delta_t = 1. / (2. / (1. - alpha) - delta_t)
            beta = 2. * delta_t / (1. - alpha)
            # updates for current iteration from queue
            if rear > n / 4:
                queue[:rear] = np.nonzero(q_mark)[0]
            for ind, u in enumerate(queue[:rear]):
                tmp = beta * rt[u] + (beta - 1.) * delta_xt[u]
                xt[u] += tmp
                delta_xt[u] = tmp
                q_mark[u] = False
                queue_pre[ind] = u
            rear_pre = rear
            rear = 0
            for (ind, u) in enumerate(queue_pre[:rear_pre]):
                val = (1. - alpha) * delta_xt[u] / degree[u]
                for v in indices[indptr[u]:indptr[u + 1]]:
                    rt[v] += val
                    if not q_mark[v] and eps_vec[v] <= np.abs(rt[v]):
                        queue[rear] = v
                        q_mark[v] = True
                        rear += 1
                rt[u] += -delta_xt[u]
    with objmode(run_time='f8'):
        run_time = time.perf_counter() - start
    algo_time = run_time
    return xt, rt, oper, algo_time


@njit(cache=True, parallel=True)
def solve_a_graph_cheby(n, indptr, indices, degree, alpha, eps, omega, graph_name, s_nodes):
    test_num = len(s_nodes)
    if graph_name == 'ogbn-papers100M' or graph_name == 'com-friendster':
        batch_num = 25
    else:
        batch_num = 50

    local_opers = np.zeros(test_num, dtype=np.int64)
    local_algo_times = np.zeros(test_num)
    global_opers = np.zeros(test_num, dtype=np.int64)
    global_algo_times = np.zeros(test_num)
    now_batch = 0
    while True:
        for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            xt, rt, oper, algo_time = ppr_cheby(n, indptr, indices, degree, s_node, alpha, eps, local=True)
            local_opers[ii] = oper
            local_algo_times[ii] = algo_time

        for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            xt, rt, oper, algo_time = ppr_cheby(n, indptr, indices, degree, s_node, alpha, eps, local=False)
            global_opers[ii] = oper
            global_algo_times[ii] = algo_time

        now_batch += 1
        if batch_num * now_batch >= test_num:
            break
    return (s_nodes, local_opers, local_algo_times, global_opers, global_algo_times)


@njit(cache=True, parallel=True)
def init_a_graph(n, indptr, indices, degree, alpha, eps, omega, graph_name, s_nodes):
    test_num = len(s_nodes)
    if graph_name == 'ogbn-papers100M' or graph_name == 'com-friendster':
        batch_num = 17
    else:
        batch_num = 50
    opers = np.zeros(test_num, dtype=np.int64)
    algo_times = np.zeros(test_num)
    xs = np.zeros((test_num, n))
    rs = np.zeros((test_num, n))
    now_batch = 0
    while True:
        for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            xt, rt, oper, algo_time = ppr_sor(n, indptr, indices, degree, s_node, alpha, eps, omega,
                                              local=True, opt_x=None, debug=False)
            opers[ii] = oper
            algo_times[ii] = algo_time
            xs[ii, :] = xt
            rs[ii, :] = rt
        now_batch += 1
        if batch_num * now_batch >= test_num:
            break
    return opers, algo_times, xs, rs


@njit(cache=True)
def updeta_graph(indptr, indices, degree, us, vs, onlyinsert = True):
    insert_neighbor = Dict.empty(
        key_type=numba.types.int64,
        value_type=list_type
    )
    delete_neighbor = Dict.empty(
        key_type=numba.types.int64,
        value_type=list_type
    )
    for edge_idx in range(len(us)):
        insertion = True
        u = us[edge_idx]
        v = vs[edge_idx]
        for i in range(indptr[u], indptr[u] + degree[u]):
            if v == indices[i]:
                insertion = False
                if onlyinsert or degree[u] == 1 or degree[v] == 1:
                    break
                indices[i] = indices[indptr[u] + degree[u] - 1]
                for k in range(indptr[v], indptr[v] + degree[v]):
                    if u == indices[k]:
                        indices[k] = indices[indptr[v] + degree[v] - 1]
                degree[u] -= 1
                degree[v] -= 1
                if u not in delete_neighbor:
                    delete_neighbor[u] = List.empty_list(numba.types.int64)
                delete_neighbor[u].append(v)
                if v not in delete_neighbor:
                    delete_neighbor[v] = List.empty_list(numba.types.int64)
                delete_neighbor[v].append(u)
                break
        if insertion:
            indices[indptr[u] + degree[u]] = v
            indices[indptr[v] + degree[v]] = u
            degree[u] += 1
            degree[v] += 1
            if u not in insert_neighbor:
                insert_neighbor[u] = List.empty_list(numba.types.int64)
            insert_neighbor[u].append(v)
            if v not in insert_neighbor:
                insert_neighbor[v] = List.empty_list(numba.types.int64)
            insert_neighbor[v].append(u)
    return insert_neighbor, delete_neighbor


@njit(cache=True)
def update_ppr(n, indptr, indices, degree, old_degree, p, r, alpha, queue, front, rear, q_mark, eps_vec,
                          insert_neighbor, delete_neighbor, omega):
    with objmode(time1='f8'):
        time1 = time.perf_counter()
    oper = int64(0)
    oper += 2 * len(insert_neighbor)
    for u in insert_neighbor:
        p[u] *= degree[u] / (old_degree[u])
        r[u] += p[u] * (old_degree[u] - degree[u]) / degree[u]
    for u in insert_neighbor:
        dr = 0.
        oper += len(insert_neighbor[u])
        for v in insert_neighbor[u]:
            dr += (1 - alpha) * p[v] / (degree[v])
        if u in delete_neighbor:
            oper += len(delete_neighbor[u])
            for v in delete_neighbor[u]:
                dr -= (1 - alpha) * p[v] / degree[v]
        r[u] += dr
        if np.abs(r[u]) >= eps_vec[u]:
            rear = (rear + 1) % (n + 2)
            queue[rear] = u
            q_mark[u] = True
    while rear != front:
        front = (front + 1) % (n + 2)
        u = queue[front]
        q_mark[u] = False
        if np.abs(r[u]) < eps_vec[u]:
            continue
        if (degree[u] == 1 and indices[indptr[u]] == u) or (degree[u] == 0):
            p[u] = r[u]
            r[u] = 0
            continue
        res_u = omega * r[u]
        p[u] += res_u
        push_u = (1. - alpha) * res_u / degree[u]
        r[u] -= res_u
        oper += degree[u]
        for v in indices[indptr[u]:indptr[u] + degree[u]]:
            r[v] += push_u
            if (q_mark[v] == False) and np.abs(r[v]) >= eps_vec[v]:
                rear = (rear + 1) % (n + 2)
                queue[rear] = v
                q_mark[v] = True
        if (q_mark[u] == False) and np.abs(r[u]) >= eps_vec[u]:
            rear = (rear + 1) % (n + 2)
            queue[rear] = u
            q_mark[u] = True
    with objmode(time2='f8'):
        time2 = time.perf_counter() - time1
    runtime = time2
    return runtime, oper


@njit(cache=True, parallel=True)
def update_ppr_parallel(n, indptr, indices, degree, alpha, eps, omega, P, R, us, vs, batchsize=50):
    old_degree = degree.copy()
    feanum = len(R)
    runtimes = np.zeros(feanum)
    opers = np.zeros(feanum, dtype=np.int64)
    insert_neighbor, delete_neighbor = updeta_graph(indptr, indices, degree, us, vs, onlyinsert=False)
    eps_vec = alpha * eps * degree
    nowbatch = 0
    while True:
        for fea in numba.prange(nowbatch * batchsize, min((nowbatch + 1) * batchsize, feanum)):
            queue = np.zeros(n + 2, dtype=np.int64)
            front, rear = np.int64(0), np.int64(0)
            p = P[fea]
            r = R[fea]
            q_mark = np.zeros(n)
            runtime, oper = update_ppr(n, indptr, indices, degree, old_degree, p, r, alpha, queue, front, rear, q_mark, eps_vec,
                          insert_neighbor, delete_neighbor, omega)
            R[fea, :] = r
            P[fea, :] = p
            runtimes[fea] = runtime
            opers[fea] = oper
        nowbatch += 1
        if nowbatch * batchsize >= feanum:
            break
    return opers, runtimes, P, R