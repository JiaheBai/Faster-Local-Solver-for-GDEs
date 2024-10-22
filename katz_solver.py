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
import gc


"""
This module is for personalized PageRank diffusion equation:
        Qx=b, where
            Q = I - (1-alpha)*A D^{-1} 
            b = alpha * e_s
"""


def katz_gd_standard_gpu(n, indptr, indices, degree, s_node, alpha, eps):
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
    rt[s_node] = 1.0
    eps_vec = eps * degree
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
                                 (indices, indptr, alpha * rt, vec_ind, n, tmp_rt, n))
        rt[:] = tmp_rt
    algo_time = time.perf_counter() - start
    xt = cp.asnumpy(xt)
    rt = cp.asnumpy(rt)
    xt[s_node] -= 1.0
    return xt, rt, oper, algo_time


def katz_gd_local_gpu(n, indptr, indices, degree, s_node, alpha, eps):
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
    rt[s_node] = 1.
    eps_vec = eps * degree
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
        sp_vec_data = alpha * rt[queue[:rear]]
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
    xt[s_node] -= 1.0
    return xt, rt, oper, algo_time


def solve_a_graph_gpu(n, indptr, indices, degree, alpha, eps, s_nodes):
    test_num = len(s_nodes)
    local_algo_times = np.zeros(test_num)
    global_algo_times = np.zeros(test_num)
    # pk = np.zeros(test_num)
    for ii in range(test_num):
        s_node = s_nodes[ii]
        xt, rt, oper, algo_time = katz_gd_standard_gpu(n, indptr, indices, degree, s_node, alpha, eps)
        global_algo_times[ii] = algo_time
        # pk[ii] = ((np.sum(np.power(xt * n, 2))) ** 2) / (np.sum(np.power(xt * n, 4)))
    for ii in range(test_num):
        s_node = s_nodes[ii]
        xt, rt, oper, algo_time = katz_gd_local_gpu(n, indptr, indices, degree, s_node, alpha, eps)
        local_algo_times[ii] = algo_time
        # pk[ii] = ((np.sum(np.power(xt * n, 2))) ** 2) / (np.sum(np.power(xt * n, 4)))
    return s_nodes, local_algo_times, global_algo_times


@njit(cache=True)
def katz_gd(n, indptr, indices, degree, s_node, alpha, eps, local):
    """
    Local/Global Gradient Descent for Qx=e_s
    """
    with objmode(start='f8'):
        start = time.perf_counter()
    m = np.sum(degree)  # total number of directed edges
    xt = np.zeros(n, dtype=float64)
    rt = np.zeros(n, dtype=float64)
    rt[s_node] = 1.
    tmp_rt = np.zeros(n, dtype=float64)
    eps_vec = eps * degree
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
                val = alpha * tmp_rt[ind]
                for v in indices[indptr[u]:indptr[u + 1]]:
                    rt[v] += val
                    if not q_mark[v] and eps_vec[v] <= np.abs(rt[v]):
                        queue[rear] = v
                        q_mark[v] = True
                        rear += 1
            with objmode(run_time='f8'):
                run_time = time.perf_counter() - start
            algo_time = run_time
            if algo_time > 14400:
                break
    # standard GD method
    else:
        while True:
            if np.sum(eps_vec <= np.abs(rt)) <= 0.:  # or np.abs(errs[-1]) <= 0.:
                break
            # standard gradient updates
            xt += rt
            tmp_rt[:] = 0.
            for u in np.arange(n):
                val = alpha * rt[u]
                for v in indices[indptr[u]:indptr[u + 1]]:
                    tmp_rt[v] += val
            rt[:] = tmp_rt
            oper += n + m
            with objmode(run_time='f8'):
                run_time = time.perf_counter() - start
            algo_time = run_time
            if algo_time > 14400:
                break
    xt[s_node] -= 1.
    with objmode(run_time='f8'):
        run_time = time.perf_counter() - start
    algo_time = run_time
    # return xt, rt, errs, opers, ct_xt, ct_rt, vol_st, vol_it, gamma_t, run_time, algo_time
    return xt, rt, oper, algo_time


@njit(cache=True)
def katz_sor(n, indptr, indices, degree, s_node, alpha, eps, omega, local):
    with objmode(start='f8'):
        start = time.perf_counter()
    m = np.sum(degree)  # total number of directed edges
    xt = np.zeros(n, dtype=float64)
    rt = np.zeros(n, dtype=float64)
    rt[s_node] = 1.0
    eps_vec = eps * degree
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
        while True:
            # next candidate node, it could be either active, inactive, or flag
            u = queue[front]
            front = (front + 1) % q_max
            q_mark[u] = False
            # case 1: u is local iteration flag
            if u == n:  # one local iteration
                # ------------------------
                if (rear - front) == 0:
                    break
                with objmode(run_time='f8'):
                    run_time = time.perf_counter() - start
                algo_time = run_time
                if algo_time > 14400:
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
            res = omega * rt[u]
            xt[u] += res
            rt[u] -= res
            push_res = alpha * res
            for v in indices[indptr[u]:indptr[u + 1]]:
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
                push_res = alpha * res
                for v in indices[indptr[u]:indptr[u + 1]]:
                    rt[v] += push_res
            oper += n + m
            if np.sum(eps_vec <= np.abs(rt)) <= 0.:  # or np.abs(errs[-1]) <= 0.:
                break
            with objmode(run_time='f8'):
                run_time = time.perf_counter() - start
            algo_time = run_time
            if algo_time > 14400:
                break
    xt[s_node] -= 1.
    with objmode(run_time='f8'):
        run_time = time.perf_counter() - start
    algo_time = run_time
    # return xt, rt, errs, opers, ct_xt, ct_rt, vol_st, vol_it, beta_t, run_time, algo_time
    return xt, rt, oper, algo_time


@njit(cache=True)
def katz_cheby(n, indptr, indices, degree, s_node, alpha, l_max, mu_min, eps, local):
    with objmode(start='f8'):
        start = time.perf_counter()
    m = np.sum(degree)  # total number of directed edges
    xt = np.zeros(n, dtype=np.float64)
    xt[s_node] = 2. / (l_max + mu_min)  # x1
    delta_xt = np.zeros(n, dtype=np.float64)
    delta_xt[:] = xt
    rt = np.zeros(n, dtype=np.float64)
    rt[s_node] = 1. - 2. / (l_max + mu_min)
    for u in [s_node]:
        val = 2. * alpha / (l_max + mu_min)
        for v in indices[indptr[u]:indptr[u + 1]]:
            rt[v] += val
    eps_vec = eps * degree
    delta_t = (l_max - mu_min) / (l_max + mu_min)
    tmp_vec = np.zeros(n, dtype=np.float64)
    tmp_vec[:] = 0.
    oper = 0

    if not local:
        while True:
            if np.sum(eps_vec <= np.abs(rt)) <= 0.:
                break
            oper += n + m
            delta_t = 1. / (2. * ((l_max + mu_min) / (l_max - mu_min)) - delta_t)
            beta1 = 4. * delta_t / (l_max - mu_min)
            beta2 = (1. - 2. * delta_t * (l_max + mu_min) / (l_max - mu_min))
            tmp_vec[:] = beta1 * rt - beta2 * delta_xt
            xt[:] += tmp_vec
            delta_xt[:] = tmp_vec
            tmp_vec[:] = 0.
            for u in np.arange(n):
                val = alpha * delta_xt[u]
                nei_u = indices[indptr[u]:indptr[u + 1]]
                tmp_vec[nei_u] += val
            rt[:] = rt + tmp_vec - delta_xt
            with objmode(run_time='f8'):
                run_time = time.perf_counter() - start
            algo_time = run_time
            if algo_time > 14400:
                break
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
            oper += np.sum(degree[queue[:rear]])
            if rear == 0:
                break
            delta_t = 1. / (2. * ((l_max + mu_min) / (l_max - mu_min)) - delta_t)
            beta1 = 4. * delta_t / (l_max - mu_min)
            beta2 = (1. - 2. * delta_t * (l_max + mu_min) / (l_max - mu_min))
            # updates for current iteration from queue
            if rear > n / 4:
                queue[:rear] = np.nonzero(q_mark)[0]
            for ind, u in enumerate(queue[:rear]):
                tmp = beta1 * rt[u] - beta2 * delta_xt[u]
                xt[u] += tmp
                delta_xt[u] = tmp
                q_mark[u] = False
                queue_pre[ind] = u
            rear_pre = rear
            rear = 0
            for (ind, u) in enumerate(queue_pre[:rear_pre]):
                val = alpha * delta_xt[u]
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
            if algo_time > 14400:
                break
    with objmode(run_time='f8'):
        run_time = time.perf_counter() - start
    algo_time = run_time
    return xt, rt, oper, algo_time


@njit(cache=True, parallel=True)
def solve_a_graph_ratio(n, indptr, indices, degree, alpha, eps, omega, graph_name, s_nodes):
    test_num = len(s_nodes)
    # path = "./results/" + graph_name
    local_sor_opers = np.zeros(test_num)
    local_sor_algo_times = np.zeros(test_num)
    global_sor_opers = np.zeros(test_num)
    global_sor_algo_times = np.zeros(test_num)

    local_akatz_opers = np.zeros(test_num)
    local_akatz_algo_times = np.zeros(test_num)
    global_akatz_opers = np.zeros(test_num)
    global_akatz_algo_times = np.zeros(test_num)

    local_gd_opers = np.zeros(test_num)
    local_gd_algo_times = np.zeros(test_num)
    global_gd_opers = np.zeros(test_num)
    global_gd_algo_times = np.zeros(test_num)
    m = degree.sum()

    for ii in numba.prange(test_num):
        s_node = s_nodes[ii]
        # print(s_node)
        '''opt_x, rt, errs, opers, ct_xt, ct_rt, vol_st, vol_it, beta_t, run_time, algo_time = katz_sor(n, indptr, indices,
                                                                                                    degree, s_node,
                                                                                                    alpha, eps, omega,
                                                                                                    local=True,
                                                                                                    opt_x=None)'''

        xt, rt, oper, algo_time = katz_sor(n, indptr, indices, degree, s_node, alpha, eps, omega, local=True)
        # np.savez(path + '/local_sor_snode'+str(s_node)+'.npz', xt=xt, rt=rt, errs=errs, opers=opers, ct_xt=ct_xt, ct_rt=ct_rt,
        #         vol_st=vol_st, vol_it=vol_it, beta_t=beta_t, run_time=run_time, algo_time=algo_time)
        local_sor_opers[ii] = oper
        local_sor_algo_times[ii] = algo_time
        # print(np.abs(rt).max())
        # print(s_node, '1')
        # print(algo_time)
    for ii in numba.prange(test_num):
        s_node = s_nodes[ii]
        xt, rt, oper, algo_time = katz_sor(n, indptr, indices, degree, s_node, alpha, eps, omega, local=False)
        # np.savez(path + '/global_sor_snode'+str(s_node)+'.npz', xt=xt, rt=rt, errs=errs, opers=opers, ct_xt=ct_xt, ct_rt=ct_rt,
        #         vol_st=vol_st, vol_it=vol_it, beta_t=beta_t, run_time=run_time, algo_time=algo_time)
        global_sor_opers[ii] = oper
        global_sor_algo_times[ii] = algo_time
        # print(np.abs(rt).max())
        # print(np.linalg.norm(xt1 - xt2))
        # print(s_node, '2')
        # print(algo_time)
    for ii in numba.prange(test_num):
        s_node = s_nodes[ii]
        xt, rt, oper, algo_time = katz_sor(n, indptr, indices, degree, s_node, alpha, eps, omega=1.0, local=True)
        # np.savez(path + '/local_sor_snode'+str(s_node)+'.npz', xt=xt, rt=rt, errs=errs, opers=opers, ct_xt=ct_xt, ct_rt=ct_rt,
        #         vol_st=vol_st, vol_it=vol_it, beta_t=beta_t, run_time=run_time, algo_time=algo_time)
        local_akatz_opers[ii] = oper
        local_akatz_algo_times[ii] = algo_time
        # print(np.abs(rt).max())
        # print(s_node, '3')
        # print(algo_time)
    for ii in numba.prange(test_num):
        s_node = s_nodes[ii]
        xt, rt, oper, algo_time = katz_sor(n, indptr, indices, degree, s_node, alpha, eps, omega=1.0, local=False)
        # np.savez(path + '/global_sor_snode'+str(s_node)+'.npz', xt=xt, rt=rt, errs=errs, opers=opers, ct_xt=ct_xt, ct_rt=ct_rt,
        #         vol_st=vol_st, vol_it=vol_it, beta_t=beta_t, run_time=run_time, algo_time=algo_time)
        global_akatz_opers[ii] = oper
        global_akatz_algo_times[ii] = algo_time
        # print(np.abs(rt).max())
        # print(np.linalg.norm(xt1 - xt3))
        # print(np.linalg.norm(xt3 - xt4))
        # print(s_node, '4')
        # print(algo_time)
    for ii in numba.prange(test_num):
        s_node = s_nodes[ii]
        xt, rt, oper, algo_time = katz_gd(n, indptr, indices, degree, s_node, alpha, eps, local=True)
        # np.savez(path + '/local_gd_snode'+str(s_node)+'.npz', xt=xt, rt=rt, errs=errs, opers=opers, ct_xt=ct_xt, ct_rt=ct_rt,
        #         vol_st=vol_st, vol_it=vol_it, gamma_t=gamma_t, run_time=run_time, algo_time=algo_time)
        local_gd_opers[ii] = oper
        local_gd_algo_times[ii] = algo_time
        # print(np.abs(rt).max())
        # print(s_node, '5')
        # print(algo_time)
    for ii in numba.prange(test_num):
        s_node = s_nodes[ii]
        xt, rt, oper, algo_time = katz_gd(n, indptr, indices, degree, s_node, alpha, eps, local=False)
        # np.savez(path + '/global_gd_snode'+str(s_node)+'.npz', xt=xt, rt=rt, errs=errs, opers=opers, ct_xt=ct_xt, ct_rt=ct_rt,
        #         vol_st=vol_st, vol_it=vol_it, gamma_t=gamma_t, run_time=run_time, algo_time=algo_time)
        global_gd_opers[ii] = oper
        global_gd_algo_times[ii] = algo_time
        # print(np.abs(rt).max())
        # print(np.linalg.norm(xt1 - xt5))
        # print(np.linalg.norm(xt5 - xt6))
        # print(s_node, '6')
        # print(algo_time)

    return (n + m, (global_sor_opers / local_sor_opers).mean(), (global_sor_algo_times / local_sor_algo_times).mean(),
            (global_akatz_opers / local_akatz_opers).mean(), (global_akatz_algo_times / local_akatz_algo_times).mean(),
            (global_gd_opers / local_gd_opers).mean(), (global_gd_algo_times / local_gd_algo_times).mean(),
            local_sor_opers.mean(), local_sor_algo_times.mean(), global_sor_opers.mean(),
            global_sor_algo_times.mean(), local_akatz_opers.mean(), local_akatz_algo_times.mean(),
            global_akatz_opers.mean(), global_akatz_algo_times.mean(), local_gd_opers.mean(),
            local_gd_algo_times.mean(), global_gd_opers.mean(), global_gd_algo_times.mean())


@njit(cache=True, parallel=True)
def solve_a_graph_pk(n, indptr, indices, degree, alpha, eps, omega, graph_name, s_nodes):
    test_num = len(s_nodes)
    if graph_name == 'ogbn-papers100M' or graph_name == 'com-friendster':
        batch_num = 40
    else:
        batch_num = 60
    local_sor_opers = np.zeros(test_num)
    local_sor_algo_times = np.zeros(test_num)
    pk = np.zeros(test_num)
    m = degree.sum()
    now_batch = 0
    while True:
        for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            xt, rt, oper, algo_time = katz_sor(n, indptr, indices, degree, s_node, alpha, eps, omega, local=True)
            local_sor_opers[ii] = oper
            local_sor_algo_times[ii] = algo_time
            pk[ii] = ((np.sum(np.power(xt * n, 2))) ** 2) / (np.sum(np.power(xt * n, 4)))
        now_batch += 1
        if batch_num * now_batch > test_num:
            break
    return (local_sor_opers, local_sor_algo_times, pk)


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

    local_akatz_opers = np.zeros(test_num, dtype=np.int64)
    local_akatz_algo_times = np.zeros(test_num)
    global_akatz_opers = np.zeros(test_num, dtype=np.int64)
    global_akatz_algo_times = np.zeros(test_num)

    local_gd_opers = np.zeros(test_num, dtype=np.int64)
    local_gd_algo_times = np.zeros(test_num)
    global_gd_opers = np.zeros(test_num, dtype=np.int64)
    global_gd_algo_times = np.zeros(test_num)

    now_batch = 0
    while True:
        for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            xt, rt, oper, algo_time = katz_sor(n, indptr, indices, degree, s_node, alpha, eps, omega, local=True)
            local_sor_opers[ii] = oper
            local_sor_algo_times[ii] = algo_time

        '''for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            xt, rt, oper, algo_time = katz_sor(n, indptr, indices, degree, s_node, alpha, eps, omega=1.0, local=True)
            local_akatz_opers[ii] = oper
            local_akatz_algo_times[ii] = algo_time'''
        now_batch += 1
        if batch_num * now_batch >= test_num:
            break
    return (s_nodes, local_sor_opers, local_sor_algo_times, global_sor_opers,
            global_sor_algo_times, local_akatz_opers, local_akatz_algo_times,
            global_akatz_opers, global_akatz_algo_times, local_gd_opers,
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

    local_akatz_opers = np.zeros(test_num, dtype=np.int64)
    local_akatz_algo_times = np.zeros(test_num)
    global_akatz_opers = np.zeros(test_num, dtype=np.int64)
    global_akatz_algo_times = np.zeros(test_num)

    local_gd_opers = np.zeros(test_num, dtype=np.int64)
    local_gd_algo_times = np.zeros(test_num)
    global_gd_opers = np.zeros(test_num, dtype=np.int64)
    global_gd_algo_times = np.zeros(test_num)

    now_batch = 0
    while True:
        for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            xt, rt, oper, algo_time = katz_sor(n, indptr, indices, degree, s_node, alpha, eps, omega, local=True)
            local_sor_opers[ii] = oper
            local_sor_algo_times[ii] = algo_time
        for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            xt, rt, oper, algo_time = katz_sor(n, indptr, indices, degree, s_node, alpha, eps, omega, local=False)
            global_sor_opers[ii] = oper
            global_sor_algo_times[ii] = algo_time
        for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            xt, rt, oper, algo_time = katz_sor(n, indptr, indices, degree, s_node, alpha, eps, omega=1.0, local=True)
            local_akatz_opers[ii] = oper
            local_akatz_algo_times[ii] = algo_time
        for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            xt, rt, oper, algo_time = katz_sor(n, indptr, indices, degree, s_node, alpha, eps, omega=1.0, local=False)
            global_akatz_opers[ii] = oper
            global_akatz_algo_times[ii] = algo_time
        for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            xt, rt, oper, algo_time = katz_gd(n, indptr, indices, degree, s_node, alpha, eps, local=True)
            local_gd_opers[ii] = oper
            local_gd_algo_times[ii] = algo_time
        for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            xt, rt, oper, algo_time = katz_gd(n, indptr, indices, degree, s_node, alpha, eps, local=False)
            global_gd_opers[ii] = oper
            global_gd_algo_times[ii] = algo_time
        now_batch += 1
        if batch_num * now_batch >= test_num:
            break
    return (s_nodes, local_sor_opers, local_sor_algo_times, global_sor_opers,
            global_sor_algo_times, local_akatz_opers, local_akatz_algo_times,
            global_akatz_opers, global_akatz_algo_times, local_gd_opers,
            local_gd_algo_times, global_gd_opers, global_gd_algo_times)


@njit(cache=True, parallel=True)

def solve_a_graph_cheby(n, indptr, indices, degree, alpha, l_max, mu_min, eps, graph_name, s_nodes):
    test_num = len(s_nodes)
    if graph_name == 'ogbn-papers100M' or graph_name == 'com-friendster':
        batch_num = 17
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
            xt, rt, oper, algo_time = katz_cheby(n, indptr, indices, degree, s_node, alpha, l_max, mu_min, eps, local=True)
            local_opers[ii] = oper
            local_algo_times[ii] = algo_time

        for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            xt, rt, oper, algo_time = katz_cheby(n, indptr, indices, degree, s_node, alpha, l_max, mu_min, eps, local=False)
            global_opers[ii] = oper
            global_algo_times[ii] = algo_time
        now_batch += 1
        if batch_num * now_batch >= test_num:
            break
    return (s_nodes, local_opers, local_algo_times, global_opers, global_algo_times)
