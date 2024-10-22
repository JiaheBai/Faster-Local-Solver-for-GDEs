import os
import sys
import math
import time
import random
import pickle
import argparse
import numba
import numpy as np
import collections
from numba import njit
from numba import objmode


@njit(cache=True)
def estimate_coefficient(t, k):
    """ calculate t^{k+1}/(k+1)!, k>=0, t >= 1 """
    assert k >= 0
    assert t >= 1.
    t0 = t
    if k == 0:
        return t0
    for i in range(k):
        t0 = t0 * (t / (i + 2))
    return t0


@njit(cache=True)
def estimate_n(t, eps):
    """
    This function is used to estimate the number of steps t such that
    || D^{-1} h - D^{-1} y|| <= \epsilon,
    where y is first N truncation of Taylor polynomial approximation of h.
    """
    assert t >= 1.
    assert eps > 0.
    k = np.ceil(t - 1)  # k is the parameter N in the paper.
    cur_est = estimate_coefficient(t, k) * ((k + 2.) / (k + 2. - t))
    if cur_est < eps * np.exp(t):
        return k
    est_k = k
    for i in range(k, 1000 * (k + 1)):
        tmp = ((k + 2. - t) * t / ((k + 2.) ** 2.)) * ((k + 3.) / (k + 3. - t))
        cur_est = cur_est * tmp
        est_k += 1
        if cur_est < eps * np.exp(t):
            return est_k
    return np.int64(est_k)


@njit(cache=True)
def hk_taylor_approx(n, indptr, indices, degree, s_node, tau, eps):
    """
    Compute Heat Kernel by using Taylor polynomial approximation.
    It returns a solution xt such that ||D^{-1}xt-D^{-1}h||_1 <= eps
    """
    with objmode(start='f8'):
        start = time.perf_counter()
    assert tau >= 1.  # the temperature of heat kernel
    assert eps >= 0.
    est_n = estimate_n(tau, eps)
    assert est_n >= 0
    cur_iter = np.zeros_like(degree, dtype=np.float64)
    cur_iter[s_node] = 1.
    xt = np.zeros_like(degree, dtype=np.float64)
    xt[:] = xt[:] + cur_iter
    tmp_vec = np.zeros_like(degree, dtype=np.float64)
    for k in range(est_n):
        tmp_vec[:] = 0.
        for u in np.arange(n):
            val = cur_iter[u] / degree[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                tmp_vec[v] += val
        cur_iter = (tau / (k + 1.)) * tmp_vec
        xt[:] = xt[:] + cur_iter
    xt[:] = xt / (np.exp(tau))
    with objmode(run_time='f8'):
        run_time = time.perf_counter() - start
    algo_time = run_time
    return est_n, xt, algo_time


def hk_relax(indptr, indices, degree, s_node, tau, eps):
    """
    This code is adopted from https://gist.github.com/dgleich/7d904a10dfd9ddfaf49a
    [1] Kloster, Kyle, and David F. Gleich. "Heat kernel based community detection." KDD, pp. 1386-1395. 2014.

    It is quite slow compared with hk_relax_queue.

    """
    # t, tol, N, psis are precomputed
    xt = {}  # Store x, r as dictionaries
    rt = {}  # initialize residual
    queue = collections.deque()  # initialize queue
    opers = 0.
    rt[(s_node, 0)] = 1.
    queue.append((s_node, 0))
    opers += 1
    est_n = int(estimate_n(tau, .5 * eps))
    psi = np.zeros(int(est_n) + 1, dtype=np.float64)
    psi[est_n] = 1.
    for i in range(est_n - 1, 0, -1):
        psi[i] = psi[i + 1] * tau / (i + 1.) + 1.
    while len(queue) > 0:
        (v, j) = queue.popleft()  # v has r[(v,j)] ...
        rvj = rt[(v, j)]
        # perform the hk-relax step
        if v not in xt:
            xt[v] = 0.
        xt[v] += rvj
        rt[(v, j)] = 0.
        update = rvj / degree[v]
        mass = (tau / (float(j) + 1.)) * update
        # for neighbors of v
        for u in indices[indptr[v]:indptr[v + 1]]:
            vv = (u, j + 1)  # in the next block
            if j + 1 == est_n:
                xt[u] += update
                continue
            if vv not in rt:
                rt[vv] = 0.
            thresh = np.exp(tau) * eps * degree[u]
            if rt[vv] < thresh / (est_n * psi[j + 1]) <= rt[vv] + mass:
                queue.append(vv)  # add u to queue
            rt[vv] = rt[vv] + mass
        opers += degree[v]
    results = np.zeros_like(degree, dtype=np.float64)
    for v in xt.keys():
        results[v] = xt[v]
    xt = results / np.exp(tau)
    return est_n, xt


@njit(cache=True)
def hk_relax_queue(n, indptr, indices, degree, s_node, tau, eps):
    """ Reimplement hk-relax using queue. """
    with objmode(start='f8'):
        start = time.perf_counter()
    xt = np.zeros(n, dtype=np.float64)
    rt = np.zeros((2, n), dtype=np.float64)

    queue = np.zeros((2, n), dtype=np.int64)
    q_mark = np.zeros((2, n), dtype=np.bool_)
    rear = np.zeros(2, dtype=np.int64)
    front = np.zeros(2, dtype=np.int64)

    queue[0][0] = s_node
    q_mark[0][s_node] = True
    front[0] = 0
    rear[0] = 1
    rt[0][s_node] = 1.

    est_n = estimate_n(tau, .5 * eps)
    psi = np.zeros(int(est_n) + 1, dtype=np.float64)
    psi[int(est_n)] = 1.
    for i in range(int(est_n) - 1, 0, -1):
        psi[i] = psi[i + 1] * tau / (i + 1.) + 1.
    eps_vec = np.exp(tau) * eps * degree / est_n

    # debug info
    oper = 0
    beta_num = 0.
    beta_t = [rt[0][s_node]]
    for j in np.arange(est_n, dtype=np.int64):
        q_cur = j % 2
        q_next = (j + 1) % 2
        if rear[q_cur] == front[q_cur]:
            break
        while front[q_cur] != rear[q_cur]:
            v = queue[q_cur][front[q_cur]]
            front[q_cur] += 1
            oper += degree[v]

            rvj = rt[q_cur][v]
            xt[v] += rvj
            rt[q_cur][v] = 0.
            beta_num += np.abs(rvj)
            update = rvj / degree[v]
            mass = (tau / (j + 1.)) * update
            for u in indices[indptr[v]:indptr[v + 1]]:
                if j + 1 == est_n:
                    xt[u] += update
                    continue
                rt_u = rt[q_next][u]
                if rt_u < eps_vec[u] / psi[j + 1] <= rt_u + mass:
                    queue[q_next][rear[q_next]] = u
                    rear[q_next] += 1
                rt[q_next][u] = rt[q_next][u] + mass
        # empty the queue
        front[q_cur] = 0
        rear[q_cur] = 0
        rt[q_cur][:] = 0.
        q_mark[q_cur][:] = 0
        if rear[q_next] != front[q_next]:
            if beta_t[-1] != 0.:
                beta_t[-1] = beta_num / beta_t[-1]
                beta_num = 0.
            beta_t.append(np.sum(np.abs(rt[q_next])))
    xt /= np.exp(tau)
    with objmode(run_time='f8'):
        run_time = time.perf_counter() - start
    algo_time = run_time
    return est_n, xt, rt, oper, beta_t, algo_time


@njit(cache=True)
def hk_sor_queue(n, indptr, indices, degree, s_node, tau, eps, omega, local):
    with objmode(start='f8'):
        start = time.perf_counter()
    m = np.sum(degree)
    xt = np.zeros(n, dtype=np.float64)
    rt = np.zeros((2, n), dtype=np.float64)

    queue = np.zeros((2, n), dtype=np.int64)
    q_mark = np.zeros((2, n), dtype=np.bool_)
    rear = np.zeros(2, dtype=np.int64)
    front = np.zeros(2, dtype=np.int64)

    queue[0][0] = s_node
    q_mark[0][s_node] = True
    front[0] = 0
    rear[0] = 1
    rt[0][s_node] = 1.

    est_n = estimate_n(tau, .5 * eps)
    psi = np.zeros(int(est_n) + 1, dtype=np.float64)
    psi[int(est_n)] = 1.
    for i in range(int(est_n) - 1, -1, -1):
        psi[i] = psi[i + 1] * tau / (i + 1.) + 1.
    eps_vec = np.exp(tau) * eps * degree / est_n

    oper = 0.
    beta_num = 0.
    beta_t = [rt[0][s_node]]
    for j in np.arange(est_n, dtype=np.int64):
        q_cur = j % 2
        q_next = (j + 1) % 2
        if local:
            if rear[q_cur] == front[q_cur]:
                break
            while front[q_cur] != rear[q_cur]:
                uu = queue[q_cur][front[q_cur] % n]
                front[q_cur] += 1
                if np.abs(rt[q_cur][uu]) < eps_vec[uu] / psi[j]:
                    continue
                oper += degree[uu]

                ru_j = omega * rt[q_cur][uu]
                xt[uu] += ru_j
                rt[q_cur][uu] -= ru_j
                beta_num += np.abs(ru_j)
                mass = (tau / (j + 1.)) * ru_j / degree[uu]
                for vv in indices[indptr[uu]:indptr[uu + 1]]:
                    if j + 1 == est_n:
                        xt[vv] += ru_j / degree[uu]
                        continue
                    rt[q_next][vv] = rt[q_next][vv] + mass
                    if np.abs(rt[q_next][vv]) >= eps_vec[vv] / psi[j + 1] and not q_mark[q_next][vv]:
                        q_mark[q_next][vv] = True
                        queue[q_next][rear[q_next]] = vv
                        rear[q_next] += 1
                if np.abs(rt[q_cur][uu]) >= eps_vec[uu] / psi[j] and not q_mark[q_next][uu]:
                    queue[q_cur][rear[q_cur] % n] = uu
                    rear[q_cur] += 1
        else:
            while True:
                for uu in range(n):
                    ru_j = omega * rt[q_cur][uu]
                    xt[uu] += ru_j
                    rt[q_cur][uu] -= ru_j
                    beta_num += np.abs(ru_j)
                    mass = (tau / (j + 1.)) * ru_j / degree[uu]
                    for vv in indices[indptr[uu]:indptr[uu + 1]]:
                        if j + 1 == est_n:
                            xt[vv] += ru_j / degree[uu]
                        else:
                            rt[q_next][vv] = rt[q_next][vv] + mass
                oper += m + n
                if np.sum((eps_vec / psi[j]) <= np.abs(rt[q_cur])) <= 0.:
                        break
        # empty the queue
        front[q_cur] = 0
        rear[q_cur] = 0
        rt[q_cur][:] = 0.
        q_mark[q_cur][:] = 0
        if rear[q_next] != front[q_next]:
            if beta_t[-1] != 0.:
                beta_t[-1] = beta_num / beta_t[-1]
                beta_num = 0.
            beta_t.append(np.sum(np.abs(rt[q_next])))
    xt /= np.exp(tau)
    with objmode(run_time='f8'):
        run_time = time.perf_counter() - start
    algo_time = run_time
    return est_n, xt, rt, oper, beta_t, run_time, algo_time


@njit(cache=True)
def hk_loc_gd_queue(n, indptr, indices, degree, s_node, tau, eps, omega, opt_x):
    pass


@njit(cache=True, parallel=True)
def solve_a_graph_all(n, indptr, indices, degree, tau, eps, omega, graph_name, s_nodes):
    test_num = len(s_nodes)
    if graph_name == 'ogbn-papers100M' or graph_name == 'com-friendster':
        batch_num = 17
    else:
        batch_num = 50

    local_sor_opers = np.zeros(test_num, dtype=np.int64)
    local_sor_algo_times = np.zeros(test_num)
    global_sor_opers = np.zeros(test_num, dtype=np.int64)
    global_sor_algo_times = np.zeros(test_num)

    local_ahk_opers = np.zeros(test_num, dtype=np.int64)
    local_ahk_algo_times = np.zeros(test_num)
    global_ahk_opers = np.zeros(test_num, dtype=np.int64)
    global_ahk_algo_times = np.zeros(test_num)

    taylor_approx_opers = np.zeros(test_num, dtype=np.int64)
    taylor_approx_algo_times = np.zeros(test_num)
    relax_opers = np.zeros(test_num, dtype=np.int64)
    relax_algo_times = np.zeros(test_num)

    now_batch = 0
    while True:
        '''for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            est_n, xt, algo_time = hk_taylor_approx(n, indptr, indices, degree, s_node, tau, eps)
            #taylor_approx_opers[ii] = oper
            taylor_approx_algo_times[ii] = algo_time'''
        for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            est_n, xt, rt, oper, beta_t, run_time, algo_time = hk_sor_queue(n, indptr, indices, degree, s_node, tau, eps, omega, local=True)
            local_sor_opers[ii] = oper
            local_sor_algo_times[ii] = algo_time

        for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            est_n, xt, rt, oper, beta_t, run_time, algo_time = hk_sor_queue(n, indptr, indices, degree, s_node, tau, eps, omega, local=False)
            global_sor_opers[ii] = oper
            global_sor_algo_times[ii] = algo_time
        for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            est_n, xt, rt, oper, beta_t, run_time, algo_time = hk_sor_queue(n, indptr, indices, degree, s_node, tau, eps, omega=1.0, local=True)
            local_ahk_opers[ii] = oper
            local_ahk_algo_times[ii] = algo_time
        for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            est_n, xt, rt, oper, beta_t, run_time, algo_time = hk_sor_queue(n, indptr, indices, degree, s_node, tau, eps, omega=1.0, local=False)
            global_ahk_opers[ii] = oper
            global_ahk_algo_times[ii] = algo_time

        '''for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            est_n, xt, rt, oper, beta_t, algo_time = hk_relax_queue(n, indptr, indices, degree, s_node, tau, eps)
            relax_opers[ii] = oper
            relax_algo_times[ii] = algo_time'''
        now_batch += 1
        if batch_num * now_batch >= test_num:
            break
    return (s_nodes, local_sor_opers, local_sor_algo_times, global_sor_opers,
            global_sor_algo_times, local_ahk_opers, local_ahk_algo_times,
            global_ahk_opers, global_ahk_algo_times, taylor_approx_opers,
            taylor_approx_algo_times, relax_opers, relax_algo_times)


@njit(cache=True, parallel=True)
def solve_a_graph_pk(n, indptr, indices, degree, tau, eps, omega, graph_name, s_nodes):
    test_num = len(s_nodes)
    if graph_name == 'ogbn-papers100M' or graph_name == 'com-friendster':
        batch_num = 40
    else:
        batch_num = 60
    local_sor_opers = np.zeros(test_num)
    local_sor_algo_times = np.zeros(test_num)
    local_sor_est_ns = np.zeros(test_num)
    pk = np.zeros(test_num)
    m = degree.sum()
    now_batch = 0
    while True:
        for ii in numba.prange(batch_num * now_batch, min(batch_num * (now_batch + 1), test_num)):
            s_node = s_nodes[ii]
            est_n, xt, rt, oper, beta_t, run_time, algo_time = hk_sor_queue(n, indptr, indices, degree, s_node, tau, eps, omega=1.0, local=False)
            local_sor_est_ns[ii] = est_n
            local_sor_opers[ii] = oper
            local_sor_algo_times[ii] = algo_time
            pk[ii] = ((np.sum(np.power(xt * n, 2))) ** 2) / (np.sum(np.power(xt * n, 4)))
        now_batch += 1
        if batch_num * now_batch > test_num:
            break
    return (local_sor_opers, local_sor_algo_times, pk, local_sor_est_ns)
