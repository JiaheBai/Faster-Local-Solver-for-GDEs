from numba import njit, objmode, cuda, jit, vectorize
from numba.typed import List, Dict
import numba
import numpy as np
import scipy.sparse as sp
import time
from tqdm import tqdm as tqdm

list_type = numba.types.ListType(numba.types.int64)


@njit(cache=True)
def instantGNNPush(n, indptr, indices, degree, alpha, eps_vec, p, r, queue, front, rear, q_mark, beta):
    with objmode(time1='f8'):
        time1 = time.perf_counter()
    while rear != front:
        front = (front + 1) % n
        u = queue[front]
        q_mark[u] = False
        if np.abs(r[u]) < eps_vec[u]:
            continue
        if (degree[u] == 1 and indices[indptr[u]] == u) or (degree[u] == 0):
            p[u] = r[u]
            r[u] = 0
            continue
        res_u = r[u]
        p[u] += alpha * res_u
        push_u = (1. - alpha) * res_u / (degree[u] ** (1 - beta))
        r[u] = 0
        for v in indices[indptr[u]:indptr[u] + degree[u]]:
            r[v] += push_u / (degree[v] ** beta)
            if (q_mark[v] == False) and np.abs(r[v]) >= eps_vec[v]:
                rear = (rear + 1) % n
                queue[rear] = v
                q_mark[v] = True
        if (q_mark[u] == False) and np.abs(r[u]) >= eps_vec[u]:
            rear = (rear + 1) % n
            queue[rear] = u
            q_mark[u] = True
    with objmode(time2='f8'):
        time2 = time.perf_counter() - time1
    runtime = time2
    return runtime


@njit(cache=True, parallel=True)
def instantGNNParallel(n, indptr, indices, degree, alpha, eps, P, R, beta, batchsize=80):
    feanum = len(R[0])
    runtimes = np.zeros(feanum)
    nowbatch = 0
    while True:
        for fea in numba.prange(nowbatch * batchsize, min((nowbatch + 1) * batchsize, feanum)):
            queue = np.zeros(n + 2, dtype=np.int32)
            front, rear = np.int32(0), np.int32(0)
            p = np.zeros(n, dtype=np.float32)
            r = R.T[fea]
            eps_vec = np.power(degree, 1 - beta) * eps
            q_mark = np.zeros(n)
            for i in range(n):
                if np.abs(r[i]) >= eps_vec[i]:
                    rear = (rear + 1) % (n + 2)
                    queue[rear] = i
                    q_mark[i] = True
            runtime = instantGNNPush(n + 2, indptr, indices, degree, alpha, eps_vec, p, r, queue, front, rear, q_mark, beta)
            R[:, fea] = r
            P[:, fea] = p
            runtimes[fea] = runtime
        nowbatch += 1
        if nowbatch * batchsize >= feanum:
            break
    return runtimes


@njit(cache=True)
def instantGNNSorPush(n, indptr, indices, degree, alpha, eps_vec, omega, p, r, queue, front, rear, q_mark, beta):
    with objmode(time1='f8'):
        time1 = time.perf_counter()
    while rear != front:
        front = (front + 1) % n
        u = queue[front]
        q_mark[u] = False
        if np.abs(r[u]) < eps_vec[u]:
            continue
        if (degree[u] == 1 and indices[indptr[u]] == u) or (degree[u] == 0):
            p[u] = r[u]
            r[u] = 0
            continue
        res_u = omega * r[u]
        p[u] += alpha * res_u
        push_u = (1. - alpha) * res_u / (degree[u] ** (1 - beta))
        r[u] = (1 - omega) * r[u]
        for v in indices[indptr[u]:indptr[u] + degree[u]]:
            r[v] += push_u / (degree[v] ** beta)
            if (q_mark[v] == False) and np.abs(r[v]) >= eps_vec[v]:
                rear = (rear + 1) % n
                queue[rear] = v
                q_mark[v] = True
        if (q_mark[u] == False) and np.abs(r[u]) >= eps_vec[u]:
            rear = (rear + 1) % n
            queue[rear] = u
            q_mark[u] = True
    with objmode(time2='f8'):
        time2 = time.perf_counter() - time1
    runtime = time2
    return runtime


@njit(cache=True, parallel=True)
def instantGNNSorParallel(n, indptr, indices, degree, alpha, eps, omega, P, R, beta, batchsize=80):
    feanum = len(R[0])
    runtimes = np.zeros(feanum)
    nowbatch = 0
    while True:
        for fea in numba.prange(nowbatch * batchsize, min((nowbatch + 1) * batchsize, feanum)):
            queue = np.zeros(n + 2, dtype=np.int64)
            front, rear = np.int64(0), np.int64(0)
            p = np.zeros(n, dtype=np.float64)
            r = R.T[fea]
            eps_vec = np.power(degree, 1 - beta) * eps
            q_mark = np.zeros(n)
            for i in range(n):
                if np.abs(r[i]) >= eps_vec[i]:
                    rear = (rear + 1) % (n + 2)
                    queue[rear] = i
                    q_mark[i] = True
            runtime = instantGNNSorPush(n + 2, indptr, indices, degree, alpha, eps_vec, omega, p, r, queue, front, rear, q_mark,
                                  beta)
            R[:, fea] = r
            P[:, fea] = p
            runtimes[fea] = runtime
        nowbatch += 1
        if nowbatch * batchsize >= feanum:
            break
    return runtimes


@njit(cache=True)
def updetaGraph(indptr, indices, degree, us, vs, onlyinsert = True):
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
def instantGNNUpdateEdge(n, indptr, indices, degree, p, r, x, alpha, queue, front, rear, q_mark, eps, u, v, algo='fwd',
                         omega=1, beta=0, oper=0, itertime=0, itercount=0):
    insertion = True
    deletion = False
    for i in range(indptr[u], indptr[u] + degree[u]):
        if v == indices[i]:
            insertion = False
            if 'insert' in algo:
                break
            if degree[u] == 1 or degree[v] == 1:
                if 'batch' in algo:
                    break
                return front, rear, itertime, itercount, oper
            deletion = True
            indices[i] = indices[indptr[u] + degree[u] - 1]
            for k in range(indptr[v], indptr[v] + degree[v]):
                if u == indices[k]:
                    indices[k] = indices[indptr[v] + degree[v] - 1]
            degree[u] -= 1
            degree[v] -= 1
            break
    if insertion:
        indices[indptr[u] + degree[u]] = v
        indices[indptr[v] + degree[v]] = u
        degree[u] += 1
        degree[v] += 1
        p[u] *= (degree[u] / (degree[u] - 1)) ** (1 - beta)
        r[u] += p[u] * ((degree[u] - 1) ** (1 - beta) - (degree[u] ** (1 - beta))) / (degree[u] ** (1 - beta)) * 1 / alpha
        p[v] *= (degree[v] / (degree[v] - 1)) ** (1 - beta)
        r[v] += p[v] * ((degree[v] - 1) ** (1 - beta) - (degree[v] ** (1 - beta))) / (degree[v] ** (1 - beta)) * 1 / alpha

        r[u] += (p[u] + alpha * r[u] - alpha * x[u]) * (((degree[u] - 1) ** beta) - (degree[u] ** beta)) / (alpha * (degree[u] ** beta))
        r[v] += (p[v] + alpha * r[v] - alpha * x[v]) * (((degree[v] - 1) ** beta) - (degree[v] ** beta)) / (alpha * (degree[v] ** beta))
        r[v] += (1 - alpha) * p[u] / ((degree[v] ** beta) * (degree[u] ** (1 - beta))) * 1 / alpha
        r[u] += (1 - alpha) * p[v] / ((degree[u] ** beta) * (degree[v] ** (1 - beta))) * 1 / alpha
    elif deletion:
        r[u] += (p[u] + alpha * r[u] - alpha * x[u]) * (((degree[u] + 1) ** beta) - (degree[u] ** beta)) / (alpha * (degree[u] ** beta))
        r[v] += (p[v] + alpha * r[v] - alpha * x[v]) * (((degree[v] + 1) ** beta) - (degree[v] ** beta)) / (alpha * (degree[v] ** beta))

        p[u] *= (degree[u] / (degree[u] + 1)) ** (1 - beta)
        r[u] += p[u] * ((degree[u] + 1) ** (1 - beta) - (degree[u] ** (1 - beta))) / (degree[u] ** (1 - beta)) * 1 / alpha
        r[v] -= (1 - alpha) * p[u] / ((degree[v] ** beta) * (degree[u] ** (1 - beta))) * 1 / alpha

        p[v] *= (degree[v] / (degree[v] + 1)) ** (1 - beta)
        r[v] += p[v] * ((degree[v] + 1) ** (1 - beta) - (degree[v] ** (1 - beta))) / (degree[v] ** (1 - beta)) * 1 / alpha
        r[u] -= (1 - alpha) * p[v] / ((degree[u] ** beta) * (degree[v] ** (1 - beta))) * 1 / alpha
    if 'onlygraph' in algo:
        return front, rear, itertime, itercount, oper
    else:  # 'batch' in algo:
        eps_vec = np.power(degree, 1 - beta) * eps
        for i in range(n):
            if np.abs(r[i]) >= eps_vec[i]:
                rear = (rear + 1) % (n + 1)
                queue[rear] = i
                q_mark[i] = True
    if rear != front:
        itercount += rear - front
        if 'sor' in algo:
            front, rear, oper = instantGNNSorPush(n + 1, indptr, indices, degree, alpha, eps_vec, omega, p, r, queue, front,
                                                  rear, q_mark, beta, oper)
        elif 'fwd' in algo:
            front, rear, oper = instantGNNPush(n + 1, indptr, indices, degree, alpha, eps_vec, p, r, queue, front, rear,
                                               q_mark, beta, oper)
    return front, rear, itertime, itercount, oper


@njit(cache=True, parallel=True)
def instantGNNUpdateEdgeParallel(n, indptr, indices, degree, alpha, eps, omega, P, R, X, us, vs, beta, algo):
    nodenum = len(us)
    feanum = len(R[0]) - 1
    for fea in numba.prange(feanum):
        _indptr = indptr.copy()
        _indices = indices.copy()
        _degree = degree.copy()
        queue = np.zeros(n + 1, dtype=np.int64)
        front, rear = np.int64(0), np.int64(0)
        p = P.T[fea]
        r = R.T[fea]
        x = X.T[fea]
        q_mark = np.zeros(n)
        for i in range(nodenum - 1):
            u = us[i]
            v = vs[i]
            _ = instantGNNUpdateEdge(n, _indptr, _indices, _degree, p, r, x, alpha, queue, front, rear, q_mark,
                                     eps, u, v, algo='onlygraphinsert', omega=omega, beta=beta, oper=0,
                                     itertime=0, itercount=0)
        u = us[nodenum - 1]
        v = vs[nodenum - 1]
        _ = instantGNNUpdateEdge(n, _indptr, _indices, _degree, p, r, x, alpha, queue, front, rear, q_mark,
                                 eps, u, v, algo=algo, omega=omega, beta=beta, oper=0,
                                 itertime=0, itercount=0)
        R[:, fea] = r
        P[:, fea] = p
    queue = np.zeros(n + 1, dtype=np.int64)
    front, rear = np.int64(0), np.int64(0)
    p = P.T[feanum]
    r = R.T[feanum]
    x = X.T[feanum]
    q_mark = np.zeros(n)
    for i in range(nodenum - 1):
        u = us[i]
        v = vs[i]
        _ = instantGNNUpdateEdge(n, indptr, indices, degree, p, r, x, alpha, queue, front, rear, q_mark,
                                 eps, u, v, algo='onlygraphinsert', omega=omega, beta=beta, oper=0,
                                 itertime=0, itercount=0)
    u = us[nodenum - 1]
    v = vs[nodenum - 1]
    _ = instantGNNUpdateEdge(n, indptr, indices, degree, p, r, x, alpha, queue, front, rear, q_mark,
                                          eps, u, v, algo=algo, omega=omega, beta=beta, oper=0,
                                          itertime=0, itercount=0)
    R[:, feanum] = r
    P[:, feanum] = p


@njit(cache=True)
def instantGNNUpdateBatch(n, indptr, indices, degree, old_degree, p, r, x, alpha, queue, front, rear, q_mark, eps_vec,
                          insert_neighbor, delete_neighbor, algo='fwd', omega=1, beta=0):
    with objmode(time1='f8'):
        time1 = time.perf_counter()
    for u in insert_neighbor:
        p[u] *= (degree[u] / (old_degree[u])) ** (1 - beta)
        r[u] += p[u] * ((old_degree[u]) ** (1 - beta) - (degree[u] ** (1 - beta))) / (degree[u] ** (1 - beta)) * 1 / alpha
    for u in insert_neighbor:
        dr = (p[u] + alpha * r[u] - alpha * x[u]) * (((old_degree[u]) ** beta) - (degree[u] ** beta)) / (degree[u] ** beta)
        for v in insert_neighbor[u]:
            dr += (1 - alpha) * p[v] / ((degree[u] ** beta) * (degree[v] ** (1 - beta)))
        if 'insert' not in algo:
            if u in delete_neighbor:
                for v in delete_neighbor[u]:
                    dr -= (1 - alpha) * p[v] / ((degree[u] ** beta) * (degree[v] ** (1 - beta)))
        r[u] += dr / alpha
        if np.abs(r[u]) >= eps_vec[u]:
            rear = (rear + 1) % (n + 2)
            queue[rear] = u
            q_mark[u] = True
    if rear != front:
        if 'sor' in algo:
            _ = instantGNNSorPush(n + 2, indptr, indices, degree, alpha, eps_vec, omega, p, r,
                                  queue, front, rear, q_mark, beta)
        elif 'fwd' in algo:
            _ = instantGNNPush(n + 2, indptr, indices, degree, alpha, eps_vec, p, r,
                               queue, front, rear, q_mark, beta)
    with objmode(time2='f8'):
        time2 = time.perf_counter() - time1
    runtime = time2
    return runtime


@njit(cache=True, parallel=True)
def instantGNNUpdateBatchParallel(n, indptr, indices, degree, alpha, eps, omega, P, R, X, us, vs, beta, algo, batchsize=80):
    old_degree = degree.copy()
    feanum = len(R[0])
    runtimes = np.zeros(feanum)
    onlyinsert = ('insert' in algo)
    insert_neighbor, delete_neighbor = updetaGraph(indptr, indices, degree, us, vs, onlyinsert=onlyinsert)
    eps_vec = np.power(degree, 1 - beta) * eps
    nowbatch = 0
    while True:
        for fea in numba.prange(nowbatch * batchsize, min((nowbatch + 1) * batchsize, feanum)):

            queue = np.zeros(n + 1, dtype=np.int32)
            front, rear = np.int32(0), np.int32(0)
            p = P.T[fea]
            r = R.T[fea]
            x = X.T[fea]
            q_mark = np.zeros(n)
            runtime = instantGNNUpdateBatch(n, indptr, indices, degree, old_degree, p, r, x, alpha, queue, front, rear, q_mark,
                                  eps_vec, insert_neighbor, delete_neighbor, algo=algo, omega=omega, beta=beta)
            R[:, fea] = r
            P[:, fea] = p

            runtimes[fea] = runtime
        nowbatch += 1
        if nowbatch * batchsize >= feanum:
            break
    return runtimes