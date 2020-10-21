import math
import heapq
from typing import List

import numpy as np
import scipy.sparse as sp
import torch

def row(A):
    return A.reshape((1, -1))

def col(A):
    return A.reshape((-1, 1))

def get_adjacency_matrix(mesh_v:np.ndarray, mesh_f:np.ndarray):
    vpv = sp.csc_matrix((len(mesh_v),len(mesh_v)))

    # for each column in the faces...
    for i in range(3):
        IS = mesh_f[:,i]
        JS = mesh_f[:,(i+1)%3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T
    return vpv

def get_vertices_per_edge(mesh_v, mesh_f):
    vc = sp.coo_matrix(get_adjacency_matrix(mesh_v, mesh_f))
    result = np.hstack((col(vc.row), col(vc.col)))
    result = result[result[:,0] < result[:,1]] # for uniqueness
    return result

def vertex_quadrics(mesh_v:np.ndarray, mesh_f:np.ndarray):
    """Computes a quadric for each vertex in the Mesh.
       v_quadrics: an (N x 4 x 4) array, where N is # vertices.
    """

    # Allocate quadrics
    v_quadrics = np.zeros((len(mesh_v), 4, 4,))

    # For each face...
    for f_idx in range(len(mesh_f)):

        # Compute normalized plane equation for that face
        vert_idxs = mesh_f[f_idx]
        
        verts = np.hstack((mesh_v[vert_idxs], np.array([1, 1, 1]).reshape(-1, 1)))
        u, s, v = np.linalg.svd(verts)
        eq = v[-1, :].reshape(-1, 1)
        eq = eq / (np.linalg.norm(eq[0:3]))

        # Add the outer product of the plane equation to the
        # quadrics of the vertices for this face
        for k in range(3):
            v_quadrics[mesh_f[f_idx, k], :, :] += np.outer(eq, eq)
    return v_quadrics

def _get_sparse_transform(faces, num_original_verts):
    verts_left = np.unique(faces.flatten())
    IS = np.arange(len(verts_left))
    JS = verts_left
    data = np.ones(len(JS))

    mp = np.arange(0, np.max(faces.flatten()) + 1)
    mp[JS] = IS
    new_faces = mp[faces.copy().flatten()].reshape((-1, 3))

    ij = np.vstack((IS.flatten(), JS.flatten()))
    mtx = sp.csc_matrix((data, ij), shape=(len(verts_left) , num_original_verts ))

    return (new_faces, mtx)

def qslim_decimator_transformer(mesh_v, mesh_f, factor=None, n_verts_desired=None):
    if factor is None and n_verts_desired is None:
        raise Exception('Need either factor or n_verts_desired.')
    if n_verts_desired is None:
        n_verts_desired = math.ceil(len(mesh_v) * factor)
    
    # compute the quadrics of the mesh
    Qv = vertex_quadrics(mesh_v, mesh_f)

    # fill out a sparse matrix indicating vertex-vertex adjacency
    vert_adj = get_vertices_per_edge(mesh_v, mesh_f)
    data = vert_adj[:, 0] * 0 + 1
    vert_adj = sp.csc_matrix(
        (data, (vert_adj[:, 0], vert_adj[:, 1])), 
        shape=(len(mesh_v), len(mesh_v)))
    
    vert_adj = vert_adj + vert_adj.T
    vert_adj = vert_adj.tocoo()

    def collapse_cost(Qv, r, c, v):
        Qsum = Qv[r, :, :] + Qv[c, :, :]
        p1 = np.vstack((v[r].reshape(-1, 1), np.array([1]).reshape(-1, 1)))
        p2 = np.vstack((v[c].reshape(-1, 1), np.array([1]).reshape(-1, 1)))

        destroy_c_cost = p1.T.dot(Qsum).dot(p1)
        destroy_r_cost = p2.T.dot(Qsum).dot(p2)
        result = {
            'destroy_c_cost': destroy_c_cost,
            'destroy_r_cost': destroy_r_cost,
            'collapse_cost': min([destroy_c_cost, destroy_r_cost]),
            'Qsum': Qsum}
        return result

    # construct a queue of edges with costs
    queue = []
    for k in range(vert_adj.nnz):
        r = vert_adj.row[k]
        c = vert_adj.col[k]

        if r > c:
            continue

        cost = collapse_cost(Qv, r, c, mesh_v)['collapse_cost']
        heapq.heappush(queue, (cost, (r, c)))

    # decimate the mesh
    collapse_list = []
    nverts_total = len(mesh_v)
    faces = mesh_f.copy()

    while nverts_total > n_verts_desired:
        e = heapq.heappop(queue)
        r = e[1][0]
        c = e[1][1]
        if r == c:
            continue

        cost = collapse_cost(Qv, r, c, mesh_v)
        if cost['collapse_cost'] > e[0]:
            heapq.heappush(queue, (cost['collapse_cost'], e[1]))
            # print 'found outdated cost, %.2f < %.2f' % (e[0], cost['collapse_cost'])
            continue
        else:

            # update old vert idxs to new one,
            # in queue and in face list
            if cost['destroy_c_cost'] < cost['destroy_r_cost']:
                to_destroy = c
                to_keep = r
            else:
                to_destroy = r
                to_keep = c

            collapse_list.append([to_keep, to_destroy])

            # in our face array, replace "to_destroy" vertidx with "to_keep" vertidx
            np.place(faces, faces == to_destroy, to_keep)

            # same for queue
            which1 = [idx for idx in range(len(queue)) if queue[idx][1][0] == to_destroy]
            which2 = [idx for idx in range(len(queue)) if queue[idx][1][1] == to_destroy]
            for k in which1:
                queue[k] = (queue[k][0], (to_keep, queue[k][1][1]))
            for k in which2:
                queue[k] = (queue[k][0], (queue[k][1][0], to_keep))

            Qv[r, :, :] = cost['Qsum']
            Qv[c, :, :] = cost['Qsum']

            a = faces[:, 0] == faces[:, 1]
            b = faces[:, 1] == faces[:, 2]
            c = faces[:, 2] == faces[:, 0]

            # remove degenerate faces
            def logical_or3(x, y, z):
                return np.logical_or(x, np.logical_or(y, z))

            faces_to_keep = np.logical_not(logical_or3(a, b, c))
            faces = faces[faces_to_keep, :].copy()

        nverts_total = (len(np.unique(faces.flatten())))
        
    new_faces, mtx = _get_sparse_transform(faces, len(mesh_v))
    return new_faces, mtx

def generate_transform_matrices(mesh_v:np.ndarray, mesh_f:np.ndarray, factors:List[float]):
    if len(mesh_v.shape) != 2 and mesh_v.shape[1] != 3 and isinstance(mesh_v.dtype, np.floating):
        raise ValueError("input vertex positions must have shape [N,3] and floating point data type")
    
    if len(mesh_f.shape) != 2 and mesh_f.shape[1] != 3 and isinstance(mesh_v.dtype, np.integer):
        raise ValueError("input vertex positions must have shape [M,3] and integer data type")

    factors = map(lambda x: 1.0 / x, factors)
    V, F, A, D = [], [], [], []
    A.append(get_adjacency_matrix(mesh_v, mesh_f).tocoo())
    V.append(mesh_v)
    F.append(mesh_f)
    
    for i,factor in enumerate(factors):
        # compute the decimation quadrics
        new_mesh_f, ds_D = qslim_decimator_transformer(V[-1], F[-1], factor=factor)
        D.append(ds_D.tocoo())
        new_mesh_v = ds_D.dot(V[-1])
        
        pos = torch.from_numpy(new_mesh_v)
        face = torch.from_numpy(new_mesh_f).t()
        
        V.append(new_mesh_v)
        F.append(new_mesh_f)
        A.append(get_adjacency_matrix(new_mesh_v, new_mesh_f).tocoo())
    return V,F,A,D