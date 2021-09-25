import torch
import torch.nn.functional as F
import numpy as np
import os
from torch_scatter import scatter_add

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

SAVE_MEMORY = False


def vertex_face_adjacency(V, F):
    """
    Return a sparse matrix for which vertices are contained in which faces.
    A weight vector can be passed which is then used instead of booleans - for example, the face areas
    weight vector format: [face0,face0,face0,face1,face1,face1,...]
    Input:
        V: |V| x 3 - vertices matrix
        F: |F| x 3 - faces matrix
    Outputs:
        VF_adj: |V| x |F| adjacency matrix
    """

    # tensor type
    device = V.device
    dtype = V.dtype

    VF_adj = torch.zeros((V.shape[0], F.shape[0]), device=device, dtype=dtype)
    v_idx = F.view(-1)
    f_idx = torch.arange(F.shape[0]).repeat(3).reshape(3, F.shape[0]).transpose(1, 0).contiguous().view(
        -1)  # [000111...FFF]

    VF_adj[v_idx, f_idx] = 1
    return VF_adj


def gradient_divergence_functions(V, F):
    """
    Input:
        V: B x |V| x 3 - batch of vertices matrices
        F: B x |F| x 3 - batch of faces matrices
    output:
        gradient(): gradient function
        divergence(): divergence function
    XF: 3 x |F| x 3 - tensor where each element is (x, y, z), each column corresponds to the 3 vertices of each face
    Na: |F| x 3 - vector of the normal vectors ((x1,y1,z1), ... , (xn,yn,zn,)) (un-normalized)
    A: |F| x 1  - vector of the areas of the faces*2 == length of the normals
    N: |F| x 3 - vector of the normal vectors ((x1,y1,z1), ... , (xn,yn,zn,)) (normalized)
    dA: |F| x 1 - # TODO: defining the descripstion
    """
    # tensor type and device
    device = V.device
    dtype = V.dtype

    V = V.reshape([-1, 3])
    F = F.reshape([-1, 3])

    XF = V[F, :].transpose(0, 1)

    Na = torch.cross(XF[1] - XF[0], XF[2] - XF[0])  # perpendicular vector of the face
    A = torch.sqrt(torch.sum(Na ** 2, -1, keepdim=True)) + 1e-6  # calculate vector's length - the area of the 2*face
    N = Na / A  # N is the normal vector of the triangle (XF[0], XF[1], XF[2])
    dA = 1 / A

    num_faces = F.shape[0]
    num_vertices = V.shape[0]

    def gradient(f):
        """
        Input:
            f: |V| x |V|
        output:
            grad_f: |F| x 3 x |V|
        v: |F| x 3
        """
        gradient_f = torch.zeros(num_faces, 3, f.shape[-1], device=device, dtype=dtype)
        for i in range(3):
            s = (i + 1) % 3
            t = (i + 2) % 3
            v = -torch.cross(XF[t] - XF[s], N)
            if SAVE_MEMORY:
                gradient_f.add_(
                    f[F[:, i], None, :] * (dA[:, 0, None, None] * v[:, :, None]))  # slower but uses less memory
            else:
                gradient_f.add_(f[F[:, i], None, :] * (dA[:, 0, None, None] * v[:, :, None]))
        return gradient_f

    def divergence(f):
        """
        Input:
            f: |F| x 3 x |V|
        output:
            divergence_f: |V| x |V|
        v: |F| x 3
        """
        divergence_f = torch.zeros(f.shape[-1], num_vertices, device=device, dtype=dtype)
        for i in range(3):
            s = (i + 1) % 3
            t = (i + 2) % 3
            v = torch.cross(XF[t] - XF[s], N)
            if SAVE_MEMORY:
                divergence_f.add_(scatter_add(torch.bmm(v[:, None, :], f)[:, 0, :].t(), F[:, i],
                                              dim_size=num_vertices))  # slower but uses less memory
            else:
                divergence_f.add_(scatter_add((f * v[:, :, None]).sum(1).t(), F[:, i], dim_size=num_vertices))
        return 0.5 * divergence_f.t()

    return gradient, divergence


def distance_geodesic_heat_method(V, F, t=1e-1):
    """
    Input:
        V: B x |V| x 3 - batch of vertices matrices
        F: B x |F| x 3 - batch of faces matrices
        t: float(scalar) - time parameter
    Outputs:
        D: |V| x |V| - geodesic distances matrix
    L: B x |V| x |V| - minus laplacian tensor
    A: |V| - vertices areas vector (one third of all immediately adjacent triangle areas)
    grad(): gradient function
    diver(): divergence function
    """

    def _geodesics_in_heat(gradient, divergence, L, A, t=1e-1):
        """
        Input:
            gradient(): gradient function
            divergence(): divergence function
            L: |V| x |V| - minus laplacian matrix
            A: |V| - vertices areas vector: V (one third of all immediately adjacent triangle areas)
            t: float(scalar) - time parameter
        Outputs:
            D: |V| x |V| -  geodesic distances matrix
        B: |V| x |V| - (I*A - t*L)
        U: |V| x |V| - indicator matrix
        u: |V| x |V| - solution vector of linear system of part 1
        grad_u: |F| x 3 x |V| - gradient vector of u
        X: |F| x 3 x |V|
        """
        nsplits = 1
        if SAVE_MEMORY:
            nsplits = 5

        # tensor type and device
        device = L.device
        dtype = L.dtype

        num_vertices = L.shape[0]
        chunk = int(num_vertices / nsplits)
        D = torch.zeros(num_vertices, num_vertices, dtype=dtype, device=device)

        B = torch.diag(A) + t * L

        for i in range(nsplits):
            i1 = i * chunk
            i2 = np.min([num_vertices, (i + 1) * chunk]).item()

            U = torch.zeros(num_vertices, i2 - i1, dtype=dtype, device=device)
            U[i1:i2, :(i2 - i1)] = torch.eye((i2 - i1), dtype=dtype, device=device)  # indicator vector?
            u = torch.solve(U, B)[0]  # part (1) in Geodesic distance calc
            gradient_u = gradient(u)
            X = gradient_u * (gradient_u.pow(2).sum(1,
                                                    keepdims=True) + 1e-12).rsqrt()  # rsqrt() returns 1 / sqrt(input_i), part (2) in Geodesic distance calc

            Di = torch.solve(divergence(X), L)[0]  # part (3) in Geodesic distance calc
            D[:, i1:i2] = Di
        return D

    L, A = laplacian_batch(V, F)
    gradient, divergence = gradient_divergence_functions(V, F)

    D = _geodesics_in_heat(gradient, divergence, L[0], A, t)
    D_diag = torch.diag(D)[:, None]

    # make D symmetric and set the diagonal to zero
    D = (D + D.t() - D_diag - D_diag.t()) / 2

    return D


def laplacian_batch(V, F):
    """
    Input:
      V: B x |V| x 3 - batch of vertices matrices
      F: B x |F| x 3 - batch of faces matrices
    Outputs:
      L: B x |V| x |V| - minus laplacian calculated via W

    A: B x |F| x 1 - faces areas
    W: B x |V| x |V| - batch of cotangents matrices
    """
    # tensor type and device
    device = V.device
    dtype = V.dtype

    indices_repeat = torch.stack([F, F, F], dim=2)

    # v1 is the list of first triangles batch_size * F * 3, v2 second and v3 third
    v1 = torch.gather(V, 1, indices_repeat[:, :, :, 0].long())
    v2 = torch.gather(V, 1, indices_repeat[:, :, :, 1].long())
    v3 = torch.gather(V, 1, indices_repeat[:, :, :, 2].long())

    # distance of edge i-j for every face batch_size * F
    l1 = torch.sqrt(((v2 - v3) ** 2).sum(2))
    l2 = torch.sqrt(((v3 - v1) ** 2).sum(2))
    l3 = torch.sqrt(((v1 - v2) ** 2).sum(2))

    # Heron's formula for area
    A = 0.5 * (torch.sum(torch.cross(v2 - v1, v3 - v2, dim=2) * 2, dim=2) * 0.5)  # VALIDATED

    # Theorem d Al Kashi : c2 = a2 + b2 - 2ab cos(angle(ab))
    cot23 = (l1 * 2 - l2 * 2 - l3 ** 2) / (8 * A)
    cot31 = (l2 * 2 - l3 * 2 - l1 ** 2) / (8 * A)
    cot12 = (l3 * 2 - l1 * 2 - l2 ** 2) / (8 * A)

    # Handle degenerate triangles:
    # cot23[np.isinf(cot23) | np.isnan(cot23)] = False
    # cot31[np.isinf(cot31) | np.isnan(cot31)] = False
    # cot12[np.isinf(cot12) | np.isnan(cot12)] = False

    batch_cot23 = cot23.view(-1)
    batch_cot31 = cot31.view(-1)
    batch_cot12 = cot12.view(-1)

    # proof page 98 http://www.cs.toronto.edu/~jacobson/images/alec-jacobson-thesis-2013-compressed.pdf
    # C = torch.stack([cot23, cot31, cot12], 2) / torch.unsqueeze(A, 2) / 8 # dim: [batch_size x F x 3] cotangent of angle at vertex 1,2,3 correspondingly

    batch_size = V.shape[0]
    num_vertices = V.shape[1]
    num_faces = F.shape[1]

    edges_23 = F[:, :, [1, 2]]
    edges_31 = F[:, :, [2, 0]]
    edges_12 = F[:, :, [0, 1]]

    batch_edges_23 = edges_23.view(-1, 2)
    batch_edges_31 = edges_31.view(-1, 2)
    batch_edges_12 = edges_12.view(-1, 2)

    W = torch.zeros(batch_size, num_vertices, num_vertices, dtype=dtype, device=device)

    repeated_batch_idx_f = torch.arange(0, batch_size).repeat(num_faces).reshape(num_faces, batch_size).transpose(1,
                                                                                                                  0).contiguous().view(
        -1)  # [000...111...BBB...], number of repetitions is: num_faces
    repeated_batch_idx_v = torch.arange(0, batch_size).repeat(num_vertices).reshape(num_vertices, batch_size).transpose(
        1, 0).contiguous().view(-1)  # [000...111...BBB...], number of repetitions is: num_vertices
    repeated_vertex_idx_b = torch.arange(0, num_vertices).repeat(batch_size)

    W[repeated_batch_idx_f, batch_edges_23[:, 0], batch_edges_23[:, 1]] = batch_cot23
    W[repeated_batch_idx_f, batch_edges_31[:, 0], batch_edges_31[:, 1]] = batch_cot31
    W[repeated_batch_idx_f, batch_edges_12[:, 0], batch_edges_12[:, 1]] = batch_cot12

    L = W + W.transpose(2, 1)  # here W == cotangent weights matrix

    batch_rows_sum_W = torch.sum(L, dim=1).view(-1)
    L[repeated_batch_idx_v, repeated_vertex_idx_b, repeated_vertex_idx_b] = -batch_rows_sum_W  # -(Laplacian matrix)
    # Warning: residual error of torch.max(torch.sum(W,dim = 1).view(-1)) is ~ 1e-18

    VF_adj = vertex_face_adjacency(V[0], F[0]).unsqueeze(0).expand(batch_size, num_vertices, num_faces)  # VALIDATED
    V_area = (torch.bmm(VF_adj, A.unsqueeze(2)) / 3).squeeze()  # VALIDATED

    return L, V_area


def euclidean_dist_matrix(V):
    """
    Input:
      V: B x N x 3 -  vertices matrix
    Outputs:
      D: B x N x N -  euclidean dist matrix
    """
    # TODO: check the shape of the tensors in the notes
    r = torch.sum(V ** 2, dim=2).unsqueeze(2)
    r_t = r.transpose(2, 1)
    inner = torch.bmm(V, V.transpose(2, 1))
    D = F.relu(
        r - 2 * inner + r_t) * 0.5  # D[i,j] = (r[i] - 2 a[i]a[j]_t + r[j])*0.5, relu - distances must be positive

    return D