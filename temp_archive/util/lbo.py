import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import normalize


def l2_norm_over_last_axis(mat):
    """ return L2 norm (vector length) along the last axis, for example to compute the length of an array of vectors """
    return np.sqrt(np.sum(mat ** 2, axis=-1))


def row_normalize(mat):
    return normalize(mat, norm='l2', axis=1)


def face_normals(v, f, unitize=True):
    a = v[f[:, 0], :]
    b = v[f[:, 1], :]
    c = v[f[:, 2], :]
    fn = np.cross(b - a, c - a)
    if unitize:
        fn = row_normalize(fn)
    return fn


def cotangent_weights(v, f):
    """
    :param v: vertices (numpy array or tensors), dim: [n_v x 3]
    :param f: faces (numpy array or tensors), dim: [n_f x 3]
    :return: The cotangent/conformal weight matrix
    [w_ij=  cot(alpha_ij)+cot(beta_ij) where alpha_ij and beta_ij are the adjacent angle to edge (i,j)]
    To compute the laplacian coordinates, use: L * v
    """
    # Compute triangle edge vectors
    l_01 = v[f[:, 0], :] - v[f[:, 1], :]
    l_02 = v[f[:, 0], :] - v[f[:, 2], :]
    l_12 = v[f[:, 1], :] - v[f[:, 2], :]
    # Compute triangle cotangent angles (dot product / mag cross product), seeing A.dot(B) = |A||B|cos(alpha) and
    # |A.cross(B)| = |A||B|sin(alpha)
    cot0 = (l_01 * l_02).sum(axis=1) / l2_norm_over_last_axis(np.cross(l_01, l_02))
    cot1 = (-l_12 * l_01).sum(axis=1) / l2_norm_over_last_axis(np.cross(-l_12, l_01))
    cot2 = (l_02 * l_12).sum(axis=1) / l2_norm_over_last_axis(np.cross(l_02, l_12))

    # TODO - Do we need to sanitize?
    #         cots[np.isinf(cots)] = 0
    #         cots[np.isnan(cots)] = 0
    #
    nv = v.shape[0]
    cot = np.concatenate((cot0, cot0, cot1, cot1, cot2, cot2))
    ii = np.concatenate([f[:, 1], f[:, 2], f[:, 2], f[:, 0], f[:, 0], f[:, 1]])
    jj = np.concatenate([f[:, 2], f[:, 1], f[:, 0], f[:, 2], f[:, 1], f[:, 0]])
    return sparse.csr_matrix((cot, (ii, jj)), shape=(nv, nv), dtype='float64')


def laplacian(v, f):
    """
    :param v: vertices (numpy array or tensors), dim: [n_v x 3]
    :param f: faces (numpy array or tensors), dim: [n_f x 3]
    :return: The laplacian specified by type
    To compute the laplacian coordinates, use: L * v
    """

    nv = v.shape[0]
    W = cotangent_weights(v, f)
    D = sparse.spdiags(np.array(np.sum(W, 1)).squeeze(), 0, nv, nv)
    L = D - W
    return L


def barycenter_vertex_mass_matrix(v, f):
    return sparse.spdiags(barycenter_vertex_areas(v, f), 0, v.shape[0], v.shape[0])


def face_areas(v, f):
    return 0.5 * l2_norm_over_last_axis(face_normals(v, f, unitize=False))


def barycenter_vertex_areas(v, f):
    nv = v.shape[0]
    fa = (1 / 3) * face_areas(v, f)
    return np.bincount(f[:, 0], fa, nv) + np.bincount(f[:, 1], fa, nv) + np.bincount(f[:, 2], fa, nv)


def laplacian_spectrum(v, f, k, decimals=None):
    L = laplacian(v, f)
    # assert is_symmetric(L.todense()) # Expensive check
    eye = 0.001 * np.eye(L.shape[0],L.shape[1])
    L = L+eye
    M = barycenter_vertex_mass_matrix(v, f)
    eig_val, eig_vec = eigsh(L, k, M, which='LM', sigma=0, tol=1e-7)

    if decimals is not None:
        eig_val = eig_val.round(decimals)
        eig_vec = eig_vec.round(decimals)
    return eig_val, eig_vec, L, M


def main():
    #from geom.tool.vis import plot_mesh_montage
    #from cfg import Assets
    #k=16
    #v,f = Assets.MAN.load()
    #eigvals,eigenfuncs,_,_ = laplacian_spectrum(v,f,k=k,decimals=4)
    #plot_mesh_montage(fb=f,vb=[v]*k,clrb=eigenfuncs,labelb=[f'{str(i)}_{eigvals[i]}' for i in range(k)])
    pass

if __name__ == '__main__':
    main()