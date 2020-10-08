def vertex_normals(v, f, normalized=True):
    fn = face_normals(v, f, normalize=False)
    matrix = vertex_face_adjacency(v.shape[0], f)
    vn = matrix.dot(fn)
    if normalized:
        vn = row_normalize(vn)


def vertex_face_adjacency(nv, f, data=None):
    """
    Return a sparse matrix for which vertices are contained in which faces.
    A data vector can be passed which is then used instead of booleans
    """
    # Input checks:
    f = np.asanyarray(f)  # Convert to an ndarray or pass if already is one
    nv = int(nv)

    # Computation
    row = f.reshape(-1)  # Flatten indices
    col = np.tile(np.arange(len(f)).reshape((-1, 1)), (1, f.shape[1])).reshape(-1)  # Data for vertices
    shape = (nv, len(f))

    if data is None:
        data = np.ones(len(col), dtype=np.bool)

    # assemble into sparse matrix
    return scipy.sparse.coo_matrix((data, (row, col)), shape=shape, dtype=data.dtype)

    # TODO - Different Implementation - check if faster & equivalent
    # return sparse.coo_matrix((np.ones((3 * npoly,)),  # data
    #                           (np.hstack(self.polys.T),  # row
    #                            np.tile(range(npoly), (1, 3)).squeeze())),  # col
    #                          (npt, npoly)).tocsr()  # size
