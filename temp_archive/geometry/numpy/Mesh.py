import OffFile
import numpy as np
from scipy.sparse import csr_matrix
import pyvista as pv

class Mesh:
    def __init__(self, off_filepath):
        # Shared vertex representation
        vertices, faces, nv, ne, nf = OffFile.read_off(off_filepath)
        self.v = vertices
        self.f = faces
        self.nv = nv
        self.nf = nf
        self.nE = ne
        # Adjacency matrix representation 0,1;1,0;0,2;2,0
        faces = faces[:, 1:]  # remove first column
        rows = np.concatenate((faces[:,0], faces[:,0], faces[:,1], faces[:,1], faces[:,2], faces[:,2]))
        cols = np.concatenate((faces[:,1], faces[:,2], faces[:,0], faces[:,2], faces[:,0], faces[:,1]))
        data = np.ones((cols.shape[0]), dtype=int)
        rows, cols, data = zip(*set(zip(rows, cols, data)))  # remove duplicates before creating sparse
        self.M = csr_matrix((data, (rows, cols)), shape=(nv, nv))

    def plot_wireframe(self):
        p_obj = pv.PolyData(self.v, self.f)
        p_obj.plot(show_edges=True, style='wireframe')

    def plot_vertices(self, f):
        p_obj = pv.PolyData(self.v)
        p_obj["elevation"] = f
        p_obj.plot(point_size=5, render_points_as_spheres=True)


    def plot_faces(self, f):
        p_obj = pv.PolyData(self.v, self.f)
        p_obj["elevation"] = f
        p_obj.plot(show_edges=True)

    def get_valence(self):
        valence = np.array(self.M.sum(axis=0))
        return np.squeeze(valence)

    def get_face_normals(self):
        faces = self.f[:, 1:]  # remove vertex number
        v1 = self.v[faces[:, 0]]
        v2 = self.v[faces[:, 1]]
        v3 = self.v[faces[:, 2]]
        v2_v1 = (v2-v1)  # v2-v1 vector
        v3_v1 = (v3-v1)  # v3-v1 vector
        cross_prod = np.cross(v2_v1, v3_v1)
        face_normals = cross_prod/np.linalg.norm(cross_prod, axis=1)[:, None]
        return face_normals

    def get_barycenters(self):
        faces = self.f[:, 1:]  # remove vertex number
        v1 = self.v[faces[:, 0]]
        v2 = self.v[faces[:, 1]]
        v3 = self.v[faces[:, 2]]
        baryc = np.mean([v1, v2, v3], axis=0)
        return baryc

    def get_face_area(self):
        faces = self.f[:, 1:]  # remove vertex number
        v1 = self.v[faces[:, 0]]
        v2 = self.v[faces[:, 1]]
        v3 = self.v[faces[:, 2]]
        v2_v1 = (v2-v1)  # v2-v1 vector
        v3_v1 = (v3-v1)  # v3-v1 vector
        face_area = np.linalg.norm(np.cross(v2_v1, v3_v1), axis=1)
        return face_area

    def get_vertex_normal(self):
        faces = self.f[:, 1:]  # remove vertex number
        v_indices = np.arange(0, self.v.shape[0], 1)  # create vertex index array
        # for each index find how many adjacent faces it has, using lists
        # (aka face 1-ring)
        adjacent_faces = [np.where(faces == x)[0] for x in v_indices]

        # for each vertex, sum over the adjacent normals with area weighting
        vertex_normal = [np.inner(self.get_face_area()[face_list][None, :],
                                  self.get_face_normals()[face_list, :].T)
                                          for face_list in adjacent_faces]

        # return to numpy and make unit vectors
        vertex_normal = np.squeeze(np.array(vertex_normal))
        ver_normal = np.linalg.norm(vertex_normal, axis=1)
        return vertex_normal/ver_normal[:, None]

    # compute the gaussian curvature of the mesh
    def gauss_curv(self):
        faces = self.f[:, 1:]  # remove vertex number
        v1 = self.v[faces[:, 0]]
        v2 = self.v[faces[:, 1]]
        v3 = self.v[faces[:, 2]]
        v2_v1 = (v2-v1)  # v2-v1 vector
        v3_v1 = (v3-v1)  # v3-v1 vector
        v3_v2 = (v3-v2)  # v3-v1 vector
        angle1 = np.arccos(np.einsum('ij,ij->i', v2_v1, v3_v1) / (np.linalg.norm(v2_v1,axis=1) * np.linalg.norm(v3_v1,axis=1)))
        angle2 = np.arccos(np.einsum('ij,ij->i', v2_v1, -v3_v2) / (np.linalg.norm(v2_v1,axis=1) * np.linalg.norm(v3_v2,axis=1)))
        angle3 = np.arccos(np.einsum('ij,ij->i', v3_v2, v3_v1) / (np.linalg.norm(v3_v1,axis=1) * np.linalg.norm(v3_v2,axis=1)))
        # up to here everything is great ( sum of angles in every face is 180 )
        # the gaussian curvature isn't equal 1/r^2 for a sphere but looks good for the hand.off
        face_angles = np.array([angle1, angle2, angle3]).flatten()
        faces = faces.ravel(order='F')
        v_indices = np.arange(0, self.v.shape[0], 1)  # create vertex index array
        angle_sum = np.zeros_like(v_indices, dtype='float64')
        for idx, vertex_idx in enumerate(faces):
            angle_sum[vertex_idx] += face_angles[idx]

        gauss = 2*np.pi - angle_sum
        a=2
        # # get the sparse matrix representation vectors
        # Adj_mat_indices = self.M.indices
        # Adj_mat_indptr = self.M.indptr
        # # create buffer for face angles for each vertex and buffer for end result
        # max_valence = np.max( self.get_valence() )
        # face_angles = np.zeros( (max_valence, 1) )
        # gauss = np.zeros((Adj_mat_indptr.shape[0]-1, 1))
        # # iterate over the adjacent vertices of every vertex and calculate gauss curvature
        # for idx, i in enumerate(Adj_mat_indptr[:-1]):
        #     for jdx, j in enumerate(np.arange( Adj_mat_indptr[idx], Adj_mat_indptr[idx+1], 1)):
        #         v1 = self.v[Adj_mat_indices[i + jdx]]
        #         # check if it's not the last vertex
        #         if(jdx != np.arange( Adj_mat_indptr[idx], Adj_mat_indptr[idx+1], 1).shape[0]-1):
        #             v2 = self.v[Adj_mat_indices[i + jdx + 1]]
        #         else: # if it's the last, find the angle between last vertex and the first (cyclic)
        #             v2 = self.v[Adj_mat_indices[i]]
        #         face_angles[jdx, 0] = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        #
        #     gauss[idx, 0] = 2*np.pi - np.sum(face_angles) # gauss curvature of this vertex
        #     face_angles = np.zeros( (max_valence, 1) )
        return gauss