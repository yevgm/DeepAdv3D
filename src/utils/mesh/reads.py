import pathlib
import os
import meshio
import numpy as np
import scipy.io as sio
from plyfile import PlyData

# from util.fileio import file_extension


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#

def read_mesh(fp, verts_only=False):
    filename, ext = os.path.splitext(fp)
    # These are faster than meshio, and achieve the same task
    tried_general_read = False
    try:
        if ext == 'off':
            return read_off_verts(fp) if verts_only else read_off(fp)
        elif ext == 'ply':
            return read_ply_verts(fp) if verts_only else read_ply(fp)
        elif ext == 'obj':
            return read_obj_verts(fp) if verts_only else read_obj(fp)
        elif ext == 'mat':
            return read_mat_fv_struct(fp, verts_only)
        else:
            tried_general_read = True
            mesh = meshio.read(fp)
            return mesh.points if verts_only else mesh.points, mesh.cells[0].data
    except OSError as e:
        if not tried_general_read:
            mesh = meshio.read(fp)
            return mesh.points if verts_only else mesh.points, mesh.cells[0].data
        else:
            raise e


def full_read_mesh(fp):
    """
    :param pathlib.Path or str fp: The file path
    :return: A meshio Mesh object with *all* existing fields
    """
    return meshio.read(fp)


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#

def read_obj_verts(fp):
    v = []
    try:
        with open(fp, 'r') as obj:
            for line in obj:
                elements = line.split()
                if elements[0] == 'v':  # Assuming vertices are first in the file
                    v.append([float(elements[1]), float(elements[2]), float(elements[3])])
                elif elements[0] == 'f':
                    continue  # Instead of break - Sometimes multiple meshes are appended on the file...
        return np.array(v)
    except Exception as e:
        raise OSError(f"Could not read or open mesh file {fp}") from e


def read_mat_fv_struct(fp, verts_only):
    mat = sio.loadmat(fp)
    if 'surface' in mat:
        mat = mat['surface']
        v = np.stack((mat['X'][0][0], mat['Y'][0][0], mat['Z'][0][0])).squeeze().T
        if not verts_only:
            f = mat['TRIV'][0][0] - 1
    elif all(k in mat for k in ('verts', 'faces')):  # TODO - unchecked
        v = mat['verts']
        if not verts_only:
            f = mat['faces'] - 1
    else:
        raise ValueError('Unhandled mat container')
    if verts_only:
        return np.array(v, dtype=np.float64)
    return np.array(v, dtype=np.float64), np.array(f, dtype=np.int32)


def read_obj(fp):
    v, f = [], []
    try:
        with open(fp, 'r') as obj:
            for line in obj:
                elements = line.split()
                if elements[0] == 'v':
                    v.append([float(elements[1]), float(elements[2]), float(elements[3])])
                elif elements[0] == 'f':
                    f.append(
                        [int(elements[1].split('/')[0]), int(elements[2].split('/')[0]),
                         int(elements[3].split('/')[0])])
        return np.array(v), np.array(f) - 1
    except Exception as e:
        raise OSError(f"Could not read or open mesh file {fp}") from e


def _read_obj_extended(file):
    # TODO - embed
    """
    Reads OBJ files
    Only handles vertices, faces and UV maps
    Input:
    - file: path to .obj file
    Outputs:
    - V: 3D vertices
    - F: 3D faces
    - Vt: UV vertices
    - Ft: UV faces
    Correspondence between mesh and UV map is implicit in F to Ft correspondences
    If no UV map data in .obj file, it shall return Vt=None and Ft=None
    """
    V, Vt, F, Ft = [], [], [], []
    with open(file, 'r') as f:
        T = f.readlines()
    for t in T:
        # 3D vertex
        if t.startswith('v '):
            v = [float(n) for n in t.replace('v ', '').split(' ')]
            V += [v]
        # UV vertex
        elif t.startswith('vt '):
            v = [float(n) for n in t.replace('vt ', '').split(' ')]
            Vt += [v]
        # Face
        elif t.startswith('f '):
            idx = [n.split('/') for n in t.replace('f ', '').split(' ')]
            f = [int(n[0]) - 1 for n in idx]
            F += [f]
            # UV face
            if '/' in t:
                f = [int(n[1]) - 1 for n in idx]
                Ft += [f]
    V = np.array(V, np.float32)
    Vt = np.array(Vt, np.float32)
    if Ft:
        assert len(F) == len(Ft), 'Inconsistent .obj file, mesh and UV map do not have the same number of faces'
    else:
        Vt, Ft = None, None
    return V, F, Vt, Ft


def read_off_verts(fp):
    v = []
    try:
        with open(fp, "r") as fh:
            first = fh.readline().strip()
            if first != "OFF" and first != "COFF":
                raise (Exception(f"Could not find OFF header for file: {fp}"))

            # fast forward to the next significant line
            while True:
                line = fh.readline().strip()
                if line and line[0] != "#":
                    break

            # <number of vertices> <number of faces> <number of edges>
            params = line.split()
            if len(params) < 2:
                raise (Exception(f"Wrong number of parameters fount at OFF file: {fp}"))

            while True:
                line = fh.readline().strip()
                if line and line[0] != "#":
                    break

            for i in range(int(params[0])):
                line = line.split()
                v.append([float(line[0]), float(line[1]), float(line[2])])
                line = fh.readline()

        return np.array(v)
    except Exception as e:
        raise OSError(f"Could not read or open mesh file {fp}") from e


def read_off(fp):
    v, f = [], []
    try:
        with open(fp, "r") as fh:
            first = fh.readline().strip()
            if first != "OFF" and first != "COFF":
                raise (Exception(f"Could not find OFF header for file: {fp}"))

            # fast forward to the next significant line
            while True:
                line = fh.readline().strip()
                if line and line[0] != "#":
                    break

            # <number of vertices> <number of faces> <number of edges>
            parameters = line.split()
            if len(parameters) < 2:
                raise (Exception(f"Wrong number of parameters fount at OFF file: {fp}"))

            while True:
                line = fh.readline().strip()
                if line and line[0] != "#":
                    break

            for i in range(int(parameters[0])):
                line = line.split()
                v.append([float(line[0]), float(line[1]), float(line[2])])
                line = fh.readline()

            for i in range(int(parameters[1])):
                line = line.split()
                f.append([int(line[1]), int(line[2]), int(line[3])])
                line = fh.readline()

        return np.array(v), np.array(f)
    except Exception as e:
        raise OSError(f"Could not read or open mesh file {fp}") from e


# def read_off(filename):
#     """
#     read an .off file and return its contents
#     :param filename: path to the .off file
#     :return: (vertices, faces) - numpy arrays
#     """
#     file_data = open(filename, "r").read().split('\n')[1:]
#     file_frm = [np.fromstring(x, sep=' ') for x in file_data]
#     n_verts, n_faces, _ = [int(x) for x in file_frm[0]]
#     verts, faces = np.array(file_frm[1:n_verts+1]), np.array(file_frm[n_verts+1:-1])[:, 1:].astype(int)
#     return verts, faces
#
# def write_off(filename, v, f):
#     """
#     write an .off file
#     :param filename: .off file path to write to
#     :param v: vertices numpy array
#     :param f: faces numpy array
#     :return:
#     """
#     faces = np.concatenate((np.full((f.shape[0], 1), 3), f), axis=1)  # add num of vertex back to faces
#     file_cont = ["OFF", f"{len(v)} {len(f)} 0"]
#     file_cont += [' '.join(map(str, x)) for x in v]
#     file_cont += [' '.join(map(str, x)) for x in faces]
#     open(filename, "w").write('\n'.join(file_cont))
#     return

def read_ply_verts(fp):
    # TODO - consider remove PlyData from dependencies
    try:
        with open(fp, 'rb') as f:
            plydata = PlyData.read(f)
        return np.column_stack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']))
    except Exception as e:
        raise OSError(f"Could not read or open mesh file {fp}") from e


def read_ply(fp):
    # TODO - consider remove PlyData from dependencies
    try:
        with open(fp, 'rb') as f:
            plydata = PlyData.read(f)
        v = np.column_stack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']))
        f = np.stack(plydata['face']['vertex_indices'])
        return v, f
    except Exception as e:
        raise OSError(f"Could not read or open mesh file {fp}") from e



