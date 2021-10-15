import meshio
import os
# from util.fileio import file_extension, align_file_extension


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#


def write_mesh(fp, v, f=None):
    filename, ext = os.path.splitext(fp)

    if ext == 'off':
        return write_off(fp, v, f)
    elif ext == 'ply':
        return write_ply(fp, v, f)
    elif ext == 'obj':
        return write_obj(fp, v, f)
    else:
        assert f is not None, "meshio doesn't support point cloud write"
        mesh = meshio.Mesh(v, f)
        meshio.write(fp, mesh)


def full_write_mesh(fp, v, f, **kwargs):
    # Optionally provide extra data on points, cells, etc.
    # point_data=point_data,
    # cell_data=cell_data,
    # field_data=field_data
    meshio.Mesh(points=v, cells=[("triangle", f)], **kwargs).write(fp)


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#


def write_off(fp, v, f=None):
    # fp = align_file_extension(fp, 'off')
    str_v = [f"{vv[0]} {vv[1]} {vv[2]}\n" for vv in v]
    if f is not None:
        str_f = [f"3 {ff[0]} {ff[1]} {ff[2]}\n" for ff in f]
    else:
        str_f = []

    with open(fp, 'w') as meshfile:
        meshfile.write(f'OFF\n{len(str_v)} {len(str_f)} 0\n{"".join(str_v)}{"".join(str_f)}')


def write_obj(fp, v, f=None):
    # fp = align_file_extension(fp, 'obj')
    str_v = [f"v {vv[0]} {vv[1]} {vv[2]}\n" for vv in v]
    if f is not None:
        # Faces are 1-based, not 0-based in obj files
        str_f = [f"f {ff[0]} {ff[1]} {ff[2]}\n" for ff in f + 1]
    else:
        str_f = []

    with open(fp, 'w') as meshfile:
        meshfile.write(f'{"".join(str_v)}{"".join(str_f)}')


def write_ply(fp, v, f=None):
    # fp = align_file_extension(fp, 'ply')
    with open(fp, 'w') as meshfile:
        meshfile.write(
            f"ply\nformat ascii 1.0\nelement vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\n")
        if f is not None:
            meshfile.write(f"element face {len(f)}\nproperty list uchar int vertex_index\n")
        meshfile.write("end_header\n")

        str_v = [f"{vv[0]} {vv[1]} {vv[2]}\n" for vv in v]
        if f is not None:
            str_f = [f"3 {ff[0]} {ff[1]} {ff[2]}\n" for ff in f]
        else:
            str_f = []
        meshfile.write(f'{"".join(str_v)}{"".join(str_f)}')


def _write_obj_extended(file, V, F, Vt=None, Ft=None):
    """
    # TODO - embed
    Writes OBJ files
    Only handles vertices, faces and UV maps
    Inputs:
    - file: path to .obj file (overwrites if exists)
    - V: 3D vertices
    - F: 3D faces
    - Vt: UV vertices
    - Ft: UV faces
    Correspondence between mesh and UV map is implicit in F to Ft correspondences
    If no UV map data as input, it will write only 3D data in .obj file
    """
    if not Vt is None:
        assert len(F) == len(Ft), 'Inconsistent data, mesh and UV map do not have the same number of faces'

    with open(file, 'w') as file:
        # Vertices
        for v in V:
            line = 'v ' + ' '.join([str(_) for _ in v]) + '\n'
            file.write(line)
        # UV verts
        if not Vt is None:
            for v in Vt:
                line = 'vt ' + ' '.join([str(_) for _ in v]) + '\n'
                file.write(line)
        # 3D Faces / UV faces
        if Ft:
            F = [[str(i + 1) + '/' + str(j + 1) for i, j in zip(f, ft)] for f, ft in zip(F, Ft)]
        else:
            F = [[str(i + 1) for i in f] for f in F]
        for f in F:
            line = 'f ' + ' '.join(f) + '\n'
            file.write(line)
