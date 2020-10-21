import numpy as np
import os
# This is a simple function that reads .off files
#  expecting: loc - string


def read_off(loc):
    with open(loc, "r") as f:
        line0 = f.readline()
        if line0 == "OFF\n":
            line1 = f.readline()
            line1_sp = line1.split()
            nV = int(line1_sp[0])
            nF = int(line1_sp[1])
            nE = int(line1_sp[2])

            lines = f.readlines()
            vertices = np.array( [w.split() for w in lines[:nV]], dtype='float64')
            faces = np.array( [w[:-1].split() for w in lines[nV:]], dtype='int')
            return vertices,faces,nV,nE,nF

        else:
            raise NameError('read_off:The given file is not an .off file!')

# This is a simple function that writes to .off files
# expecting : vertices,faces - numpy arrays
#             loc - string


def write_off(loc, vertices, faces, override=False):
    if os.path.exists(loc) and (override == False):
        raise NameError('write_off:The given file already exists. Pass True to override\n')

    nV = vertices.shape[0]
    nF = faces.shape[0]
    nE = nV + nF - 2  # euler characteristic

    face_edges = faces.shape[1]*np.ones((nF,1))
    faces_ = np.concatenate((face_edges, faces), axis=1)
    with open(loc, "w") as f:
        f.write("OFF\n")
        f.write('{} {} {}\n'.format(nV, nF, nE))
        np.savetxt(f, vertices, delimiter=' ',fmt='%.17f')
        np.savetxt(f, faces_, delimiter=' ',fmt='%d')