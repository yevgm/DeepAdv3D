import os

import torch
import numpy as np

import utils

class Downscaler(object):
    def __init__(self, filename, mesh, factor=4):
        if filename[-4:] != ".npy":filename+=".npy"
        self.downscaled_cache_file = filename
        self._mesh = mesh
        self._E, self._F, self._D = None, None, None
        self.factor = factor

    @property
    def _ds_cached(self):
        return os.path.exists(self.downscaled_cache_file)

    @property
    def _ds_loaded(self):
        return self._D is not None and not self._E is None  and not self._F is None
    
    def _load_transfrom_data(self):
        # if not cached, then compute and store
        if self._ds_cached and self._ds_loaded:
            return
        else:
            if self._ds_cached: #data is cached, but not loaded (for example after a restart)
                E,F,D = np.load(self.downscaled_cache_file, allow_pickle=True) #load data
            else: # data is neither cached nor loaded
                data = self._mesh
                v, f = data.pos.numpy(), data.face.t().numpy()
                _,F,E,D = utils.generate_transform_matrices(v, f, [self.factor]*3)
                np.save(self.downscaled_cache_file, (E,F,D))
                
            # assign data to respective fields
            F_t = [torch.tensor(f).t() for f in F]
            D_t = [_scipy_to_torch_sparse(d) for d in D]
            E_t = [_scipy_to_torch_sparse(e) for e in E]
            self._E, self._F, self._D = E_t, F_t, D_t

    @property
    def downscale_matrices(self):
        self._load_transfrom_data()
        return self._D

    @property
    def downscaled_edges(self):
        self._load_transfrom_data()
        return self._E

    @property
    def downscaled_faces(self):
        self._load_transfrom_data()
        return self._F

def _scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape
    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor
