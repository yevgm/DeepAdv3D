import math
import random

import torch
import torch_geometric as tgeo
from torch_geometric.data import Data

class Rotate(object):
    def __init__(self, dims=[0,1,2]):
        super().__init__()
        self.dims = dims

    def __call__(self, mesh:Data):
        dims = self.dims
        phi_n = [random.random()*2*math.pi for _ in dims]
        cos_n = [math.cos(phi) for phi in phi_n]
        sin_n = [math.sin(phi) for phi in phi_n]

        pos = mesh.pos
        device = pos.device
        R = torch.tensor(
            [[1, 0, 0],
            [ 0, 1, 0],
            [ 0, 0, 1]], device=device, dtype=torch.float)

        random.shuffle(dims) # add randomness
        for i in range(len(dims)):
            dim, phi = dims[i], phi_n[i]
            cos, sin = math.cos(phi), math.sin(phi)

            if dim == 0:
                tmp = torch.tensor(
                [[1,   0,   0],
                [ 0, cos,-sin],
                [ 0, sin, cos]], device=device)
            elif dim == 1:
                tmp = torch.tensor(
                [[cos, 0,-sin],
                [ 0,   1,   0],
                [ sin, 0, cos]], device=device)
            elif dim == 2:
                tmp = torch.tensor(
                [[cos,-sin,  0],
                [ sin, cos,  0],
                [   0,   0,  1]], device=device)
            R = R.mm(tmp)
        mesh.pos = torch.matmul(pos, R.t())
        return mesh 


class Move(object):
    def __init__(self, mean=[0,0,0], std=[0.05,0.05,0.05]):
        super().__init__()
        self.mean = mean
        self.std = std
    
    def __call__(self, mesh:Data):
        pos = mesh.pos
        n = pos.shape[0]
        comp_device = pos.device
        comp_type = pos.dtype
        mean = torch.tensor([self.mean], device=comp_device, dtype=comp_type)
        std = torch.tensor([self.std], device=comp_device, dtype=comp_type)

        centroid = pos.sum(dim=0, keepdim=True)/n
        if (std == 0).all():
            offset =  mean
        else:
            offset =  torch.normal(mean=mean, std=std) 

        mesh.pos = offset + (pos - centroid)
        return mesh

class ToDevice(object):
    def __init__(self, device:torch.device):
        super().__init__()
        self.argument = device
    
    def __call__(self, mesh:Data):
        mesh.pos = mesh.pos.to(self.argument)
        mesh.face = mesh.face.to(self.argument)
        mesh.edge_index = mesh.edge_index.to(self.argument)
        mesh.y = mesh.y.to(self.argument)
        return mesh
