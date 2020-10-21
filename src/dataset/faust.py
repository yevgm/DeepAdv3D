import os

import numpy as np
import torch
import torch_geometric
import torch_geometric.data
import torch_geometric.io as gio
import torch_geometric.transforms as transforms
import tqdm

import dataset.downscale as dscale
from utils.transforms import Move, Rotate, ToDevice

class FaustDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, 
        root:str, 
        device:torch.device=torch.device("cpu"),
        train:bool=True, test:bool=True,
        transform_data:bool=True):
        self.url = 'http://faust.is.tue.mpg.de/'

         # center each mesh into its centroid
        if transform_data:
            # rotate and move
            transform = transforms.Compose([
                Move(mean=[0,0,0], std=[0.05,0.05,0.05]), 
                Rotate(dims=[0,1,2]),
                ToDevice(device)])
        else:
            transform = ToDevice(device)

        super().__init__(root=root, transform=transform, pre_transform=transforms.Center())
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.downscaler = dscale.Downscaler(
            filename=os.path.join(self.processed_dir,"ds"), mesh=self.get(0), factor=4)

        if train and not test:
            self.data, self.slices = self.collate([self.get(i) for i in range(20, 100)])
        elif not train and test:
            self.data, self.slices = self.collate([self.get(i) for i in range(0, 20)])

    @property
    def raw_file_names(self):
        tofilename =  lambda x : "tr_reg_"+str(x).zfill(3)+".ply"
        return [tofilename(fi) for fi in range(100)]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download {} from {} and move it to {}'.format(self.raw_file_names, self.url, self.raw_dir))
    
    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        f2e = transforms.FaceToEdge(remove_faces=False)
        for i, path in enumerate(tqdm.tqdm(self.raw_paths)):
            mesh = torch_geometric.io.read_ply(path)
            mesh.y = i%10 # set the mesh class (note that FAUST models are ordered by class)
            f2e(mesh)
            if self.pre_filter is not None and not self.pre_filter(mesh):continue
            if self.pre_transform is not None: mesh = self.pre_transform(mesh)
            data_list.append(mesh)

        data, slices = self.collate(data_list)
        if not os.path.exists(self.processed_dir):
            os.mkdir(path)
        torch.save( (data, slices), self.processed_paths[0])

    @property
    def downscale_matrices(self): return self.downscaler.downscale_matrices

    @property
    def downscaled_edges(self): return self.downscaler.downscaled_edges

    @property
    def downscaled_faces(self): return self.downscaler.downscaled_faces


