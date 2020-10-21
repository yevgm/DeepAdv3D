import os
from os.path import basename


import numpy as np
import torch
import torch_sparse as tsparse
import torch_geometric
import torch_geometric.data
import torch_geometric.io as gio
import torch_geometric.transforms as transforms
import tqdm

import utils
from utils import generate_transform_matrices
import dataset.downscale as dscale
from utils.transforms import Move, Rotate, ToDevice

from torch_geometric.data.dataloader import DataLoader

class Shrec14Dataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, 
        root:str, 
        device:torch.device=torch.device("cpu"),
        train:bool=True, test:bool=True,
        transform_data:bool=True):
        self.url = 'http://www.cs.cf.ac.uk/shaperetrieval/shrec14/'

        if transform_data:
            # rotate and move
            transform = transforms.Compose([  
                transforms.Center(),
#                 transforms.RandomScale((0.8,1.2)),
                Rotate(dims=[1]), 
                Move(mean=[0,0,0], std=[0.05,0.05,0.05]), 
                transforms.RandomTranslate(0.01),
                ToDevice(device)])
        else:
            transform = ToDevice(device)

        # center each mesh into its centroid
        super().__init__(root=root, transform=transform, pre_transform=transforms.Center())

        self.data, self.slices = torch.load(self.processed_paths[0])

        testset_slice, trainset_slice = list(range(0,40))+list(range(200,240)) , list(range(40,200))+list(range(240,400))
        if train and not test:
            self.data, self.slices = self.collate([self[i] for i in trainset_slice])

        elif not train and test:
            self.data, self.slices = self.collate([self[i] for i in testset_slice])


    @property
    def raw_file_names(self):
        tofilename =  lambda x : "Data/{}.obj".format(x)
        return [tofilename(fi) for fi in range(400)]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        print('Downloaded?')
#         raise RuntimeError(
#             'Dataset not found. Please download it from {} and move it to {}'.format(self.url, self.raw_dir))
    
    def process(self):
        # Read data into huge `Data` list.
        face2edge = transforms.FaceToEdge(remove_faces=False)
        data_list=[]
        num_classes = 10
        for path in tqdm.tqdm(self.raw_paths):
            i = int(basename(path)[:-4])
            mesh = torch_geometric.io.read_obj(path)
            mesh.y = i%num_classes
            mesh.subject = int(i/num_classes)
            face2edge(mesh)
            mesh.idx = i
            if self.pre_filter is not None and not self.pre_filter(mesh):continue
            if self.pre_transform is not None: mesh = self.pre_transform(mesh)
            data_list.append(mesh)
        data, slices = self.collate(data_list)
        torch.save( (data, slices), self.processed_paths[0])