from os import listdir, mkdir
from os.path import isfile, join, exists, split
import tarfile

import tqdm
import torch
import torch_geometric.data
import torch_geometric.transforms as transforms

import dataset.downscale as dscale
from utils.transforms import Move, Rotate, ToDevice


class SmalDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, 
        root:str, 
        device:torch.device=torch.device("cpu"),
        train:bool=True, 
        test:bool=True,
        transform_data:bool=True):

        self.url = 'https://drive.google.com/file/d/1dp4sMvZ8cmIIITE-qj6zYpZb0-v-4Kgf/view?usp=sharing'
        self.categories = ["big_cats","cows","dogs","hippos","horses"]

        # center each mesh into its centroid
        pre_transform = transforms.Center()

        # transform
        if transform_data:
            # rotate and move
            transform = transforms.Compose([
                Move(mean=[0,0,0], std=[0.05,0.05,0.05]), 
                Rotate(dims=[0,1,2]), 
                ToDevice(device)])
        else:
            transform=ToDevice(device)

        super().__init__(root=root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.downscaler = dscale.Downscaler(
            filename=join(self.processed_dir,"ds"), mesh=self.get(0), factor=2)


        if train and not test:
            self.data, self.slices = self.collate([self.get(i) for i in range(len(self)) if self.get(i).pose < 16])
        elif not train and test:
            self.data, self.slices = self.collate([self.get(i) for i in range(len(self)) if self.get(i).pose >= 16])

    @property
    def raw_file_names(self):
        files = sorted(listdir(self.raw_dir))
        categ_files = [f for f in files if isfile(join(self.raw_dir, f)) and f.split(".")[-1]=="ply"]
        return categ_files
        
    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download {} from {} and move it to {}'.format(self.raw_file_names, self.url, self.raw_dir))
    
    def process(self):
        data_list = []
        f2e = transforms.FaceToEdge(remove_faces=False)
        for pindex, path in enumerate(tqdm.tqdm(self.raw_paths)):
            mesh = torch_geometric.io.read_ply(path)
            f2e(mesh)
            tmp = split(path)[1].split(".")[0].split("_")
            model_str, pose_str = tmp[-2], tmp[-1]
            category = "_".join(tmp[:-2])
            mesh.model = int(model_str[5:])
            mesh.pose = int(pose_str[4:])
            mesh.y = self.categories.index(category)
            if self.pre_filter is not None and not self.pre_filter(mesh) : continue
            if self.pre_transform is not None: mesh = self.pre_transform(mesh)
            data_list.append(mesh)
        data, slices = self.collate(data_list)
        torch.save( (data, slices), self.processed_paths[0])

    @property
    def downscale_matrices(self): return self.downscaler.downscale_matrices

    @property
    def downscaled_edges(self): return self.downscaler.downscaled_edges

    @property
    def downscaled_faces(self): return self.downscaler.downscaled_faces
