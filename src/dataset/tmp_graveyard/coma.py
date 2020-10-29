import os.path as osp
from glob import glob
import random

import torch
from torch_geometric.data import InMemoryDataset, extract_zip
from torch_geometric.datasets import CoMA
from torch_geometric.io import read_ply

import dataset.downscale

class CoMADataset(CoMA):
    def __init__(self, root, train=True, category_samples=50, test_categories=[0,1]):
        self.test_categories = test_categories
        self.category_samples = category_samples
        super(CoMA, self).__init__(root, None, None, None)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)
        self.data.y = self.data.subject
        self.slices["y"] = self.slices["subject"]

    def process(self):
        folders = sorted(glob(osp.join(self.raw_dir, 'FaceTalk_*')))
        if len(folders) == 0:
            extract_zip(self.raw_paths[0], self.raw_dir, log=False)
            folders = sorted(glob(osp.join(self.raw_dir, 'FaceTalk_*')))

        train_data_list, test_data_list = [], []
        for fi, folder in enumerate(folders):
            for ci, category in enumerate(self.categories):
                files = sorted(glob(osp.join(folder, category, '*.ply')))
                files = random.sample(files, k=min(self.category_samples, len(files)))
                for j, f in enumerate(files):
                    data = read_ply(f)
                    data.category = torch.tensor([ci], dtype=torch.long)
                    data.subject = torch.tensor([fi], dtype=torch.long)
                    if self.pre_filter is not None and\
                       not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    if not ci in self.test_categories:
                        train_data_list.append(data)
                    else:
                        test_data_list.append(data)

        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(test_data_list), self.processed_paths[1])

