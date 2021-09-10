from __future__ import print_function

# library imports
import os
import random
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json

# module imports
import torch.utils.data as data
from plyfile import PlyData, PlyElement
from utils.transforms import random_uniform_rotation
from utils.eigenpairs import eigenpairs
from utils.misc import edges_from_faces
from utils.ios import read_obj_verts


# ----------------------------------------------------------------------------------------------------------------------#
#                                                   Functions
# ----------------------------------------------------------------------------------------------------------------------#
# def get_segmentation_classes(root):
#     catfile = os.path.join(root, 'synsetoffset2category.txt')
#     cat = {}
#     meta = {}
#
#     with open(catfile, 'r') as f:
#         for line in f:
#             ls = line.strip().split()
#             cat[ls[0]] = ls[1]
#
#     for item in cat:
#         dir_seg = os.path.join(root, cat[item], 'points_label')
#         dir_point = os.path.join(root, cat[item], 'points')
#         fns = sorted(os.listdir(dir_point))
#         meta[item] = []
#         for fn in fns:
#             token = (os.path.splitext(os.path.basename(fn))[0])
#             meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))
#
#     with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'w') as f:
#         for item in cat:
#             datapath = []
#             num_seg_classes = 0
#             for fn in meta[item]:
#                 datapath.append((item, fn[0], fn[1]))
#
#             for i in tqdm(range(len(datapath))):
#                 l = len(np.unique(np.loadtxt(datapath[i][-1]).astype(np.uint8)))
#                 if l > num_seg_classes:
#                     num_seg_classes = l
#
#             print("category {} num segmentation classes {}".format(item, num_seg_classes))
#             f.write("{}\t{}\n".format(item, num_seg_classes))
#
# def gen_modelnet_id(root):
#     classes = []
#     with open(os.path.join(root, 'train.txt'), 'r') as f:
#         for line in f:
#             classes.append(line.strip().split('/')[0])
#     classes = np.unique(classes)
#     with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
#         for i in range(len(classes)):
#             f.write('{}\t{}\n'.format(classes[i], i))
#
# class ShapeNetDataset(data.Dataset):
#     def __init__(self,
#                  root,
#                  npoints=2500,
#                  classification=False,
#                  class_choice=None,
#                  split='train',
#                  data_augmentation=True):
#         self.npoints = npoints
#         self.root = root
#         self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
#         self.cat = {}
#         self.data_augmentation = data_augmentation
#         self.classification = classification
#         self.seg_classes = {}
#
#         with open(self.catfile, 'r') as f:
#             for line in f:
#                 ls = line.strip().split()
#                 self.cat[ls[0]] = ls[1]
#         #print(self.cat)
#         if not class_choice is None:
#             self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
#
#         self.id2cat = {v: k for k, v in self.cat.items()}
#
#         self.meta = {}
#         splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
#         #from IPython import embed; embed()
#         filelist = json.load(open(splitfile, 'r'))
#         for item in self.cat:
#             self.meta[item] = []
#
#         for file in filelist:
#             _, category, uuid = file.split('/')
#             if category in self.cat.values():
#                 self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid+'.pts'),
#                                         os.path.join(self.root, category, 'points_label', uuid+'.seg')))
#
#         self.datapath = []
#         for item in self.cat:
#             for fn in self.meta[item]:
#                 self.datapath.append((item, fn[0], fn[1]))
#
#         self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
#         print(self.classes)
#         with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
#             for line in f:
#                 ls = line.strip().split()
#                 self.seg_classes[ls[0]] = int(ls[1])
#         self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
#         print(self.seg_classes, self.num_seg_classes)
#
#     def __getitem__(self, index):
#         fn = self.datapath[index]
#         cls = self.classes[self.datapath[index][0]]
#         point_set = np.loadtxt(fn[1]).astype(np.float32)
#         seg = np.loadtxt(fn[2]).astype(np.int64)
#         #print(point_set.shape, seg.shape)
#
#         choice = np.random.choice(len(seg), self.npoints, replace=True)
#         #resample
#         point_set = point_set[choice, :]
#
#         point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
#         dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
#         point_set = point_set / dist #scale
#
#         if self.data_augmentation:
#             theta = np.random.uniform(0,np.pi*2)
#             rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
#             point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
#             point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter
#
#         seg = seg[choice]
#         point_set = torch.from_numpy(point_set)
#         seg = torch.from_numpy(seg)
#         cls = torch.from_numpy(np.array([cls]).astype(np.int64))
#
#         if self.classification:
#             return point_set, cls
#         else:
#             return point_set, seg
#
#     def __len__(self):
#         return len(self.datapath)

# class ModelNetDataset(data.Dataset):
#     def __init__(self,
#                  root,
#                  npoints=2500,
#                  split='train',
#                  data_augmentation=True,
#                  classification=True):
#         self.npoints = npoints
#         self.root = root
#         self.split = split
#         self.data_augmentation = data_augmentation
#         self.fns = []
#         with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
#             for line in f:
#                 self.fns.append(line.strip())
#
#         self.cat = {}
#         with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'r') as f:
#             for line in f:
#                 ls = line.strip().split()
#                 self.cat[ls[0]] = int(ls[1])
#
#         print(self.cat)
#         self.classes = list(self.cat.keys())
#
#     def __getitem__(self, index):
#         fn = self.fns[index]
#         cls = self.cat[fn.split('/')[0]]
#         with open(os.path.join(self.root, fn), 'rb') as f:
#             plydata = PlyData.read(f)
#         pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
#         choice = np.random.choice(len(pts), self.npoints, replace=True)
#         point_set = pts[choice, :]
#
#         point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
#         dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
#         point_set = point_set / dist  # scale
#
#         if self.data_augmentation:
#             theta = np.random.uniform(0, np.pi * 2)
#             rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
#             point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
#             point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter
#
#         point_set = torch.from_numpy(point_set.astype(np.float32))
#         cls = torch.from_numpy(np.array([cls]).astype(np.int64))
#         return point_set, cls
#
#
#     def __len__(self):
#         return len(self.fns)


# class FaustDataset(data.Dataset):
#     def __init__(self, root, split='train', data_augmentation=False):
#         self.root = root
#         self.split = split
#         self.data_augmentation = data_augmentation
#
#         # create list of valid files
#         self.fns = []
#         for file in os.listdir(self.root):
#             if file.endswith(".ply"):
#                 self.fns.append(file)
#         assert len(self.fns) == 100 , "assumed that there are 100 train examples"
#
#         # split tran\test
#         if self.split == 'train':
#             self.fns = self.fns[0:70]
#         elif self.split == 'validation':
#             self.fns = self.fns[70:85]
#         else:
#             self.fns = self.fns[85:]
#
#         # self.set_targets()
#         # self.num_vertices = 6890  # hardcoded for now
#
#
#     def __getitem__(self, index):
#         if index < 10 :
#             fn = 'tr_reg_00'+str(index)+'.ply'
#         else:
#             fn = 'tr_reg_0' + str(index)+'.ply'
#
#         with open(os.path.join(self.root, fn), 'rb') as f:
#             plydata = PlyData.read(f)
#         x = plydata['vertex']['x']
#         y = plydata['vertex']['y']
#         z = plydata['vertex']['z']
#         v = np.column_stack((x, y, z))
#         if 'red' in plydata['vertex']._property_lookup:
#             r = plydata['vertex']['red']
#             g = plydata['vertex']['green']
#             b = plydata['vertex']['blue']
#             rgb = np.column_stack((r, g, b))
#             rgb = torch.from_numpy(rgb.astype(np.float32))
#         else:
#             rgb = None
#         f = np.stack(plydata['face']['vertex_indices'])
#         faces = torch.from_numpy(f).type(torch.long).to(run_config['DEVICE'])
#         self.rgb = rgb
#
#
#         # calculate edges from faces for local euclidean similarity
#         if (self.split == 'train') & (LOSS == 'local_euclidean'):
#             e = edges_from_faces(f)
#             edges = torch.from_numpy(e).type(torch.long).to(run_config['DEVICE'])
#         else:
#             edges = 0
#
#         # # center and scale
#         v = v - np.expand_dims(np.mean(v, axis=0), 0)  # center
#         # dist = np.max(np.sqrt(np.sum(v ** 2, axis=1)), 0)
#         # v = v / dist  # scale
#
#         if self.data_augmentation:
#             # theta = np.random.uniform(0, np.pi * 2)
#             # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
#             # v[:, [0, 2]] = v[:, [0, 2]].dot(rotation_matrix)  # random Z rotation
#
#             # random unitary rotation
#             r = random_uniform_rotation()
#             v = v @ r
#             # random translation
#             v += np.random.normal(0, 0.01, size=(1, 3))
#
#         v = torch.from_numpy(v.astype(np.float32))
#         cls = torch.from_numpy(np.array([index % 10]).astype(np.int64))
#
#         # calculate laplacian eigenvectors matrix and areas
#         if TRAINING_CLASSIFIER:
#             eigvals, eigvecs, vertex_area = 0, 0, 0
#             targets = 0
#         else:
#             eigvals, eigvecs, vertex_area = eigenpairs(v, faces, K, double_precision=True)
#             # eigvals = eigvals.to(run_config['DEVICE'])
#             eigvecs = eigvecs.to(run_config['DEVICE'])
#             vertex_area = vertex_area.to(run_config['DEVICE'])
#
#             # draw new targets every time a new data is created
#             targets = self.set_targets()
#             targets = targets.to(run_config['DEVICE'])[index]
#
#         return v.to(run_config['DEVICE']), cls.to(run_config['DEVICE']), eigvals, eigvecs, vertex_area, \
#                targets, faces, edges
#
#     def __len__(self):
#         return len(self.fns)
#
#     def set_targets(self):
#         # draw random target for each shape
#         targets = np.zeros(len(self))-1
#         for i in np.arange(0, len(self)):
#             target = random.randint(0, 9)
#             while target == (i % 10):
#                 target = random.randint(0, 9)
#             targets[i] = target
#         targets = torch.from_numpy(targets).long().to(run_config['DEVICE'])
#         return targets



class FaustDatasetInMemory(data.Dataset):

    def __init__(self, run_config, root, split='train', data_augmentation=False):
        self.run_config = run_config
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        run_config['DEVICE'] = run_config['DEVICE']
        CALCULATE_EIGENVECTORS = run_config['CALCULATE_EIGENVECTORS']

        # create list of valid files
        self.fns = []
        for file in os.listdir(self.root):
            if file.endswith(".ply"):
                self.fns.append(file)
        assert len(self.fns) == 100, "assumed that there are 100 train examples"

        # Define Classes
        cls = np.arange(0,100) % 10
        # sort - very important
        list.sort(self.fns)
        # split tran\test
        if self.split == 'train':
            self.fns = self.fns[0:70]
            self.cls = cls[0:70]
        elif self.split == 'validation':
            self.fns = self.fns[70:85]
            self.cls = cls[70:85]
        elif self.split == "test":
            self.fns = self.fns[85:]
            self.cls = cls[85:]

        self.cls = torch.from_numpy(self.cls.astype(np.int64)).to(run_config['DEVICE'])

        # load all dataset to memory
        self.v = []
        self.faces = []
        self.edges = []
        self.eigvecs = []
        self.vertex_area = []
        for index, fn in enumerate(self.fns):

            with open(os.path.join(self.root, fn), 'rb') as f:
                plydata = PlyData.read(f)
            x = plydata['vertex']['x']
            y = plydata['vertex']['y']
            z = plydata['vertex']['z']
            v = np.column_stack((x, y, z))
            v = v - np.expand_dims(np.mean(v, axis=0), 0)  # center
            v = torch.from_numpy(v.astype(np.float32))
            self.v.append(v)

            f = np.stack(plydata['face']['vertex_indices'])
            self.faces = torch.from_numpy(f).type(torch.long)
            # self.faces.append(faces)


        # calculate edges from faces for local euclidean similarity
        if (self.split == 'train') & (self.run_config['LOSS'] == 'local_euclidean'):
            e = edges_from_faces(self.faces)
            edges = torch.from_numpy(e).type(torch.long)
        else:
            edges = 0

        # Calculate Eigenvectors and areas of all the data
        if CALCULATE_EIGENVECTORS and (data_augmentation == False):
            for v in self.v:
                eigvals, eigvecs, vertex_area = eigenpairs(v, self.faces, self.run_config['K'], double_precision=True)
                self.eigvecs.append(eigvecs)
                self.vertex_area.append(vertex_area)


        self.edges = edges
        self.targets = self.set_targets()



    def __getitem__(self, index):

        # # center and scale
        v = self.v[index] #- np.expand_dims(np.mean(self.v[index], axis=0), 0)  # center
        # dist = np.max(np.sqrt(np.sum(v ** 2, axis=1)), 0)
        # v = v / dist  # scale
        if self.data_augmentation:
            # theta = np.random.uniform(0, np.pi * 2)
            # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            # v[:, [0, 2]] = v[:, [0, 2]].dot(rotation_matrix)  # random Z rotation

            # random unitary rotation
            r = random_uniform_rotation()
            v = v @ r

            # random translation
            # jitter = np.random.normal(0, 0.01, size=(1, 3))
            # v += jitter

        # v = torch.from_numpy(np.random.rand(6890,3).astype(np.float32))  # TODO remove - it is debug!


        # calculate laplacian eigenvectors matrix and areas
        if not self.run_config['CALCULATE_EIGENVECTORS']:
            eigvals, eigvecs, vertex_area = 0, 0, 0
        else:

            if self.data_augmentation == True:
                eigvals, eigvecs, vertex_area = eigenpairs(v, self.faces, self.run_config['K'], double_precision=True)
                # eigvals = eigvals.to(run_config['DEVICE'])
                eigvecs = eigvecs.to(self.run_config['DEVICE'])
                vertex_area = vertex_area.to(self.run_config['DEVICE'])
            else:
                eigvals = 0
                eigvecs = 0
                # eigvecs = self.eigvecs[index].to(self.run_config['DEVICE'])
                vertex_area = self.vertex_area[index].to(self.run_config['DEVICE'])

            # draw new targets every time a new data is created
            # targets = self.set_targets()
        targets = self.targets[index]

        return v.to(self.run_config['DEVICE']), self.cls[index],  eigvals, eigvecs, vertex_area \
            , targets, self.faces, self.edges

    def __len__(self):
        return len(self.fns)

    def set_targets(self):
        # draw random target for each shape
        targets = np.zeros(len(self))-1
        for i in np.arange(0, len(self)):
            target = random.randint(0, 9)
            while target == (i % 10):
                target = random.randint(0, 9)
            targets[i] = target
        targets = torch.from_numpy(targets).long().to(self.run_config['DEVICE'])
        return targets

# class Shrec14Dataset(data.Dataset):
#     def __init__(self, root, split='train', data_augmentation=False):
#         self.root = root
#         self.split = split
#         self.data_augmentation = data_augmentation
#
#         # create list of valid files
#         self.fns = []
#         for file in os.listdir(self.root):
#             if file.endswith(".obj"):
#                 self.fns.append(file.split('.')[0])
#         assert len(self.fns) == 400 , "assumed that there are 400 train examples"
#         # sort - very important
#         list.sort(self.fns, key=int)
#         self.fns = [os.path.join(self.root, s + '.obj') for s in self.fns]
#
#         # split tran\test
#         if self.split == 'train':
#             self.fns = self.fns[0:320]
#         elif self.split == 'validation':
#             self.fns = self.fns[320:360]
#         else:
#             self.fns = self.fns[360:]
#
#
#     def __getitem__(self, index):
#         assert index >= 0 & index < 400, "bad index"
#
#         v, f = read_obj_verts(self.fns[index], 15000, 29996)
#         faces = torch.from_numpy(f).type(torch.long).to(DEVICE)
#
#         # calculate edges from faces for local euclidean similarity
#         if (self.split == 'train') & (LOSS == 'local_euclidean'):
#             e = edges_from_faces(f)
#             edges = torch.from_numpy(e).type(torch.long).to(DEVICE)
#         else:
#             edges = 0
#
#         # center
#         v = v - np.expand_dims(np.mean(v, axis=0), 0)  # center
#
#         if self.data_augmentation:
#             # theta = np.random.uniform(0, np.pi * 2)
#             # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
#             # v[:, [0, 2]] = v[:, [0, 2]].dot(rotation_matrix)  # random Z rotation
#
#             # random unitary rotation
#             r = random_uniform_rotation()
#             v = v @ r
#             # random translation
#             v += np.random.normal(0, 0.01, size=(1, 3))
#
#         v = torch.from_numpy(v.astype(np.float32))
#         cls = torch.from_numpy(np.array([index % 10]).astype(np.int64))
#
#         # calculate laplacian eigenvectors matrix and areas
#         if TRAINING_CLASSIFIER:
#             eigvals, eigvecs, vertex_area = 0, 0, 0
#             targets = 0
#         else:
#             eigvals, eigvecs, vertex_area = eigenpairs(v, faces, K, double_precision=True)
#             # eigvals = eigvals.to(DEVICE)
#             eigvecs = eigvecs.to(DEVICE)
#             vertex_area = vertex_area.to(DEVICE)
#
#             # draw new targets every time a new data is created
#             targets = self.set_targets()
#             targets = targets.to(DEVICE)[index]
#
#         return v.to(DEVICE), cls.to(DEVICE), eigvals, eigvecs, vertex_area, \
#                targets, faces, edges
#
#     def __len__(self):
#         return len(self.fns)
#
#     def set_targets(self):
#         # draw random target for each shape
#         targets = np.zeros(len(self))-1
#         for i in np.arange(0, len(self)):
#             target = random.randint(0, 9)
#             while target == (i % 10):
#                 target = random.randint(0, 9)
#             targets[i] = target
#         targets = torch.from_numpy(targets).long().to(DEVICE)
#         return targets


class Shrec14DatasetInMemory(data.Dataset):
    def __init__(self, run_config, root, split='train', data_augmentation=False):
        self.run_config = run_config
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        run_config['DEVICE'] = run_config['DEVICE']
        CALCULATE_EIGENVECTORS = run_config['CALCULATE_EIGENVECTORS']

        # create list of valid files
        self.fns = []
        for file in os.listdir(self.root):
            if file.endswith(".obj"):
                self.fns.append(file.split('.')[0])
        assert len(self.fns) == 400, "assumed that there are 400 train examples"
        # Define Classes
        cls = np.arange(0, 400) % 10
        # sort - very important
        self.fns = [int(x) for x in self.fns]
        list.sort(self.fns)
        self.fns = [str(x) for x in self.fns]
        self.fns = [os.path.join(self.root, s + '.obj') for s in self.fns]

        # split tran\test
        if self.split == 'train':
            self.fns = self.fns[0:self.run_config['DATASET_TRAIN_SIZE']]
            self.cls = cls[0:self.run_config['DATASET_TRAIN_SIZE']]
        elif self.split == 'validation':
            self.fns = self.fns[self.run_config['DATASET_TRAIN_SIZE']:360]
            self.cls = cls[self.run_config['DATASET_TRAIN_SIZE']:360]
        elif self.split == "test":
            self.fns = self.fns[360:]
            self.cls = cls[360:]

        self.cls = torch.from_numpy(self.cls.astype(np.int64)).to(self.run_config['DEVICE'])
        # load all dataset to memory
        self.v = []
        self.faces = []
        self.edges = []
        for index, fn in enumerate(self.fns):

            with open(os.path.join(self.root, fn), 'rb') as f:
                v, f_ = read_obj_verts(self.fns[index], 6892, 13780)
            v = v - np.expand_dims(np.mean(v, axis=0), 0)  # center
            v = torch.from_numpy(v.astype(np.float32))
            self.v.append(v)

            f = np.stack(f_)
            faces = torch.from_numpy(f).type(torch.long)
            self.faces.append(faces)

        # calculate edges from faces for local euclidean similarity
        if (self.split == 'train') & (self.run_config['LOSS'] == 'local_euclidean'):
            e = edges_from_faces(self.faces)
            edges = torch.from_numpy(e).type(torch.long).to(self.run_config['DEVICE'])
        else:
            edges = 0

        self.edges = edges
        self.targets = self.set_targets()


    def __getitem__(self, index):
        assert index >= 0 & index < 400, "bad index"

        # # center and scale
        v = self.v[index] # - np.expand_dims(np.mean(self.v[index], axis=0), 0)  # center
        # dist = np.max(np.sqrt(np.sum(v ** 2, axis=1)), 0)
        # v = v / dist  # scale
        if self.data_augmentation:
            # theta = np.random.uniform(0, np.pi * 2)
            # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            # v[:, [0, 2]] = v[:, [0, 2]].dot(rotation_matrix)  # random Z rotation

            # random unitary rotation
            r = random_uniform_rotation()
            v = v @ r

            # random translation
            # jitter = np.random.normal(0, 0.01, size=(1, 3))
            # v += jitter


        # calculate laplacian eigenvectors matrix and areas
        if not self.run_config['CALCULATE_EIGENVECTORS']:
            eigvals, eigvecs, vertex_area = 0, 0, 0
        else:
            eigvals, eigvecs, vertex_area = eigenpairs(v, self.faces[index], self.run_config['K'], double_precision=True)
            # eigvals = eigvals.to(DEVICE)
            # eigvecs = eigvecs.to(DEVICE)
            vertex_area = vertex_area.to(self.run_config['DEVICE'])

            # draw new targets every time a new data is created
            # targets = self.set_targets()
        targets = self.targets[index]

        return v.to(self.run_config['DEVICE']), self.cls[index], eigvals, eigvecs, vertex_area \
            , targets, self.faces[index], self.edges

    def __len__(self):
        return len(self.fns)

    def set_targets(self):
        # draw random target for each shape
        targets = np.zeros(len(self)) - 1
        for i in np.arange(0, len(self)):
            target = random.randint(0, 9)
            while target == (i % 10):
                target = random.randint(0, 9)
            targets[i] = target
        targets = torch.from_numpy(targets).long().to(self.run_config['DEVICE'])
        return targets


if __name__ == '__main__':
    pass

