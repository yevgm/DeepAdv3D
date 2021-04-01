# This is the main script that runs out model1:

# Using the central article's method in the neural setting
# revolves around regressing for the optimal smooth deformation field parameters needed to optimally deform the target
# shape to achieve target/untargeted adversarial attack success
# Architecture: Simple PointNet (Without T-Nets, see implementation in the Shape Completion Repo)
# + switch last layer to regression layer

# variable definitions
from config import *

# repository modules
from models.Origin_pointnet import PointNetCls, Regressor
from model1.deep_adv_3d import *
import dataset
from dataset.data_loaders import FaustDataset, FaustDatasetInMemory

# # geometric loader
# def load_datasets_for_regressor(train_batch=8, test_batch=20):
#     # it uses carlini's FaustDataset class that inherits from torch_geometric.data.InMemoryDataset
#     traindata = dataset.FaustDataset(FAUST, device=DEVICE, train=True, test=False, transform_data=True)
#     testdata = dataset.FaustDataset(FAUST, device=DEVICE, train=False, test=True, transform_data=True)
#
#     trainLoader = torch.utils.data.DataLoader(train_dataset,
#                                                    batch_size=train_batch,
#                                                    shuffle=True,
#                                                    num_workers=4)
#     testLoader = torch.utils.data.DataLoader(test_dataset,
#                                                    batch_size=test_batch,
#                                                    shuffle=False,
#                                                    num_workers=4)
#
#     return trainLoader, testLoader, traindata, testdata

def load_datasets(train_batch=8, test_batch=20):
    # here we use FaustDataset class that inherits from torch.utils.data.Dataloader. it's a map-style dataset.
    if LOAD_WHOLE_DATA_TO_MEMORY:
        train_dataset = FaustDatasetInMemory(
            root=os.path.join(FAUST, r'raw'),
            split='train',
            data_augmentation=TRAIN_DATA_AUG)

        test_dataset = FaustDatasetInMemory(
            root=os.path.join(FAUST, r'raw'),
            split='test',
            data_augmentation=TEST_DATA_AUG)
    else:
        train_dataset = FaustDataset(
            root=os.path.join(FAUST, r'raw'),
            split='train',
            data_augmentation=TRAIN_DATA_AUG)

        test_dataset = FaustDataset(
            root=os.path.join(FAUST, r'raw'),
            split='test',
            data_augmentation=TEST_DATA_AUG)

    trainLoader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch,
                                               shuffle=True,
                                               num_workers=NUM_WORKERS)
    testLoader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=test_batch,
                                               shuffle=SHUFFLE_TEST_DATA,
                                               num_workers=NUM_WORKERS)

    return trainLoader, testLoader

# # TODO: remove this block - it's debug
# def random_uniform_rotation(dim=3):
#     H = np.eye(dim)
#     D = np.ones((dim,))
#     for n in range(1, dim):
#         x = np.random.normal(size=(dim - n + 1,))
#         D[n - 1] = np.sign(x[0])
#         x[0] -= D[n - 1] * np.sqrt((x * x).sum())
#         # Householder transformation
#         Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
#         mat = np.eye(dim)
#         mat[n - 1:, n - 1:] = Hx
#         H = np.dot(H, mat)
#         # Fix the last sign such that the determinant is 1
#     D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
#     # Equivalent to np.dot(np.diag(D), H) but faster, apparently
#     H = (D * H.T).T
#     return H

if __name__ == '__main__':
    # Data Loading and pre-processing
    trainLoader, testLoader = load_datasets(train_batch=TRAIN_BATCH_SIZE, test_batch=TEST_BATCH_SIZE)

    # classifier and model definition
    classifier = PointNetCls(k=10, feature_transform=False, global_transform=False)
    classifier.load_state_dict(torch.load(PARAMS_FILE, map_location=DEVICE), strict=CLS_STRICT_PARAM_LOADING)  #strict = False for dropping running mean and var of train batchnorm
    model = Regressor(numVertices=K)  # K - additive vector field (V) dimension in eigen-space

    # classifier = classifier.eval()
    # # TODO: remove this block - it's debug
    # for i, data in enumerate(trainLoader):
    #     orig_vertices, label, _, eigvecs, vertex_area, targets, faces, edges = data
    #     # plot_mesh_montage([orig_vertices.transpose(2, 1)[0].T], [faces[0]])
    #     label = label[:, 0]
    #     logits, _, _ = classifier(orig_vertices.transpose(2, 1))
    #     s = logits.data.max(1)[1]
    #     t = s.eq(label).cpu().sum()
    #     a=1

    # count = 0 # TODO: remove this block - it's debug
    # number_of_tests = 20
    # import numpy as np
    # # rotation invariance testing
    # for i in np.arange(0, number_of_tests, 1):
    #     R = torch.Tensor(random_uniform_rotation())
    #
    #     v_orig = testLoader.dataset[i % 20][0]
    #     true_y = testLoader.dataset[i % 20][1]
    #     v = testLoader.dataset[i % 20][0]
    #     # faces = testLoader.dataset.f
    #     Z, _, _ = classifier(v)
    #     # f = torch.nn.functional.log_softmax(Z, dim=1)
    #     pred_y = Z.argmax()
    #
    #     v_rot = v
    #     v_rot = torch.mm(v, R)
    #     # v_rot = np.abs(np.random.normal()) * v
    #     v_rot = v_rot + torch.Tensor(np.random.normal(0, 0.01, size=(1, 3)).astype('f'))
    #     # theta = np.random.uniform(0, np.pi * 2)
    #     # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    #     # v = v.cpu().numpy()
    #     # v[:, [0, 2]] = v[:, [0, 2]].dot(rotation_matrix)  # random rotation
    #     # v = torch.from_numpy(v).to(DEVICE)
    #     Z_rot, _, _ = classifier(v_rot)
    #     # f_rot = torch.nn.functional.log_softmax(Z_rot, dim=1)
    #     pred_y_rot = Z_rot.argmax()
    #
    #     # plot_mesh_montage([v, v_rot], [faces, faces])
    #
    #     count += pred_y_rot == pred_y
    #
    # print("accuracy is :", count / float(number_of_tests))

    train_ins = trainer(train_data=trainLoader, test_data=testLoader,
                        model=model, classifier=classifier)
<<<<<<< HEAD
    train_ins.train()
=======

    # train network
    train_ins.train()
    # evaluate network
>>>>>>> refs/remotes/origin/main
    # train_ins.evaluate(TEST_PARAMS_DIR)

