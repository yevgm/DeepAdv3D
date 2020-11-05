# built-in libraries
import sys
import os
from datetime import datetime

# third party libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
import torch.nn.functional as func
import random

# variable definitions
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath('__file__')),""))  # need ".." in linux
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# DEVICE = torch.device("cpu")
SRC_DIR = os.path.join(REPO_ROOT,"src")
FAUST = os.path.join(REPO_ROOT,"datasets/faust")
PARAMS_FILE = os.path.join(REPO_ROOT, "model_data/FAUST10_pointnet_rot.pt")

# repository modules
sys.path.insert(0, SRC_DIR)
from utils.ios import write_off
import vista.adv_plotter
from vista.adv_plotter import show_perturbation, show_all_perturbations
from vista.geom_vis import plot_mesh, plot_mesh_montage
import adversarial.output_handler as op
import vista.animation
from vista.animation import animate, multianimate


import models
import ntrain
import dataset
import utils
from models.pointnet import SimplePointNet
from models.Origin_pointnet import PointNetCls
from dataset.data_loaders import FaustDataset
import adversarial.carlini_wagner as cw
from adversarial.carlini_wagner import CWBuilder, LowbandPerturbation


def load_datasets(train_batch=8, test_batch=20):
    train_dataset = FaustDataset(
        root=os.path.join(FAUST, r'raw'),
        classification=True,
        split='train',
        data_augmentation=True)

    test_dataset = FaustDataset(
        root=os.path.join(FAUST, r'raw'),
        classification=True,
        split='test',
        data_augmentation=False)

    trainLoader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch,
                                               shuffle=True,
                                               num_workers=4)
    testLoader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=test_batch,
                                               shuffle=False,
                                               num_workers=4)
    # load data in different format for Adversarial code
    traindata = dataset.FaustDataset(FAUST, device=DEVICE, train=True, test=False, transform_data=True)
    testdata = dataset.FaustDataset(FAUST, device=DEVICE, train=False, test=True, transform_data=True)

    return trainLoader,testLoader,traindata,testdata

def random_uniform_rotation(dim=3):
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = np.random.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat = np.eye(dim)
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H

def rotation_x(angle, degrees=True):
    if degrees:
        angle = np.deg2rad(angle)
    cx = np.cos(angle)
    sx = np.sin(angle)
    return np.array([[1, 0, 0],
                     [0, cx, sx],
                     [0, -sx, cx]])


def rotation_y(angle, degrees=True):
    if degrees:
        angle = np.deg2rad(angle)
    cy = np.cos(angle)
    sy = np.sin(angle)
    return np.array([[cy, 0, -sy],
                     [0, 1, 0],
                     [sy, 0, cy]])


def rotation_z(angle, degrees=True):
    if degrees:
        angle = np.deg2rad(angle)
    cz = np.cos(angle)
    sz = np.sin(angle)
    return np.array([[cz, sz, 0],
                     [-sz, cz, 0],
                     [0, 0, 1]])

def show_model_accuracy(PARAMS_FILE, model):
    loss_values, test_mean_loss, test_accuracy = ntrain.train(
        train_data=trainLoader,
        test_data=testLoader,
        classifier=model,
        batchSize=20,
        parameters_file=PARAMS_FILE,
        learning_rate=1e-3,
        train=False)

    print('test mean loss:', test_mean_loss, ' test_accuracy:', test_accuracy)

if __name__ == "__main__":

    model = PointNetCls(k=10, feature_transform=False, global_transform=False)
    model = model.to(DEVICE)
    # print(model)
    batchsize = 8
    trainLoader, testLoader, traindata, testdata = load_datasets(train_batch=batchsize, test_batch=20)

    # train network
    loss_values, test_mean_loss, test_accuracy = ntrain.train(
                                                            train_data=trainLoader,
                                                            test_data=testLoader,
                                                            classifier=model,
                                                            batchSize=batchsize,
                                                            parameters_file=PARAMS_FILE,
                                                            epoch_number=50,
                                                            learning_rate=4e-3,
                                                            train=True)
    # # temp train visualizer - in the future : add tensorboard?
    print('test mean loss:',test_mean_loss,' test_accuracy:',test_accuracy)
    loss_values = np.array(loss_values)
    sliced_loss = loss_values[0::5]#sliced

    fig, axs = plt.subplots(2)
    fig.suptitle('losses')
    axs[0].plot(np.arange(1,len(sliced_loss)+1,1), sliced_loss)
    axs[1].plot(np.arange(1,len(loss_values)+1,1), loss_values)

    axs[0].set(xlabel='5*batches index', ylabel='loss')
    axs[0].grid()
    axs[1].set(xlabel='batches index', ylabel='loss')
    axs[1].grid()
    plt.show()
    #
    # show_model_accuracy(PARAMS_FILE, model)
    model.load_state_dict(torch.load(PARAMS_FILE, map_location=DEVICE))
    model.eval()

    count = 0
    number_of_tests = 100
    # rotation invariance testing
    for i in np.arange(0,number_of_tests,1):
        R = torch.Tensor(random_uniform_rotation()).to(DEVICE)

        # v_orig = trainLoader.dataset[i][0]
        v = trainLoader.dataset[i][0]
        faces = trainLoader.dataset.f
        true_y = trainLoader.dataset[i][1]
        Z, _, _ = model(v)
        f = torch.nn.functional.log_softmax(Z, dim=1)
        pred_y = f.argmax()

        v_rot = torch.mm(v, R)
        # v_rot = np.abs(np.random.normal()) * v
        # v_rot = v + np.random.normal(0, 0.5,size=(1, 3)).astype('f')
        # theta = np.random.uniform(0, np.pi * 2)
        # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        # v = v.numpy()
        # v[:, [0, 2]] = v[:, [0, 2]].dot(rotation_matrix)  # random rotation
        # v = torch.from_numpy(v)
        Z_rot, _, _ = model(v_rot)
        f_rot = torch.nn.functional.log_softmax(Z_rot, dim=1)
        pred_y_rot = f_rot.argmax()

        plot_mesh_montage([v, v_rot], [faces, faces])



        count += pred_y_rot==pred_y
        print(pred_y_rot==pred_y)

    print("accuracy is :", count/float(number_of_tests))


