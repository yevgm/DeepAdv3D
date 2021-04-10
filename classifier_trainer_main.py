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
from config import *


# repository modules
sys.path.insert(0, SRC_DIR)
from utils.ios import write_off
import vista.adv_plotter
from vista.adv_plotter import show_perturbation, show_all_perturbations
from vista.geom_vis import plot_mesh, plot_mesh_montage
import adversarial.output_handler as op
import vista.animation
from vista.animation import animate, multianimate

import classifier_trainer
from models.pointnet import PointNet
from model_trainer_main import load_datasets
from utils.torch.nn import *
# from dataset.faust import FaustDataset as FaustData

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


if __name__ == "__main__":

    model = PointNet(k=10, feature_transform=False, global_transform=False)
    model = model.to(DEVICE)

    # set seed for all platforms
    set_determinsitic_run()

    # Data Loading and pre-processing
    trainLoader, testLoader = load_datasets(dataset=DATASET_NAME, train_batch=TRAIN_BATCH_SIZE,
                                            test_batch=TEST_BATCH_SIZE)

    classifier_trainer.train(train_data=trainLoader, test_data=testLoader, classifier=model)


    # classifier_trainer.evaluate(testLoader, classifier)
    # ntrain.train(
    #             train_data=trainLoader,
    #             test_data=testLoader,
    #             classifier=model,
    #             batchSize=TRAIN_BATCH_SIZE,
    #             parameters_file=PARAMS_FILE,
    #             epoch_number=N_EPOCH,
    #             learning_rate=LR,
    #             train=True)


    # model.load_state_dict(torch.load(PARAMS_FILE, map_location=DEVICE))
    # model.eval()
    #
    # count = 0
    # number_of_tests = 100
    # # rotation invariance testing
    # for i in np.arange(0, number_of_tests,1):
    #     R = torch.Tensor(random_uniform_rotation()).to(DEVICE)
    #
    #     v_orig = testLoader.dataset[i%20][0]
    #     true_y = testLoader.dataset[i%20][1].to(DEVICE)
    #     v = testLoader.dataset[i%20][0].to(DEVICE)
    #     faces = testLoader.dataset.f
    #     Z, _, _ = model(v)
    #     f = torch.nn.functional.log_softmax(Z, dim=1)
    #     pred_y = f.argmax()
    #
    #     v_rot = v
    #     v_rot = torch.mm(v, R).to(DEVICE)
    #     # v_rot = np.abs(np.random.normal()) * v
    #     v_rot = v_rot + torch.Tensor(np.random.normal(0, 0.01, size=(1, 3)).astype('f')).to(DEVICE)
    #     # theta = np.random.uniform(0, np.pi * 2)
    #     # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    #     # v = v.cpu().numpy()
    #     # v[:, [0, 2]] = v[:, [0, 2]].dot(rotation_matrix)  # random rotation
    #     # v = torch.from_numpy(v).to(DEVICE)
    #     Z_rot, _, _ = model(v_rot)
    #     f_rot = torch.nn.functional.log_softmax(Z_rot, dim=1)
    #     pred_y_rot = f_rot.argmax()
    #
    #     # plot_mesh_montage([v, v_rot], [faces, faces])
    #
    #
    #
    #     count += pred_y_rot==pred_y
    #
    # print("accuracy is :", count/float(number_of_tests))


