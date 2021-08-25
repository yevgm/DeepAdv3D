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
import wandb

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
from src.deep_adv_3d.train_loop import Trainer

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
    # set seed for all platforms
    set_determinsitic_run()

    if USE_WANDB:
        wandb.init(entity="deepadv3d", project="DeepAdv3D")

    model = PointNet(k=10)
    model = model.to(DEVICE)


    # Data Loading and pre-processing
    trainLoader, validationLoader, testLoader = load_datasets(dataset=DATASET_NAME, train_batch=TRAIN_BATCH_SIZE,
                                                              test_batch=TEST_BATCH_SIZE, val_batch=VAL_BATCH_SIZE)

    # classifier_trainer.train(train_data=trainLoader, val_data=validationLoader, test_data=testLoader, classifier=model)
    train_ins = Trainer(train_data=trainLoader, validation_data=validationLoader, test_data=testLoader,
                            model=model)

    train_ins.train()