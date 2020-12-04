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
from utils.ios import write_off
import vista.adv_plotter
from vista.adv_plotter import show_perturbation, show_all_perturbations
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

# ----------------------------------------------------------------------------------------------------------------------#
#                                                   Functions
# ----------------------------------------------------------------------------------------------------------------------#

def load_datasets(train_batch=8,test_batch=20):
    train_dataset = FaustDataset(
        root=os.path.join(FAUST, r'raw'),
        classification=True,
        split='train')

    test_dataset = FaustDataset(
        root=os.path.join(FAUST, r'raw'),
        classification=True,
        split='test',
        data_augmentation=False)

    trainLoader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch,
                                               shuffle=True,
                                               num_workers=10)
    testLoader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=test_batch,
                                               shuffle=False,
                                               num_workers=10)
    # load data in different format for Adversarial code
    traindata = dataset.FaustDataset(FAUST, device=DEVICE, train=True, test=False, transform_data=True)
    testdata = dataset.FaustDataset(FAUST, device=DEVICE, train=False, test=True, transform_data=True)

    return trainLoader,testLoader,traindata,testdata

def show_model_accuracy(PARAMS_FILE,model):
    loss_values, test_mean_loss, test_accuracy = ntrain.train(
        train_data=trainLoader,
        test_data=testLoader,
        classifier=model,
        batchSize=20,
        parameters_file=PARAMS_FILE,
        learning_rate=1e-3,
        train=False)

    print('test mean loss:', test_mean_loss, ' test_accuracy:', test_accuracy)


def find_perturbed_shape(to_class, testdata, model, params, max_dim=None, animate=False, **hyperParams):
    '''
    to_class = 'rand'/'all' choose how many output shapes to find
    model = attacked classification torch model
    testdata = test data of "torch.util.data.Dataset" type
    params =                        # iterative algorithm hyper parameters
        {CWBuilder.USETQDM: True,
        CWBuilder.MIN_IT: 100,
        CWBuilder.LEARN_RATE: 1e-4,
        CWBuilder.ADV_COEFF: 1,
        CWBuilder.REG_COEFF: 15,
        LowbandPerturbation.EIGS_NUMBER: 40}
    '''
    example_list = []
    if isinstance(to_class, int):
        while True:
            i=to_class
            target = random.randint(0, testdata.num_classes - 1)
            ground = testdata[i].y.item()
            if ground != target: break
        nclasses = 1
    elif isinstance(to_class, str) & (to_class == 'rand'):
        # choose random target
        while True:
            i = random.randint(0, len(testdata) - 1)
            target = random.randint(0, testdata.num_classes - 1)
            ground = testdata[i].y.item()
            if ground != target: break
        nclasses = 1

    elif isinstance(to_class, str) & (to_class == 'all'):
        # class_arr = np.arange(0, testdata.num_classes, 1)
        # iterations = class_arr.tolist() * 10
        nclasses = testdata.num_classes
    else:
        assert False, 'Provided bad to_class argument'

    #Debug - reduce number of classes
    if (max_dim is not None) & (to_class == 'all'):
        nclasses = max_dim


    for gt_class in np.arange(0, nclasses, 1):
        for adv_target in np.arange(0, nclasses, 1):
            # search for adversarial example
            if nclasses == 1:
                mesh = testdata[i]
                adv_target = target
                testidx = i
            else:
                mesh = testdata[int(gt_class)]
                testidx = int(gt_class)

            # perturb target toward adv_target
            adex = cw.generate_adversarial_example(
                mesh=mesh, classifier=model, target=int(adv_target),
                search_iterations=hyperParams['search_iterations'],
                lowband_perturbation=hyperParams['lowband_perturbation'],
                adversarial_loss=hyperParams['adversarial_loss'],
                similarity_loss=hyperParams['similarity_loss'],
                animate=animate,
                **params)
            adex.target_testidx = int(testidx)
            example_list.append(adex)
    return example_list


if __name__ == "__main__":
    model = PointNetCls(k=10, feature_transform=False, global_transform=False)
    model = model.to(DEVICE)
    trainLoader, testLoader, traindata, testdata = load_datasets(train_batch=8, test_batch=20)

    # load parameters
    model.load_state_dict(torch.load(PARAMS_FILE, map_location=DEVICE))
    model.eval()

    # ------------------------ hyper parameters ------------------------------
    # ------------------------------------------------------------------------
    CWparams = {
        CWBuilder.USETQDM: True,
        CWBuilder.MIN_IT: 200,  # 200 is good
        CWBuilder.LEARN_RATE: 1e-4,
        CWBuilder.ADV_COEFF: 1,  # 1 is good for results, ~3 for animation
        CWBuilder.REG_COEFF: 2,
        CWBuilder.K_nn: 10,  # 140 is good
        CWBuilder.NN_CUTOFF: 40,  # 40 is good
        LowbandPerturbation.EIGS_NUMBER: 10}  # 10 is good, 40 in article

    hyperParams = {
            'search_iterations': 5,
            'lowband_perturbation' : True,
            'adversarial_loss' : "carlini_wagner",
            'similarity_loss' : "local_euclidean"}
    generate_examples = 1  # how many potential random examples to create in output folder
    compute_animation = False
    save_flag = True
    mode = 'rand'
    max_dim = 5  # matrix size
    # ------------------------------------------------------------------------

    now = datetime.now()
    d = now.strftime("%b-%d-%Y_%H-%M-%S")
    if mode == 'rand':
        for example in np.arange(0, generate_examples, 1):
            print('------- example number '+str(example)+' --------')
            example_list = find_perturbed_shape(mode, testdata, model, CWparams, animate=compute_animation,
                                                **hyperParams, max_dim=1)
            if save_flag:
                op.save_results(example_list, testdata, CWparams=CWparams, hyperParams=hyperParams
                                , folder_name=d, file_name=str(example))
    elif mode == 'all':
        print('------------- Computing Matrix --------------')
        example_list = find_perturbed_shape(mode, testdata, model, CWparams, animate=compute_animation,
                                            **hyperParams, max_dim=max_dim)
        if save_flag:
            op.save_results(example_list, testdata, CWparams=CWparams, hyperParams=hyperParams
                            , folder_name=d, file_name='matrix')
    elif mode == 'test':
        for example in np.arange(0, 20, 1):
            print('------- example number ' + str(example) + ' --------')
            example_list = find_perturbed_shape(int(example), testdata, model, CWparams, animate=compute_animation,
                                                **hyperParams, max_dim=1)
            if save_flag:
                op.save_results(example_list, testdata, CWparams=CWparams, hyperParams=hyperParams
                                , folder_name=d, file_name=str(example))


    if compute_animation:
        # vertices_list = []
        # for example in example_list:
        #     vertices_list.append(example.perturbed_pos)
        # animate(vertices_list, gif_name='gif0.gif')
        animate(example_list[0].animation_vertices, f=example_list[0].animation_faces[0], gif_name='gif10.gif')


    elif len(example_list) == 1:
        # show the original shape, the perturbed figure and both of them overlapped
        show_perturbation(example_list, testdata)
    else:
        # show only the perturbed shape
        show_all_perturbations(example_list)



