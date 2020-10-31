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
PARAMS_FILE = os.path.join(REPO_ROOT, "model_data/FAUST10_pointnet.pt")

# repository modules
sys.path.insert(0, SRC_DIR)
from utils.ios import write_off
import vista.adv_plotter
from vista.adv_plotter import show_perturbation, show_all_perturbations
import adversarial.output_handler as op

import models
# import train
import ntrain
import dataset
import utils
from models.pointnet import SimplePointNet
from models.Origin_pointnet import PointNetCls
from dataset.data_loaders import FaustDataset
import adversarial.carlini_wagner as cw
from adversarial.carlini_wagner import CWBuilder, LowbandPerturbation


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
    traindata = dataset.FaustDataset(FAUST, device=DEVICE, train=True, test=False, transform_data=False)
    testdata = dataset.FaustDataset(FAUST, device=DEVICE, train=False, test=True, transform_data=False)

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



def find_perturbed_shape(to_class, testdata, model, params, max_dim=None, **hyperParams):
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

    if isinstance(to_class, str) & (to_class == 'rand'):
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

    example_list = []
    for gt_class in np.arange(5, nclasses, 1):
        for adv_target in np.arange(5, nclasses, 1):
            # search for adversarial example
            if nclasses == 1:
                mesh = testdata[i]
                adv_target = target
            else:
                mesh = testdata[int(gt_class)]

            # perturb target toward adv_target
            adex = cw.generate_adversarial_example(
                mesh=mesh, classifier=model, target=int(adv_target),
                search_iterations=hyperParams['search_iterations'],
                lowband_perturbation=hyperParams['lowband_perturbation'],
                adversarial_loss=hyperParams['adversarial_loss'],
                similarity_loss=hyperParams['similarity_loss'],
                **params)
            example_list.append(adex)
    return example_list


if __name__ == "__main__":
    model = PointNetCls(k=10, feature_transform=False)
    model = model.to(DEVICE)
    # print(model)
    trainLoader,testLoader, traindata, testdata = load_datasets(train_batch=8, test_batch=20)

    # train network
    # loss_values, test_mean_loss, test_accuracy = nTrain.train(
    #                                                         train_data=trainLoader,
    #                                                         test_data=testLoader,
    #                                                         classifier=model,
    #                                                         batchSize=batchsize,
    #                                                         parameters_file=PARAMS_FILE,
    #                                                         epoch_number=50,
    #                                                         learning_rate=4e-3,
    #                                                         train=True)
    # temp train visualizer - in the future : add tensorboard?
    # print('test mean loss:',test_mean_loss,' test_accuracy:',test_accuracy)
    # loss_values = np.array(loss_values)
    # sliced_loss = loss_values[0::5]#sliced
    #
    # fig, axs = plt.subplots(2)
    # fig.suptitle('losses')
    # axs[0].plot(np.arange(1,len(sliced_loss)+1,1), sliced_loss)
    # axs[1].plot(np.arange(1,len(loss_values)+1,1), loss_values)
    #
    # axs[0].set(xlabel='5*batches index', ylabel='loss')
    # axs[0].grid()
    # axs[1].set(xlabel='batches index', ylabel='loss')
    # axs[1].grid()
    # plt.show()

    # load parameters
    model.load_state_dict(torch.load(PARAMS_FILE, map_location=DEVICE))
    model.eval()
    # show_model_accuracy(PARAMS_FILE, model)

    # ------------------------ hyper parameters ------------------------------
    # ------------------------------------------------------------------------
    CWparams = {
        CWBuilder.USETQDM: True,
        CWBuilder.MIN_IT: 500,
        CWBuilder.LEARN_RATE: 1e-4,
        CWBuilder.ADV_COEFF: 1,
        CWBuilder.REG_COEFF: 50,
        CWBuilder.K_nn: 140,# 140
        CWBuilder.NN_CUTOFF: 20, # 40
        LowbandPerturbation.EIGS_NUMBER: 40} # 10 is good
    hyperParams = {
            'search_iterations': 5,
            'lowband_perturbation' : True,
            'adversarial_loss' : "carlini_wagner",
            'similarity_loss' : "local_euclidean"}
    generate_examples = 1 # how many potential random examples to create in output folder
    # ------------------------------------------------------------------------

    now = datetime.now()
    d = now.strftime("%b-%d-%Y_%H-%M-%S")
    for example in np.arange(0, generate_examples, 1):
        print('------- example number '+str(example)+' --------')
        example_list = find_perturbed_shape('all', testdata, model, CWparams,
                                            **hyperParams, max_dim=10)
        op.save_results(example_list, CWparams=CWparams, hyperParams=hyperParams
                        , folder_name=d, file_name=str(example))


    if len(example_list) == 1:
        # show the original shape, the perturbed figure and both of them overlapped
        show_perturbation(example_list)
    else:
        # show only the perturbed shape
        show_all_perturbations(example_list)



