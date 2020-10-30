import os
from collections import Counter

import numpy as np
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import tqdm

import adversarial.pgd as pgd

def train(
    train_data:Dataset, 
    classifier:torch.nn.Module,
    parameters_file:str,
    epoch_number:int = 1,
    learning_rate:float=1e-3):
        
    # meters
    loss_values = []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=5e-4)

    # train module
    classifier.train()

    for epoch in range(epoch_number):
        # start epoch
        print("epoch "+str(epoch+1)+" of "+str(epoch_number))
        for i in tqdm.trange(len(train_data)):
            x = train_data[i].pos
            y = train_data[i].y

            optimizer.zero_grad()
            out = classifier(x)
            out = out.view(1,-1)

            loss = criterion(out, y)
            loss_values.append(loss.item())

            loss.backward()
            optimizer.step()

    # save the model's parameters
    torch.save(classifier.state_dict(), parameters_file)
    return loss_values

def evaluate(
    eval_data:Dataset, 
    classifier:torch.nn.Module,
    epoch_number=1):

    classifier.eval()
    confusion = None
    for epoch in range(epoch_number):
        for i in tqdm.trange(len(eval_data)):
            x = eval_data[i].pos
            y = eval_data[i].y

            out:torch.Tensor = classifier(x)
            
            if confusion is None:
                num_classes = out.shape[-1]
                confusion = torch.zeros([num_classes, num_classes])
            
            _, prediction = out.max(dim=-1)
            target = int(y)
            confusion[target, prediction] +=1
            
            correct = torch.diag(confusion).sum()
            accuracy = correct/confusion.sum()
    return accuracy, confusion


#------------------------------------------------------------------------------

def PGD_train(
    train_data:Dataset, 
    classifier:torch.nn.Module,
    parameters_file:str,
    epoch_number:int = 1,
    learning_rate:float=1e-3,
    steps=10,
    eps=0.001,
    alpha=0.032):
        
    # meters
    loss_values = []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=5e-4)

    # train module
    classifier.train()
    projection = lambda a,x: pgd.clip(a, pgd.lowband_filter(a,x))
    for epoch in range(epoch_number):
        # start epoch
        print("epoch "+str(epoch+1)+" of "+str(epoch_number))
        for i in tqdm.trange(len(train_data)):
            # create adversarial example
            builder = pgd.PGDBuilder().set_classifier(classifier)
            device = train_data[i].pos.device
            builder.set_mesh(
                train_data[i].pos,
                train_data[i].edge_index.t().to(device), 
                train_data[i].face.t().to(device))
            builder.set_iterations(steps).set_epsilon(eps).set_alpha(alpha).set_eigs_number(36)
            builder.set_projection(projection)
            x = builder.build().perturbed_pos
            y = train_data[i].y

            optimizer.zero_grad()
            out = classifier(x)
            out = out.view(1,-1)

            loss = criterion(out, y)
            loss_values.append(loss.item())

            loss.backward()
            optimizer.step()

    # save the model's parameters
    torch.save(classifier.state_dict(), parameters_file)
    return loss_values
