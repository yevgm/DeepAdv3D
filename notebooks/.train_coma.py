import os
from collections import Counter

import torch
from torch_geometric.data import Dataset, DataLoader
import tqdm

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
    
    traindata_gtruth = [mesh.y for mesh in train_data]
    traindata_pos = [mesh.pos for mesh in train_data]

    # train module
    classifier.train()
    for epoch in range(epoch_number):
        # start epoch
        print("epoch "+str(epoch+1)+" of "+str(epoch_number))
        for i in tqdm.trange(len(train_data)):
            x = traindata_pos[i]
            y = traindata_gtruth[i]

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
    eval_data:DataLoader, 
    classifier:torch.nn.Module,
    epoch_number=1):

   
    classifier.eval()
#     evaldata_pos = [mesh.pos for mesh in eval_data]
#     evaldata_gtruth = [mesh.y.item() for mesh in eval_data]

    confusion = None
    for epoch in range(epoch_number):
        for data in tqdm.tqdm(eval_data):
#             x = eval_data[i].pos
            y = data.y.data.cpu().numpy()
#             x = x.to('cuda')
            data = data.to('cuda')

#             print('%d] '%i,x)
            out:torch.Tensor = classifier(data)
            if confusion is None:
                num_classes = out.shape[-1]
                confusion = torch.zeros([num_classes, num_classes])
            
            _, prediction = out.max(dim=-1)
            target = y
            confusion[target, prediction] +=1
            
            correct = torch.diag(confusion).sum()
            accuracy = correct/confusion.sum()
    return accuracy, confusion
