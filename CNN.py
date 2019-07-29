# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:17:28 2019

@author: Administrator
"""
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import seaborn as sns
sns.set_style("whitegrid")
from collections import OrderedDict


def correlationCoeff(label, output):
    
    N, _ = np.shape(label)
 
    corrcoefficient = []

    for i in range(N):
               
        corrcoefficient.append(np.corrcoef(label[i, :], output[i, :])[0][1])

    return np.array(corrcoefficient)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Hidden layers, use conv1d to process time series
        self.hidden1 = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)),
            ("relu1", nn.ReLU()),
            ("pool1", nn.MaxPool1d(kernel_size=2))
        ]))
        self.hidden2 = nn.Sequential(OrderedDict([
            ("conv2", nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)),
            ("relu2", nn.ReLU()),
            ("pool2", nn.MaxPool1d(kernel_size=2))
        ]))
        # Fully connected layer (readout)
        self.fc = nn.Linear(32 * 200, 10)
    
    def forward(self, x):
        out = self.hidden1(x)
        out = self.hidden2(out)
        # Resize
        # Original size: (128, 32, 200)
        # out.size(0): 128
        # New out size: (128, 32*200)
        out = out.view(out.size(0), -1)
        # Linear function (readout)
        out = self.fc(out)
        return out


# train CNN model in one epoch
def train(model, iterator, optimizer, criterion, clip, device, correlationCoefficientList):
    
    # set model to train mode
    model.train()

    # total loss of the epoch
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch[0].to(device)   # data
        trg = batch[1].to(device)   # label
        
        res = model(src.float())
                 
        loss = criterion(res.float(), trg.float())      # mean square error
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()                                # apply gradients
        
        epoch_loss += loss.item()
        
        if i % 20 == 0:
            print("epoch loss:", epoch_loss)

    corrcoefficient = correlationCoeff(trg.to('cpu').detach().numpy(), res.to('cpu').detach().numpy())
    correlationCoefficientList.append(corrcoefficient[0])
    print(corrcoefficient)
          
    return epoch_loss / len(iterator), correlationCoefficientList


def evaluate(model, iterator, criterion, device, correlationCoefficientList_eva):
    
    # set model to evaluation mode
    model.eval()
    
    # total loss of the epoch
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):
        
            src = batch[0].to(device)
            trg = batch[1].to(device)

            res = model(src.float())

            loss = criterion(res.float(), trg.float())
            epoch_loss += loss.item()
            corrcoefficient = correlationCoeff(trg.to('cpu').detach().numpy(), res.to('cpu').detach().numpy())
            correlationCoefficientList_eva.append(corrcoefficient[0])
            
        print(corrcoefficient)
        
    return epoch_loss / len(iterator), correlationCoefficientList_eva


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
