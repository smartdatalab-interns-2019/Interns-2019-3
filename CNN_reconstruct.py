# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:17:28 2019

@author: Administrator
"""
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from collections import OrderedDict
import plot_confusion_matrix as pcm
import matplotlib.pyplot as plt


class CNNModel(nn.Module):
    """A simple CNN-based autoencoder with 2D convolutional layer

    Attributes:
        __init__: Create encoder and decoder
        forward: Pass input to the network and get results from readout layer
    """
    def __init__(self):
        super(CNNModel, self).__init__()
        # encoder, use conv2d to process one 10 * 400 data matrix
        # output 32 * 5 * 200 data to decoder
        self.encoder = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)),
            ("relu1", nn.ReLU()),
            ("pool1", nn.MaxPool2d(kernel_size=2)),
            ("conv2", nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)),
            ("relu2", nn.ReLU()),
        ]))
        # decoder, just symmetric to encoder
        # reconstruct 10 * 400 data from output of encoder
        # output of one batch will be of shape (128, 10, 400) and the channel dimension is ignored since it's 1
        self.decoder = nn.Sequential(OrderedDict([
            ("deconv1", nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)),
            ("relu3", nn.ReLU()),
            ("upsample", nn.Upsample(scale_factor=2, mode="nearest")),
            ("deconv2", nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=5, stride=1, padding=2)),
            ("relu4", nn.ReLU()),
        ]))
    
    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out


# train CNN model in one epoch
def train(model, iterator, optimizer, criterion, clip, device):
    
    # set model to train mode
    model.train()

    # total loss of the epoch
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch[0].to(device)               # data of shape (128, 10, 400)
        src = torch.unsqueeze(src, dim=1)       # add channel dimension (becomes(128, 1, 10, 400))
        
        res = model(src.float()).view(-1, 10, 400)      # reshape result to match target shape
        src = torch.squeeze(src)
                 
        loss = criterion(res.float(), src.float())      # MSE loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()                                # apply gradients
        
        epoch_loss += loss.item()
        
        if i % 5 == 0:
            print("epoch loss:", epoch_loss)

    return epoch_loss / len(iterator)


def evaluate(model, iterator_train, iterator, criterion, device, epoch, network_type):
    
    # set model to evaluation mode
    model.eval()
    
    # total loss of the epoch
    epoch_loss = 0

    # threshold of two kinds of data
    threshold = 50
    
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    loss_total = np.empty((0, 10))
    predicted_total = np.empty(0)
    target_total = np.empty(0)

    # calculate baseline for test data regularization
    with torch.no_grad():
        for i, batch in enumerate(iterator_train):
            src = batch[0].to(device)               # data of shape (128, 10, 400)
            src = torch.unsqueeze(src, dim=1)       # add channel dimension (becomes(128, 1, 10, 400))
            trg = batch[1].to(device)               # label of shape (128, 10, 400)
            
            res = model(src.float()).view(-1, 10, 400)      # reshape result to match target shape
            src = torch.squeeze(src)
            
            # record property-wise loss on training data in one batch
            # the property wise loss of one training data is 400 loss value summed
            loss_train = torch.abs(src.float() - res).to(device)
            loss_train = torch.sum(loss_train, 2)
            loss_total = np.concatenate((loss_total, loss_train.cpu().numpy()))

        loss_total = loss_total.T
        # now loss_total is of shape (10, train size)
        # calculate property-wise loss mean and standard deviation matrix
        loss_mean = np.mean(loss_total, 1)
        loss_std = np.cov(loss_total)

        for i, batch in enumerate(iterator):
            src = batch[0].to(device)
            src = torch.unsqueeze(src, dim=1)
            trg = batch[1].long().to(device)

            res = model(src.float()).view(-1, 10, 400)   # reshape to match target shape
            src = torch.squeeze(src)

            # property-wise loss on test data in one batch
            loss_test = torch.abs(src.float() - res).cpu().numpy()
            loss_test = np.sum(loss_test, 2)

            # normalize loss and predict
            # take no mass as positive for convenience
            loss_final = np.ndarray(res.size()[0])
            loss_test -= loss_mean
            for j in range(res.size()[0]):
                loss_final[j] = loss_test[j].dot(np.linalg.inv(loss_std)).dot(loss_test[j].T)
            
            predicted = torch.ones(res.size()[0], dtype=torch.float).long().to(device)
            for k in range(res.size()[0]):
                if (np.abs(loss_final[k]) >= threshold):
                    predicted[k] = 0

            tp += ((predicted == 1) & (trg == 1)).sum().item()
            tn += ((predicted == 0) & (trg == 0)).sum().item()
            fn += ((predicted == 0) & (trg == 1)).sum().item()
            fp += ((predicted == 1) & (trg == 0)).sum().item()
            predicted_total = np.concatenate([predicted_total, predicted.cpu().numpy()])
            target_total = np.concatenate([target_total, trg.cpu().numpy()])

    precision = 100 * tp / max((tp + fp), 1)
    recall = 100 * tp / max((tp + fn), 1)
    accuracy = 100 * (tp + tn) / (tp + fp + tn + fn)
    f1 = 2 * precision * recall / max((precision + recall), 1)
    
    predicted_total = predicted_total.astype("int32")
    target_total = target_total.astype("int32")

    # get result of the first and last epoch
    if (epoch == 0 or epoch == 59):
        pcm.plot_confusion_matrix(target_total, predicted_total, np.array(["mass", "no mass"]))
        plt.savefig('Results/CNN_' + network_type + '_epoch' + str(epoch + 1) + '_result.png')
        pcm.plot_confusion_matrix(target_total, predicted_total, np.array(["mass", "no mass"]), normalize=True)
        plt.savefig('Results/CNN_' + network_type + '_epoch' + str(epoch + 1) + '_normalized_result.png')
        plt.clf()

    return epoch_loss / len(iterator), accuracy, precision, recall, f1


def evaluate_complete(model, iterator, criterion, device, epoch, network_type, timestamps):
    
    # set model to evaluation mode
    model.eval()
    
    # total loss of the epoch
    epoch_loss = 0

    # threshold of two kinds of data
    threshold = torch.tensor(0.01).to(device)
    
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    tp_time = []
    tn_time = []
    fp_time = []
    fn_time = []
    timestamps = timestamps[:, 0, 0]
    predicted_total = np.array([])
    target_total = np.array([])

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0].to(device)
            src = torch.unsqueeze(src, dim=1)
            trg = batch[1].long().to(device)

            res = model(src.float()).view(-1)   # reshape from (128, 1) to (128) to match target shape
            print(res)

            # average loss of a batch
            loss = criterion(res.float(), trg.float())
            epoch_loss += loss.item()

            # take no mass as positive for convenience
            predicted = torch.ones(res.size()[0]).long().to(device)
            for j in range(res.size()[0]):
                if (torch.abs(res[j] - torch.tensor(1).to(device)) >= threshold):
                    predicted[j] = 0

            tp += ((predicted == 1) & (trg == 1)).sum().item()
            tn += ((predicted == 0) & (trg == 0)).sum().item()
            fn += ((predicted == 0) & (trg == 1)).sum().item()
            fp += ((predicted == 1) & (trg == 0)).sum().item()
            predicted_total = np.concatenate([predicted_total, predicted.cpu().numpy()])
            target_total = np.concatenate([target_total, trg.cpu().numpy()])

    precision = 100 * tp / (tp + fp)
    recall = 100 * tp / (tp + fn)
    accuracy = 100 * (tp + tn) / (tp + fp + tn + fn)
    
    predicted_total = predicted_total.astype("int32")
    target_total = target_total.astype("int32")

    for i in range(len(timestamps)):
        if (predicted_total[i] == target_total[i]):
            if (predicted_total[i] == 1):
                tp_time.append(timestamps[i])
            else:
                tn_time.append(timestamps[i])
        elif (predicted_total[i] > target_total[i]):
            fp_time.append(timestamps[i])
        else:
            fn_time.append(timestamps[i])

    # get result of the first and last epoch
    if (epoch == 0 or epoch == 59):
        plt.subplot(211)
        plt.title("targets")
        plt.plot(timestamps, target_total, 'bo')
        plt.subplot(212)
        plt.title("prediction")
        plt.plot(tp_time, np.ones(len(tp_time)), 'ro', label="tp")
        plt.plot(tn_time, np.zeros(len(tn_time)), 'yo', label="tn")
        plt.plot(fp_time, np.ones(len(fp_time)), 'go', label="fp")
        plt.plot(fn_time, np.zeros(len(fn_time)), 'co', label="fn")
        plt.legend()
        plt.savefig("Results/CNN_" + network_type + "_epoch" + str(epoch + 1) + "_prediction_target.png")
        pcm.plot_confusion_matrix(target_total, predicted_total, np.array(["mass", "no mass"]))
        plt.savefig('Results/CNN_' + network_type + '_epoch' + str(epoch + 1) + '_result.png')
        pcm.plot_confusion_matrix(target_total, predicted_total, np.array(["mass", "no mass"]), normalize=True)
        plt.savefig('Results/CNN_' + network_type + '_epoch' + str(epoch + 1) + '_normalized_result.png')
        plt.clf()

    return epoch_loss / len(iterator), accuracy, precision, recall


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
