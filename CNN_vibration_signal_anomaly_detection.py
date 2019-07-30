# -*- coding: utf-8 -*-
import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import pickle
import time
import random
import math
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import pairwise_distances_argmin
import sklearn.metrics as metrics

import create_data_for_deeplearning
import CNN
import detection_algorithm
import plot_confusion_matrix as pcm

'''-------------------------------------------------------------------------'''
'''------------------------------- function --------------------------------'''
'''-------------------------------------------------------------------------'''


def load_dataset(filename, if_save_dataset):

    if os.path.isfile(filename):
        
        with open(filename, 'rb') as handle:
            dataset = pickle.load(handle)
            
        data_T = dataset['data']
        Tag = dataset['tag']
        timestamp_T = dataset['timestamp']
        n_tag_0 = dataset['tag0']
        n_tag_1 = dataset['tag1']
        scale_norm = dataset['scale_norm']
        data_type = dataset['data_type']
        
    else:
        print("*" * 50)
        print("start to create dataset")
        print("*" * 50)
        print("\n")
        
        file1 = 'Data/plate_ultrasonic_dataset_197_no_mass.pickle'
        file2 = 'Data/plate_ultrasonic_dataset_197_damage.pickle'
        
        data_T, label_T, Tag, timestamp_T, n_tag_0, n_tag_1, scale_norm, data_type = \
            create_data_for_deeplearning.create_dataset(DATA_MODE, file1, file2, N_FILE)
        
        dataset = {'data': data_T, 'label': label_T, 'tag': Tag, 'timestamp': timestamp_T,
                   'tag0': n_tag_0, 'tag1': n_tag_1, 'scale_norm': scale_norm, 'data_type': data_type}
        
        if if_save_dataset:
            with open(filename, 'wb') as handle:
                pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
    return data_T, Tag, timestamp_T, n_tag_0, n_tag_1, scale_norm, data_type


def create_no_mass_index(data_mode, n_tag_0, n_tag_1):
    
    if data_mode == 'predict_input':
        no_mass_index = np.concatenate((np.arange(n_tag_0 * 8),
                                        np.arange(n_tag_0 * 8 + n_tag_1 * 8, n_tag_0 * 9 + n_tag_1 * 8),
                                        np.arange(n_tag_0 * 9 + n_tag_1 * 9, n_tag_0 * 10 + n_tag_1 * 9)), axis=0)
    
    if data_mode == 'predict_temperature':
        no_mass_index = np.arange(n_tag_0 * 8)
        
    if data_mode == 'predict_humidity':
        no_mass_index = np.arange(n_tag_0 * 8)

    return no_mass_index


'''-------------------------------------------------------------------------'''
'''--------------------------- Hyper Parameters ----------------------------'''
'''-------------------------------------------------------------------------'''
EPOCH = 60
BATCH_SIZE = 128
LR = 0.0001     # learning rate
CLIP = 1
# baseline 100th measurement in Rawdata_data00025
DATA_MODE = 'predict_input'
N_FILE = 2

BASELINE_FILE = 197
BASELINE_MEASUREMENT = 1

Loading_DATA = True
SAVE_CREATED_DATA = True
WITH_MASS_LABEL = True

ANALYSIS_DATA = True
TRAIN = True
EVALUATE = False
COMPRRSSION_DATA = False
DETECTION_ANOMALY = True

DATA_TYPE = ['correlation coefficient', 'temperature', 'humidity']

pt_filename = 'pt/cnn_2d_' + DATA_MODE + '.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''-------------------------------------------------------------------------'''
'''------------------------- Load Data -------------------------------------'''
'''-------------------------------------------------------------------------'''
print("loading data")

if Loading_DATA:

    filename_preprocess_data = 'Data/plate_ultrasonic_dataset_197_process_' + DATA_MODE + '_cnn.pickle'
    
    data_T, label_T, timestamp_T, n_tag_0, n_tag_1, scale_norm, data_type = \
        load_dataset(filename=filename_preprocess_data, if_save_dataset=SAVE_CREATED_DATA)

    # decide whether data with mass need to be rid of
    if WITH_MASS_LABEL:
    
        train_input, validation_input, train_label, validation_label, data_type_train, data_type_test = \
            train_test_split(data_T, label_T, data_type, test_size=0.2)
        
    else:
        no_mass_index = create_no_mass_index(data_mode=DATA_MODE, n_tag_0=n_tag_0, n_tag_1=n_tag_1)
        train_input, validation_input, train_label, validation_label, data_type_train, data_type_test = \
            train_test_split(data_T[no_mass_index, :], label_T[no_mass_index, :], data_type[no_mass_index], test_size=0.2)
    
    train_input = torch.from_numpy(train_input)
    train_label = torch.from_numpy(train_label)
    train_data = torch.utils.data.TensorDataset(train_input, train_label)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        
    validation_input = torch.from_numpy(validation_input)
    validation_label = torch.from_numpy(validation_label)
    validation_data = torch.utils.data.TensorDataset(validation_input, validation_label)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True)

'''-------------------------------------------------------------------------'''
'''------------------------------ create model -----------------------------'''
'''-------------------------------------------------------------------------'''
print("creating model")

cnn = CNN.CNNModel().to(device)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

loss_record = []

'''-------------------------------------------------------------------------'''
'''---------------------------- train model --------------------------------'''
'''-------------------------------------------------------------------------'''
print("training model")

if TRAIN:

    if os.path.isfile(pt_filename):
        cnn.load_state_dict(torch.load(pt_filename))
    best_valid_loss = float('inf')
       
    for epoch in range(EPOCH):
        start_time = time.time()
        
        train_loss = CNN.train(cnn, train_loader, optimizer, loss_func, CLIP, device)
        valid_loss, accuracy = CNN.evaluate(cnn, validation_loader, loss_func, device)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = CNN.epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(cnn.state_dict(), pt_filename)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.5f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.5f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        print(f"\tAccuracy: {accuracy:.5f}%")
    
        loss_record.append([train_loss, valid_loss])

'''-------------------------------------------------------------------------'''
'''---------------------- evaluate the model -------------------------------'''
'''-------------------------------------------------------------------------'''
if EVALUATE:

    cnn.load_state_dict(torch.load(pt_filename))
    
    evaluate_data_index = np.random.randint(validation_input.shape[0], size = 64)
    
    evaluation_data = validation_input[evaluate_data_index]
    encoded_data_eva, decoded_data_eva = autoencoder(evaluation_data.to(device).float())
    
    if DATA_MODE == 'predict_input':
        
        data1 = data_type_test[evaluate_data_index]
        
    elif DATA_MODE == 'predict_temperature':
        
        data1 = np.ones(64).astype('int')
        
    elif DATA_MODE == 'predict_humidity':
        
        data1 = 2 * np.ones(64).astype('int')
        
    recovered_validation_data = denormalized_data(validation_label[evaluate_data_index].numpy(), \
                                                  scale_norm, data_type1 = data1)
    recovered_validation_data_decoded = denormalized_data(decoded_data_eva.data.to('cpu').detach().numpy(), \
                                                          scale_norm, data_type1 = data1)
'''
    for i in range(64):
    
        plt.ion()
        
        fig = plt.figure(figsize = (10,8))
        ax = fig.add_subplot(211)
        #ax.plot(timestamp_T[1,:], validation_label[i].numpy())
        ax.plot(timestamp_T[i,:], recovered_validation_data[i], label = "true curve")
        ax.set_ylabel(DATA_TYPE[data_type_test[i]], fontsize = 15)
        ax.set_xlabel("time", fontsize = 15)        
        ax.set_title("the change of "+ DATA_TYPE[data_type_test[i]] + " in one measurement file", fontsize = 15)
        ax.legend(loc = "upper right")
        ax = fig.add_subplot(212)
        #ax.plot(timestamp_T[1,:], decoded_data_eva.data.to('cpu').detach().numpy()[i])
        ax.plot(timestamp_T[i,:], recovered_validation_data_decoded[i], label = "predicted curve")
        ax.set_title("the change of predicted "+ DATA_TYPE[data_type_test[i]] + " in one measurement file", fontsize = 15)
        ax.set_ylabel(DATA_TYPE[data_type_test[i]], fontsize = 15)
        ax.set_xlabel("time", fontsize = 15)
        ax.legend(loc = "upper right")
        plt.subplots_adjust(wspace = 0.1, hspace = 0.25)

        plt.pause(2)
        # plt.savefig('D:/Research/DeepLearning/Results/autoencoder/predict_temperature' + str(i) +'.png')
        plt.close()
      
    plt.figure(2)
    plt.plot(loss_record)
    plt.title("the change of loss in each epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
    
    plt.figure(3)
    plt.plot(correlationCoefficientList_eva)
    plt.title("correlation coefficient between input and output in one bach")
    plt.xlabel("measurement")
    plt.ylabel("correlation coefficient")
    plt.show()
'''