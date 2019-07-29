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
        label_T = dataset['label']
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
                
    return data_T, label_T, Tag, timestamp_T, n_tag_0, n_tag_1, scale_norm, data_type


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

pt_filename = 'pt/cnn_' + DATA_MODE + '_1.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''-------------------------------------------------------------------------'''
'''------------------------- Load Data -------------------------------------'''
'''-------------------------------------------------------------------------'''
if Loading_DATA:

    filename_preprocess_data = 'Data/plate_ultrasonic_dataset_197_process_' + DATA_MODE + '.pickle'
    
    data_T, label_T, Tag, timestamp_T, n_tag_0, n_tag_1, scale_norm, data_type = load_dataset(filename=filename_preprocess_data, if_save_dataset=SAVE_CREATED_DATA)
    # special process for cnn
    Tag = np.concatenate((Tag, np.zeros(n_tag_0), np.ones(n_tag_1)), axis = 0)
    Tag = np.concatenate((Tag, np.zeros(n_tag_0), np.ones(n_tag_1)), axis = 0)

    # decide whether data with mass need to be rid
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
cnn = CNN.CNNModel().to(device)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

loss_record = []
correlationCoefficientList = []
correlationCoefficientList_eva = []

'''-------------------------------------------------------------------------'''
'''---------------------------- train model --------------------------------'''
'''-------------------------------------------------------------------------'''
if TRAIN:

    cnn.load_state_dict(torch.load(pt_filename))
    best_valid_loss = float('inf')
       
    for epoch in range(EPOCH):
        start_time = time.time()
        
        train_loss, correlationCoefficientList = CNN.train(cnn, train_loader, optimizer, loss_func, CLIP, device, correlationCoefficientList)
        valid_loss, correlationCoefficientList_eva = CNN.evaluate(cnn, validation_loader, loss_func, device, correlationCoefficientList_eva)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = CNN.epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(cnn.state_dict(), pt_filename)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    
        loss_record.append([train_loss, valid_loss])
