# -*- coding: utf-8 -*-
import os.path
import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import pickle
import time

from sklearn.model_selection import train_test_split

import create_data_for_CNN
import CNN

'''-------------------------------------------------------------------------'''
'''------------------------------- function --------------------------------'''
'''-------------------------------------------------------------------------'''


def load_dataset(filename, save_dataset):
    """Get or create dataset from file.

    Args:
        filename: string
            Path of the file containing dataset or raw data.
        save_dataset: bool
            Whether dump a newly created dataset to file.
            Only useful when creating dataset from raw data.

    Returns:
        data_T: ndarray
            An array of shape 5891 * 10 * 400. Each 10 * 400 matrix represents a raw data file, with dtype of "float64"
            arranged in the order of "corrcoef * 8, temperature, humidity".
            The index of those 400-element arrays represents the group number of that data.
        Tag: ndarray
            A 5891-element array containing tags of data matrices of the same indices with dtype of "float64".
            0 means no mass and 1 means with mass.
            The first 5485 data matrices have tag 0, and the rest have 1.
        timestamp_T: ndarray
            An array of shape 5891 * 10 * 400. Each element is a datetime object,
            containing the time of measurement of data in the same position in data_T.
        n_tag_0: int
            Number of no mass data (5485).
        n_tag_1: int
            Number of mass data (406).
        scale_norm: dict
            The scale of normalization of each kind of data.
        data_type: ndarray
            An array of shape 5891 * 10 with data type of each 400-element array.
            Data type are represented by np.int32. 0 means correlation coefficient, 1 means temperature and 2 means humidity.

    Raises:
        None
    """

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
        
        data_T, Tag, timestamp_T, n_tag_0, n_tag_1, scale_norm, data_type = \
            create_data_for_CNN.create_dataset(file1, file2)
        
        dataset = {'data': data_T, 'tag': Tag, 'timestamp': timestamp_T,
                   'tag0': n_tag_0, 'tag1': n_tag_1, 'scale_norm': scale_norm, 'data_type': data_type}
        
        if save_dataset:
            with open(filename, 'wb') as handle:
                pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
    return data_T, Tag, timestamp_T, n_tag_0, n_tag_1, scale_norm, data_type


'''-------------------------------------------------------------------------'''
'''--------------------------- Hyper Parameters ----------------------------'''
'''-------------------------------------------------------------------------'''
EPOCH = 60
BATCH_SIZE = 128
LR = 0.0001     # learning rate
CLIP = 1
# baseline 100th measurement in Rawdata_data00025

BASELINE_FILE = 197
BASELINE_MEASUREMENT = 1

NETWORK_TYPE = "1D"

SAVE_CREATED_DATA = True
TRAIN = True
EVALUATE = False

pt_filename = 'pt/CNN_' + NETWORK_TYPE + '_predict_input.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''-------------------------------------------------------------------------'''
'''------------------------- Load Data -------------------------------------'''
'''-------------------------------------------------------------------------'''
print("loading data")

data_T, label_T, timestamp_T, n_tag_0, n_tag_1, scale_norm, data_type = \
    load_dataset('Data/plate_ultrasonic_dataset_197_process_predict_input_cnn.pickle', save_dataset=SAVE_CREATED_DATA)

train_input, validation_input, train_label, validation_label, data_type_train, data_type_test = \
    train_test_split(data_T, label_T, data_type, test_size=0.2)

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

if (NETWORK_TYPE == "2D"):
    cnn = CNN.CNNModel2D().to(device)
elif (NETWORK_TYPE == "1D"):
    cnn = CNN.CNNModel2D().to(device)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# validation loss of every epoch
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
        valid_loss, _, _, _ = CNN.evaluate(cnn, validation_loader, loss_func, device, epoch, NETWORK_TYPE)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = CNN.epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(cnn.state_dict(), pt_filename)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.5f}')
        print(f'\tValid Loss: {valid_loss:.5f}')

        loss_record.append(valid_loss)

    plt.plot(loss_record)
    plt.title("the change of validation loss in each epoch")
    plt.xlabel("epoch")
    plt.ylabel("validation loss")
    plt.savefig('Results/CNN_' + NETWORK_TYPE + '_validation_loss_change.png')

'''-------------------------------------------------------------------------'''
'''---------------------- evaluate the model -------------------------------'''
'''-------------------------------------------------------------------------'''
if EVALUATE:

    if os.path.isfile(pt_filename):
        cnn.load_state_dict(torch.load(pt_filename))

    start_time = time.time()
    
    valid_loss, accuracy, precision, recall = CNN.evaluate(cnn, validation_loader, loss_func, device, 59, NETWORK_TYPE)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = CNN.epoch_time(start_time, end_time)
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.5f}')
    print(f'\tValid Loss: {valid_loss:.5f}')
    print(f"\tAccuracy: {accuracy:.5f}%")
    print(f"\tPrecision: {precision:.5f}%")
    print(f"\tRecall: {recall:.5f}%")
