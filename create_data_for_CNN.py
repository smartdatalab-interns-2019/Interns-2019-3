# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:03:01 2019

@author: Administrator
"""

import numpy as np
import seaborn as sns
sns.set_style("whitegrid")
import pickle
from datetime import datetime
import time


def normalize_data(data_correlation_coeff, data_temperature, data_humidity):
    """normalize coefficients, temperature and humidity

    constrain data to a certain range if necessary,
    then normalize by (x - min) / (max - min) and add a dictionary
    """
    max_value_correlation_coeff = np.max(data_correlation_coeff)
    min_value_correlation_coeff = np.min(data_correlation_coeff)
    data_correlation_coeff = (data_correlation_coeff - min_value_correlation_coeff) / (max_value_correlation_coeff - min_value_correlation_coeff)
    
    data_temperature[np.where(data_temperature < -3)] = -3
    data_temperature[np.where(data_temperature > 50)] = 50
    max_value_temperature = np.max(data_temperature)
    min_value_temperature = np.min(data_temperature)
    data_temperature = (data_temperature - min_value_temperature) / (max_value_temperature - min_value_temperature)
    
    data_humidity[np.where(data_humidity > 100)] = 100
    max_value_humidity = np.max(data_humidity)
    min_value_humidity = np.min(data_humidity)
    data_humidity = (data_humidity - min_value_humidity) / (max_value_humidity - min_value_humidity)
    
    scale_norm = {'correlation_coeff_max': max_value_correlation_coeff,
                  'correlation_coeff_min': min_value_correlation_coeff,
                  'temperature_max': max_value_temperature,
                  'temperature_min': min_value_temperature,
                  'humidity_max': max_value_humidity,
                  'humidity_min': min_value_humidity}

    return data_correlation_coeff, data_temperature, data_humidity, scale_norm


def generate_timestamp(data_day, data_time):
    print("\nstart to generate time stamp\n")
    
    timestamp_T = []
    
    for i in range(np.shape(data_day)[0]):
        temp_timestamp_T = []
        for j in range(np.shape(data_time)[1]):
            try:
                temp_timestamp_T.append(datetime.strptime(data_day[i][j] + ' ' + data_time[i][j].split(".")[0], '%Y/%m/%d %H:%M:%S'))
            except ValueError:
                temp_timestamp = time.strptime(data_day[i][j] + data_time[i][j], '%Y/%m/%d%H:%M:%S.%f')
                temp_timestamp = time.mktime(temp_timestamp) + 0.0001
                temp_timestamp_T.append(datetime.fromtimestamp(temp_timestamp))
                
        timestamp_T.append(temp_timestamp_T)
        if i % 100 == 0:
            print(i, " files has generated time stamps")
    
    timestamp_T = np.array(timestamp_T)

    print("\ncomplete generating time stamp\n")

    return timestamp_T


def read_pickle_plate_dataset(filename_pickle):
    """Read raw data and return five members
    """
    with open(filename_pickle, 'rb') as file:
        plate_ultrasonic_dataset = pickle.load(file)
        
    print("complete reading pickle ultrasonic dataset: ", filename_pickle)
    print(plate_ultrasonic_dataset.keys())
        
    dataset_original = plate_ultrasonic_dataset['correlation_coefficient']
    data_temperature_original = plate_ultrasonic_dataset['temperature']
    data_humidity_original = plate_ultrasonic_dataset['humidity']
    data_day_original = plate_ultrasonic_dataset['day']
    data_time_original = plate_ultrasonic_dataset['time']
    
    return dataset_original, data_temperature_original, data_humidity_original,\
        data_day_original, data_time_original


def padding_dataset(data_correlation_coeff, data_temperature, data_humidity, data_day, data_time,
                    dataset_original, data_temperature_original, data_humidity_original,
                    data_day_original, data_time_original, padding_size=400, startpoint=1):
    
    for i in range(startpoint, len(dataset_original)):
        
        tempdata = dataset_original[i].T
        pad_len = padding_size - np.shape(tempdata)[1]
        data_correlation_coeff = np.concatenate((data_correlation_coeff, np.pad(tempdata, ((0, 0), (0, pad_len)), 'edge')), axis=0)
        data_temperature = np.concatenate((data_temperature, np.pad(data_temperature_original[i][np.newaxis, :], ((0, 0), (0, pad_len)), 'edge')), axis=0)
        data_humidity = np.concatenate((data_humidity, np.pad(data_humidity_original[i][np.newaxis, :], ((0, 0), (0, pad_len)), 'edge')), axis=0)
        data_day = np.concatenate((data_day, np.pad(data_day_original[i][np.newaxis, :], ((0, 0), (0, pad_len)), 'edge')), axis=0)
        data_time = np.concatenate((data_time, np.pad(data_time_original[i][np.newaxis, :], ((0, 0), (0, pad_len)), 'edge')), axis=0)
        
        if i % 200 == 0:
            print(f'\t{i} files have been loaded')
    
    return data_correlation_coeff, data_temperature, data_humidity, data_day, data_time


def create_dataset(file1, file2):
    """Process raw data and create a new dataset for models to use.

    Args:
        DATA_MODE: the way of organizing data
        file1: path of file consisting of data without mass
        file2: path of file consisting of data with mass
        N_FILE_TYPE: 2 means data with mass in file2

    Returns:
        A tuple containing following elements:
        data_T, label_T, Tag, timestamp_T, tag_0, tag_1, scale_norm, data_type

    Raises:
        None
    """
    # process file1
    dataset_original, data_temperature_original, data_humidity_original, \
        data_day_original, data_time_original = read_pickle_plate_dataset(file1)
     
    data_correlation_coeff = dataset_original[0].T
    data_temperature = np.expand_dims(data_temperature_original[0], axis=0)
    data_humidity = np.expand_dims(data_humidity_original[0], axis=0)
    data_day = np.expand_dims(data_day_original[0], axis=0)
    data_time = np.expand_dims(data_time_original[0], axis=0)
    
    tag_0 = np.shape(dataset_original)[0]
    
    data_correlation_coeff, data_temperature, data_humidity, data_day, data_time = \
        padding_dataset(data_correlation_coeff, data_temperature, data_humidity, data_day, data_time,
                        dataset_original, data_temperature_original, data_humidity_original,
                        data_day_original, data_time_original, padding_size=400, startpoint=1)
    
    data_correlation_coeff[33749, 294] = -0.0913444744020018
    
    # process file2

    dataset_original, data_temperature_original, data_humidity_original, \
        data_day_original, data_time_original = read_pickle_plate_dataset(file2)
    
    tag_1 = np.shape(dataset_original)[0]
    
    data_correlation_coeff, data_temperature, data_humidity, data_day, data_time = \
        padding_dataset(data_correlation_coeff, data_temperature, data_humidity, data_day, data_time,
                        dataset_original, data_temperature_original, data_humidity_original,
                        data_day_original, data_time_original, padding_size=400, startpoint=0)
        
    data_correlation_coeff[33237, 294] = -0.0913444744020018
    
    # generate formatted dataset
    data_correlation_coeff, data_temperature, data_humidity, scale_norm = \
        normalize_data(data_correlation_coeff, data_temperature, data_humidity)
 
    data_T = np.concatenate((data_correlation_coeff, data_temperature, data_humidity), axis=0)
    data_type = np.concatenate((np.zeros(np.shape(data_correlation_coeff)[0]).astype('int'),
                                np.ones(np.shape(data_temperature)[0]).astype('int'),
                                2 * np.ones(np.shape(data_humidity)[0]).astype('int')), axis=0)
    
    if tag_1 != 0:
        Tag = np.concatenate((np.zeros(tag_0 * 8), np.ones(tag_1 * 8)), axis=0)
    else:
        Tag = np.zeros(tag_0 * 8)
        
    timestamp_T = generate_timestamp(data_day, data_time)
    timestamp_T_10 = np.repeat(timestamp_T, 8, axis=0)
    timestamp_T_10 = np.concatenate((timestamp_T_10, timestamp_T), axis=0)
    timestamp_T_10 = np.concatenate((timestamp_T_10, timestamp_T), axis=0)
    timestamp_T = timestamp_T_10

    print("the shape of dataset is", np.shape(data_T))

    data_new = np.empty((5891, 10, 400), dtype=np.float64)
    tag_new = np.empty(5891, dtype=np.float64)
    timestamp_new = np.empty((5891, 10, 400), dtype=datetime)
    data_type_new = np.empty((5891, 10), dtype=np.int32)
    n_tag = tag_0 + tag_1

    for i in range(0, n_tag):
        for j in range(8):
            data_new[i][j] = data_T[i * 8 + j]
            timestamp_new[i][j] = timestamp_T[i * 8 + j]
            data_type_new[i][j] = data_type[i * 8 + j]

        data_new[i][8] = data_T[n_tag * 8 + i]
        timestamp_new[i][8] = timestamp_T[n_tag * 8 + i]
        data_type_new[i][8] = data_type[n_tag * 8 + i]

        data_new[i][9] = data_T[n_tag * 9 + i]
        timestamp_new[i][9] = timestamp_T[n_tag * 9 + i]
        data_type_new[i][9] = data_type[n_tag * 9 + i]

        tag_new[i] = Tag[i * 8]

    dataset_new = {'data': data_new, 'tag': tag_new, 'timestamp': timestamp_new,
                   'tag0': tag_0, 'tag1': tag_1, 'scale_norm': scale_norm, 'data_type': data_type_new}
    
    with open('Data/plate_ultrasonic_dataset_197_process_predict_input_cnn.pickle', 'wb') as handle:
        pickle.dump(dataset_new, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return data_T, Tag, timestamp_T, tag_0, tag_1, scale_norm, data_type
