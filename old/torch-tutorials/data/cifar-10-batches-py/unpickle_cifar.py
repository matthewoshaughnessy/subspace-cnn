#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 12:19:24 2018

@author: Rakshith
"""
import numpy as np
import math
import pickle
from copy import deepcopy

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

test_batch = unpickle('test_batch')
test_data = test_batch[b'data']

N = test_data.shape[0]
D = test_data.shape[1]

# Add noise with certain SNR
SNRdB = 25
test_data_noisy = np.zeros((N,D))

for ii in range(N):
    img = test_data[ii,:]
    norm_img = np.linalg.norm(test_data[ii,:])
    norm_noise = norm_img*10**(-SNRdB/20) 
    noise = np.random.normal(0,math.sqrt(norm_noise**2/D),(D))
    test_data_noisy[ii,:] = img+noise

test_batch_noisy25dB = deepcopy(test_batch)
pickle.dump(test_batch_noisy25dB,open("test_batch_25dB","wb"))


