# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 09:22:43 2016

@author: lcao
"""
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load data
(X_train,y_train), (X_test,y_test) = mnist.load_data()
# flatten 28*28 images to a 784 vector of each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0],num_pixels).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train/255
X_test = X_test/255

# one hot encode outputs fpr neural network model
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# transform data to binary for BMM
train_data_binary = np.where(X_train > 0.5, 1, 0)
test_data_binary = np.where(X_test > 0.5, 1, 0)


# data for simple convolutional neural network
(X_train,y_train), (X_test,y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]