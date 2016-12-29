# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 15:30:24 2016

@author: lcao
"""

import classifier
import bmm
import gmm


# import data
import numpy as np
from keras.datasets import mnist
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
# transform data to binary for BMM 
train_data_binary = np.where(X_train > 0.5, 1, 0)
test_data_binary = np.where(X_test > 0.5, 1, 0)


# BMM model
# first fit bmm model on all classes
k = 10
model = bmm(k, n_iter=20, verbose=True)
model.fit(train_data_binary)

# fit training data using classifier for each class
c = 7
bayesian_classifier = classifier(c, means_init_heuristic='kmeans',
                                 means=None, model_type='bmm')
bayesian_classifier.fit(train_data_binary, y_train)
# predict test data
label_set = set(y_test)
predicted_labels = bayesian_classifier.predict(test_data_binary, label_set)


# GMM model
