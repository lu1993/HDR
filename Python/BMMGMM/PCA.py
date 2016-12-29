# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 10:25:17 2016

@author: lcao
"""

# dimension reduction
from kmeans import generate_kmeans
kmeans = generate_kmeans(X_train, 20)
import sklearn.decomposition
d = 40
reducer = sklearn.decomposition.PCA(n_components=d)
reducer.fit(X_train)
train_data_reduced = reducer.transform(X_train)
test_data_reduced = reducer.transform(X_test)
kmeans_reduced = reducer.transform(kmeans)