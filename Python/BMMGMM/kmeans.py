# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 16:03:32 2016

@author: lcao
"""

from sklearn.cluster import KMeans

def generate_kmeans(x, k, verbose=False):

    kmeans = KMeans(n_clusters=k, verbose=int(verbose)).fit(x).cluster_centers_

    return kmeans

