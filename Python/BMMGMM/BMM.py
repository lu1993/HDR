# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 08:19:49 2016

@author: lcao
"""
import numpy as np
EPS = np.finfo(float).eps

class bmm(mixture):

    def __init__(self, n_components, covariance_type='diag',
                 n_iter=100, verbose=False):

        super(bmm,self).__init__(n_components, covariance_type=covariance_type,
                         n_iter=n_iter, verbose=verbose)

    def _log_support(self, x):

        k = self.n_components 
        pi = self.weights
        mu = self.means

        x_c = 1 - x
        mu_c = 1 - mu

        log_support = np.ndarray(shape=(x.shape[0], k))

        for i in range(k):
            log_support[:, i] = (
                np.sum(x * np.log(mu[i, :].clip(min=1e-50)), 1) \
                + np.sum(x_c * np.log(mu_c[i, :].clip(min=1e-50)), 1))

        return log_support
        
# fit BMM
k = 10
model = bmm(k, n_iter=20, verbose=True)
model.fit(train_data_binary)
      
# fit and time classifier with raw data
import time        
c = 7
start = time.time()
bayesian_classifier = classifier(c, means_init_heuristic='kmeans',
                                 means=kmeans, model_type='bmm')
bayesian_classifier.fit(train_data_binary, y_train)
end = time.time()
print(end-start)

# predict test data
label_set = set(y_test)
start = time.time()
predicted_labels = bayesian_classifier.predict(test_data_binary, label_set)
end = time.time()
print(end-start)

#%load_ext memory_profiler
#%memit -r 1 bayesian_classifier.fit bayesian_classifier.fit(train_data_binary, y_train)
#%mprun -f bayesian_classifier.fit bayesian_classifier.fit(train_data_binary, y_train)
#%memit -r 1 bayesian_classifier.predict bayesian_classifier.predict(test_data_binary, label_set)
#%mprun -f bayesian_classifier.predict bayesian_classifier.predict(train_data_binary, y_train)

print('accuracy: {}'.format(np.mean(predicted_labels == y_test)))

# dimension reduction
from kmeans import generate_kmeans
kmeans = generate_kmeans(train_data_binary, 20)
import sklearn.decomposition
d = 40
reducer = sklearn.decomposition.PCA(n_components=d)
reducer.fit(train_data_binary)
train_data_reduced = reducer.transform(train_data_binary)
test_data_reduced = reducer.transform(test_data_binary)
kmeans_reduced = reducer.transform(kmeans)
# fit and time classifier with reduced data 
import time        
c = 7
start = time.time()
bayesian_classifier = classifier(c, means_init_heuristic='kmeans',
                                 means=kmeans_reduced, model_type='bmm')
bayesian_classifier.fit(train_data_reduced, y_train)
end = time.time()
print(end-start)

# predict test data
label_set = set(y_test)
start = time.time()
predicted_labels = bayesian_classifier.predict(test_data_reduced, label_set)
end = time.time()
print(end-start)
print('accuracy: {}'.format(np.mean(predicted_labels == y_test)))
