# handwritten-digits-recognition
Data set:
The files digits.RData contain the same training and test data of handwritten digit images for R. There are 10 digits classes. Each digit image x belongs to {0,1}^20\*20 pixels in size and consists of binary values 0's and 1's only. There are 500 training images and 1000 test images per digit class. The whole training set is stored in a 10\*500\*20\*20 array and the whole test set is stored in a 10\*1000\*20\*20 array.

Goal:
The goal of this project is to experiment with different classifiers for this data set. Two generative: (1) LDA (2) Mixture of products of Bernoulli's; Two discriminative (3) SVM: full solution and iterative solution (4) Logistic regression. For each classifier we will report the error rate on the test set. 

Parameter:
For all classifiers there will be a free parameter that needs to be adjusted using a held out part of the training set. (LDA: lambda-smoothing parameter for covariance matrix, Mixture: number of components M for the mixture model, SVM: the C paremeter and Logistic Regression: regularization parameter lambda). For these methods, run the algorithm five times on a random subset of 400 training examples per class, and evaluate the error on the remaining 100 examples per class, for a range of values of the free parameter, to obtain the value that minimize the average error rate over the five runs. Once we choose this value, check the resulting algorithm on the test set. 
