# **Handwritten Digits Recognition**
## **Objective**

In this project, we want to compare the performance of variaous classification algotithms on handwritten digits recognition using MNIST data. We measure performance based on not only accuracy but also training and recognition time. 


## **Data**
#### 1. Simplified dataset
We use R programming to show the mathematical principles of the models and run them on a simplied dataset.

The files digits.RData contain the same training and test data of handwritten digit images for R. There are 10 digits classes. Each digit image x belongs to {0,1}^20\*20 pixels in size and consists of binary values 0's and 1's only. There are 500 training images and 1000 test images per digit class. The whole training set is stored in a 10\*500\*20\*20 array and the whole test set is stored in a 10\*1000\*20\*20 array.

#### 2. MNIST dataset
There are 600,000 images in the training set and 100,000 images in the test set. Each image of digits is a 28*28 pixel square. The pixel values are gray scale between 0 and 255.   


## **Model**
#### Generative model 

(1) LDA (Linear Discriminant analysis) 

(2) BMM (Bernoulli Mixture) and GMM (Gaussian Mixture Model) 

#### Discriminative model

(3) SVM (Support vector machine)

(4) Logistic linear regression

#### Neural network

(5) Simple neural network 
 
(6) Simple convolutional neural network 

(7) Large convolutional neural network


## **Optimization and Evaluation**
#### Parameters
For all classifiers there will be a free parameter that needs to be adjusted using a holdout part of the training set. 

(1) LDA: smoothing parameter for covariance matrix 

(2) Mixture model: number of components for the mixture model 

(3) SVM and logistica linear regression: regularization parameter 

For these methods, we run the algorithm B times on a random subset of training data per class, and evaluate the error on the remaining data for a value range of the free parameters. We choose the values that minimize the average error rate over the B runs and check the resulting algorithms on the test set.

#### Cross-validatioan

We use cross-validation to avoid overfitting. 

#### BootStrap

We use bootstrap to compute the rejection probability on the test set. Rejection rate is more statistically meaningful than raw error or accuracy rate. 

