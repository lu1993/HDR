# **Handwritten Digits Recognition**
## **Objective**

In this project, we want to compare the performance of variaous classification algotithms in handwritten digits recognition based on not only accuracy but also training and recognition time. 


## **Data**
#### 1. 'toy' dataset

We reduce the MNIST dateset to a 'toy' dataset and use R programming to show the mathematical principles of the models and run them on the 'toy' dataset.

The 'toy' dataset is store in files digits.RData. There are 5,000 images in the training set and 10,000 images in the test set with equal number of images for each class. Each image is a 20*20 pixel square. The pixel values are 0 or 1.The training set is stored in a 10\*500\*20\*20 array and the test set is stored in a 10\*1000\*20\*20 array.

#### 2. MNIST dataset
There are 600,000 images in the training set and 100,000 images in the test set. Each image is a 28*28 pixel square. The pixel values are gray scale between 0 and 255.  


## **Challenge**

The followings are challenges that we face and the solutions that we come up with. 
#### 1. Dimension reduction 
There are 28*28 = 784 features (that is the number of pixel positions). We need to decide if we need dimension reduction and what kind of dimension reduction techniques we should use to filter out noise and speed up computation. 

     Solution: We run algorithm under the following three conditions
 
     (1) without dimension reduction
 
     (2) Use principle component analysis (PCA) to do dimension reduction
 
     (3) Summarize 784 features into N by keeping the N biggest singular values


#### 2. Variance in class
The main challenge in handwritten digits recognition is that people don't write the same digits the same way, which makes the distribution assumption of a class invalid or makes finding clear boundaries between classes difficult. 

    Solution: We use nested method and build classification model within a class. 
              We compare the peformance between common method and nested method. 

#### 3. Initialization
Bad initialization could cost more computation time.

    Solution: We use three kinds of initializations:

    (1) random

    (2) class-based-parameter

    (3) KMeans-based-parameter

#### 4. Computation speed 
Besides accuracy, computation speed also matters. 

    Solution: We try the following two approaches to increase speed:

    (1) We use stochastic gradient descent to compute optimal parameters. 

    (2) We divide dataset into minibatches with size 600 and fit model over N epochs. 
        There are 2 main modules in python provide methods for us to load and deal with large-scale data: Keras and Theano. 
        Theano allows us to store the dataset in shared variables and to just copy a minibatch everytime is need. 
        This would lead to large increase in computation speed and decrease in memory usage. 

#### 5. Overfitting problem
    Solution: We use cross-validation to avoid overfitting.

#### 6. Evaluation
    Solution: We use not only classification accuray rate (or error rate) but also rejection to evaluate model performance. 
              We use bootstrap to compute the rejection probability on the test set. 
              Rejection rate is more statistically meaningful than raw error or accuracy rate. 


## **Model**
#### Generative model 

(1) Linear Discriminant analysis (LDA)

LDA assumes a multivariate Gausssian distribution with equal variance and computes the log likelihood given each class to assign class.

(2) Bernoulli Mixture (BMM) and Gaussian Mixture Model (GMM)

- Linear discriminant analysis (LDA) assumes the same covariance matrix for every class

- Quadratic discriminant analysis (QDA) assumes different covariance matrix for every class

- If we're given class labels and use GMM for classification, then we're essentially performing QDA, which is a generative model whose marginal data likelihood is exactly the GMM density.

- When we fit GMM without labels, that's when the EM algorithm comes in. 

- EM algorithm is an iterative algorithm that starts from some initial estimate of parameters (e.g. random), and then proceeds to iteratively update parameters until convergence is detected. 

- We can perform GMM without labels inside each class. This can solve the within class variance problem. 

#### Discriminative model

(3) SVM (Support vector machine) 

SVM performs classification by separating classes with hyperplanes.

(4) Logistic linear regression

- Linear SVMs and logistic regression generally perform comparably in practice. Generally, we use SVM with a nonlinear kernel if we have reason to believe the data won't be linearly separable (or we need to be more robust to outliers than LR will normally tolerate). Otherwise, just try logistic regression first and see how we do with that simpler model. If logistic regression fails, try an SVM with a non-linear kernel like a RBF.

_The foundamental difference between generative and discriminative model is_:

- Discriminative models learn the (hard or soft) boundary between classes

- Generative models model the distribution of individual classes

#### Neural network

(5) Simple neural network 
 
(6) Simple convolutional neural network 

(7) Large convolutional neural network

The network topology of a large convolutional neural network can be summarized as follows:
- Convolutional layer with 30 feature maps of size 5×5.
- Pooling layer taking the max over 2*2 patches.
- Convolutional layer with 15 feature maps of size 3×3.
- Pooling layer taking the max over 2*2 patches.
- Dropout layer with a probability of 20%.
- Flatten layer.
- Fully connected layer with 128 neurons and rectifier activation.
- Fully connected layer with 50 neurons and rectifier activation.
- Output layer.


## **Parameter Optimization**
For all classifiers there will be a free parameter that needs to be adjusted using a holdout part of the training set. 

(1) LDA: smoothing parameter for covariance matrix 

(2) Mixture model: number of components for the mixture model 

(3) SVM and logistica linear regression: regularization parameter 

For these methods, we run the algorithm B times on a random subset of training data per class, and evaluate the error on the remaining data for a value range of the free parameters. We choose the values that minimize the average error rate over the B runs and check the resulting algorithms on the test set.


## **Result**
#### 'toy' dateset

We run algorithms without dimension reduction and with random initial parameters on the 'toy' dataset. 

(1) LDA: the optimal smoothing parameter is 0.1. The correction rate is 0.776 on training and 0.712 on test set. 

(2) BMM: the optimal number of component is 3. The correction rate is 0.823 on training and 0.805 on test set. GMM: the optimal number of component is 3. The correction rate is 0.831 on training and 0.810 on test set. 

(3) SVM: the optimal regularization parameter is 0.0015. The correction rate is 0.922 on training and 0.889 on test set. 

(4) Logistic regression: the optimal regularization parameter is 0.0026. The correction rate is 0.919 on training and 0.8878 on test set. 

#### MNIST dateset
Based on the performance on the 'toy' dataset, we choose to run BMM/GMM, Logistic regression on MNIST. Besides, we build neural network model. We divide dataset into minibatches with size 600 and fit model over N epochs. 

(1) BMM/GMM

- For both methods, Kmeans-based-parameter initialization and PCA dimension reduction give slightly better result than other parameter initializaiton and dimension reduction methods. 
- For GMM, full covariance matrix is better than diagonal convariance matrix. 

For BMM and GMM (full covariance matrix) with kmeans-based-parameter initialization, accuracy and time(in seconds) are reported.

![BMMGMM_profile.JPG]({{site.baseurl}}/BMMGMM_profile.JPG)

- For GMM, PCA-based dimension reduction helps improve performance.

- For BMM, PCA-based dimension reduction decreases performance.

- It's because for GMM, dimension reduction helps filter out noise. However, for BMM, some information has missed after transforming the raw vectors to binary vectors. Dimension reduction would cause more information missing and decrease performance. 


(2) Logistic linear regression

- For logistic linear regression, PCA dimension reduction helps decrease training and test time, but also decreases accuracy. Considering the good performance without dimension reduction, we think dimension reduction in this case is unnecessary. 

![logistic_profile.JPG]({{site.baseurl}}/logistic_profile.JPG)


(3) Neural network

For simple neural network (SNN), convolutioanl neural network (CNN) and large convolutional neural network (LCNN), accuracy and time(in seconds) are reported.

![NN_profile.JPG]({{site.baseurl}}/NN_profile.JPG)
