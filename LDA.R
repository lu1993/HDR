#input the data
load("C:/Users/mac/Desktop/spring course/246/midtermproject/digits.RData")
#set up the data for use num.class <- dim(training.data)[1] # Number of classes num.training <- dim(training.data)[2] # Number of training data per class d <- prod(dim(training.data)[3:4]) # Dimension of each training image (rowsxcolumns) num.test <- dim(test.data)[2] # Number of test data dim(training.data) <- c(num.class * num.training, d) # Reshape training data to 2-dim matrix dim(test.data) <- c(num.class * num.test, d) # Same for test. training.label <- rep(0:9, num.training) # Labels of training data. test.label <- rep(0:9, num.test) # Labels of test data rownames(training.data)=training.label #rownames of training data rownames(test.data)=test.label #rownames of test data
##project 1: LDA
#makeLDADiscF:returns a function that will form a discriminant
makeLDADiscF=function (mean,sigma,prior){
  sigmaInv=solve(sigma )
  constantPart=((0.5)*((t(mean)%*%sigmaInv)%*%mean))[1,1]
  function (X) {
    (((X %*%sigmaInv)%*%mean)-constantPart+log(prior))
  }
}
#Compute the value of smoothing parameter by the following procedure: #divide the training set per class into 2 parts:400 training examples and 100 test examples #compute the adjusted covariance matrix with different smoothing parameters #run the algorithm above five times on a random subset of 400 training examples per class #evaluate the error on the remaining 100 examples per class #obtain the value of smoothing parameter that minimizes the average error of the 5 times
set.seed(123)
pCorrectTrain=matrix(NA,nrow=5,ncol=100) pCorrectTest=matrix(NA,nrow=5,ncol=100) #to contain correct rates under each iteration
#run five times
for(i in 1:5){
  train_ind0=sample((nrow(training.data[rownames(training.data)=="0",])), size=400,replace = FALSE, prob = NULL) #sample rows
  train0=training.data[rownames(training.data)=="0",][train_ind0, ]#get the training set of class 0 test0=training.data[rownames(training.data)=="0",][-train_ind0, ]#get the test set of class 0
  train_ind1=sample((nrow(training.data[rownames(training.data)=="1",])),
                    size=400,replace = FALSE, prob = NULL)
  train1=training.data[rownames(training.data)=="1",][train_ind1, ]
  test1=training.data[rownames(training.data)=="1",][-train_ind1, ]
  train_ind2=sample((nrow(training.data[rownames(training.data)=="2",])),
                    size=400,replace = FALSE, prob = NULL)
  train2=training.data[rownames(training.data)=="2",][train_ind2, ]
  test2=training.data[rownames(training.data)=="2",][-train_ind2, ]
  train_ind3=sample((nrow(training.data[rownames(training.data)=="3",])),
                    size=400,replace = FALSE, prob = NULL)
  train3=training.data[rownames(training.data)=="3",][train_ind3, ]
  test3=training.data[rownames(training.data)=="3",][-train_ind3, ]
  train_ind4=sample((nrow(training.data[rownames(training.data)=="4",])),
                    size=400,replace = FALSE, prob = NULL)
  train4=training.data[rownames(training.data)=="4",][train_ind4, ]
  test4=training.data[rownames(training.data)=="4",][-train_ind4, ]
  train_ind5=sample((nrow(training.data[rownames(training.data)=="5",])),
                    size=400,replace = FALSE, prob = NULL)
  train5=training.data[rownames(training.data)=="5",][train_ind5, ]
  test5=training.data[rownames(training.data)=="5",][-train_ind5, ]
  train_ind6=sample((nrow(training.data[rownames(training.data)=="6",])),
                    size=400,replace = FALSE, prob = NULL)
  train6=training.data[rownames(training.data)=="6",][train_ind6, ]
  test6=training.data[rownames(training.data)=="6",][-train_ind6, ]
  train_ind7=sample((nrow(training.data[rownames(training.data)=="7",])),
                    size=400,replace = FALSE, prob = NULL)
  train7=training.data[rownames(training.data)=="7",][train_ind7, ]
  test7=training.data[rownames(training.data)=="7",][-train_ind7, ]
  train_ind8=sample((nrow(training.data[rownames(training.data)=="8",])),
                    size=400,replace = FALSE, prob = NULL)
  train8=training.data[rownames(training.data)=="8",][train_ind8, ]
  test8=training.data[rownames(training.data)=="8",][-train_ind8, ]
  train_ind9=sample((nrow(training.data[rownames(training.data)=="9",])),
                    size=400,replace = FALSE, prob = NULL)
  train9=training.data[rownames(training.data)=="9",][train_ind9, ]
  test9=training.data[rownames(training.data)=="9",][-train_ind9, ]
  #combine 10 samples together again
  train=rbind(train0,train1,train2,train3,train4,train5,train6,train7,train8,train9)
  test=rbind(test0,test1,test2,test3,test4,test5,test6,test7,test8,test9)
  #compute the covariance matrix and mean of each class with 400 training examples
  cov0=cov(train0)
  cov1=cov(train1)
  cov2=cov(train2)
  cov3=cov(train3)
  cov4=cov(train4)
  cov5=cov(train5)
  cov6=cov(train6)
  cov7=cov(train7)
  cov8=cov(train8)
  cov9=cov(train9)
  mean0=colMeans(train0)
  mean1=colMeans(train1)
  mean2=colMeans(train3)
  mean4=colMeans(train4)
  mean5=colMeans(train5)
  mean6=colMeans(train6)
  mean7=colMeans(train7)
  mean8=colMeans(train8)
  mean9=colMeans(train9)
  avgcov=(cov0+cov1+cov2+cov3+cov4+cov5+cov6+cov7+cov8+cov9)/10 #average covaraince
  # let the smooth parameter k=0.01,0.02,0.03,...,1
  for(j in 1:100){
    k=seq(0,1,by=0.01)[j+1] adj.avgcov=(1-k)*avgcov+k*diag(1/4,nrow=400, ncol=400) #adjusted average covariance
    ldadisc0=makeLDADiscF(mean0,adj.avgcov,1/10)
    ldadisc1=makeLDADiscF(mean1,adj.avgcov,1/10)
    ldadisc2=makeLDADiscF(mean2,adj.avgcov,1/10)
    ldadisc3=makeLDADiscF(mean3,adj.avgcov,1/10)
    ldadisc4=makeLDADiscF(mean4,adj.avgcov,1/10)
    ldadisc5=makeLDADiscF(mean5,adj.avgcov,1/10)
    ldadisc6=makeLDADiscF(mean6,adj.avgcov,1/10)
    ldadisc7=makeLDADiscF(mean7,adj.avgcov,1/10)
    ldadisc8=makeLDADiscF(mean8,adj.avgcov,1/10) ldadisc9=makeLDADiscF(mean9,adj.avgcov,1/10) #10 discriminants
    #use discriminants to classify validation set
    trainPredicted=apply(cbind(ldadisc0(train),
                               ldadisc1(train),
                               ldadisc2(train),
                               ldadisc3(train),
                               ldadisc4(train),
                               ldadisc5(train),
                               ldadisc6(train),
                               ldadisc7(train),
                               ldadisc8(train),
                               ldadisc9(train)),1,which.max)
    testPredicted=apply(cbind(ldadisc0(test),
                              ldadisc1(test),
                              ldadisc2(test),
                              ldadisc3(test),
                              ldadisc4(test),
                              ldadisc5(test),
                              ldadisc6(test),
                              ldadisc7(test),
                              ldadisc8(test),
                              ldadisc9(test)),1,which.max)
    #compute the correct rates
    pCorrectTrain[i,j]=sum((trainPredicted-1)==rownames(train))/nrow(train)
    pCorrectTest[i,j]=sum((testPredicted-1)==rownames(test))/nrow(test)
  }
} pCorrectTest.mean=colMeans(pCorrectTest) #compute the average correct rates
#plot the average correct rates versus the smoothing parameter k
plot(seq(1:100),pCorrectTest.mean,ylab="correction rate",xlab="smooth parameter k",type="b",xaxt = 'n')
k=seq(0,1,by=0.01)[2:101]
axis(1,at=1:100,labels=k)
#get the value of smoothing parameter that maximizes the average correct rate
smooth.k=0.1*(apply(pCorrectionTest.mean,1,which.max)-1)
# The above plot shows the average correct rates on the validation data sets versus k
# we get the optimal smoothing parameter k=0.1

#apply the smoothing parameter k=0.1 on all the training and test data
#compute covariance matrix of the 10 classes
Cov0=cov(training.data[rownames(training.data)=="0",])
Cov1=cov(training.data[rownames(training.data)=="1",])
Cov2=cov(training.data[rownames(training.data)=="2",])
Cov3=cov(training.data[rownames(training.data)=="3",])
Cov4=cov(training.data[rownames(training.data)=="4",])
Cov5=cov(training.data[rownames(training.data)=="5",])
Cov6=cov(training.data[rownames(training.data)=="6",])
Cov7=cov(training.data[rownames(training.data)=="7",])
Cov8=cov(training.data[rownames(training.data)=="8",])
Cov9=cov(training.data[rownames(training.data)=="9",])
#compute the average covariance matrix
avgCov=(Cov0+Cov1+Cov2+Cov3+Cov4+Cov5+Cov6+Cov7+Cov8+Cov9)/10
#adjusted average covariance
adj.avgCov=(1-smooth.k)*avgCov+smooth.k*diag(1/4,nrow=400, ncol=400)
#compute the means of the 10 classes
Mean0=colMeans(training.data[rownames(training.data)=="0",])
Mean1=colMeans(training.data[rownames(training.data)=="1",])
Mean2=colMeans(training.data[rownames(training.data)=="2",])
Mean3=colMeans(training.data[rownames(training.data)=="3",])
Mean4=colMeans(training.data[rownames(training.data)=="4",])
Mean5=colMeans(training.data[rownames(training.data)=="5",])
Mean6=colMeans(training.data[rownames(training.data)=="6",])
Mean7=colMeans(training.data[rownames(training.data)=="7",])
Mean8=colMeans(training.data[rownames(training.data)=="8",])
Mean9=colMeans(training.data[rownames(training.data)=="9",])
#get the discriminants of the 10 classes
Ldadisc0=makeLDADiscF(Mean0,adj.avgCov,1/10)
Ldadisc1=makeLDADiscF(Mean1,adj.avgCov,1/10)
Ldadisc2=makeLDADiscF(Mean2,adj.avgCov,1/10)
Ldadisc3=makeLDADiscF(Mean3,adj.avgCov,1/10)
Ldadisc4=makeLDADiscF(Mean4,adj.avgCov,1/10)
Ldadisc5=makeLDADiscF(Mean5,adj.avgCov,1/10)
Ldadisc6=makeLDADiscF(Mean6,adj.avgCov,1/10)
Ldadisc7=makeLDADiscF(Mean7,adj.avgCov,1/10)
Ldadisc8=makeLDADiscF(Mean8,adj.avgCov,1/10)
Ldadisc9=makeLDADiscF(Mean9,adj.avgCov,1/10)
#use discriminants to classify test data set
TrainPredicted=apply(cbind(Ldadisc0(training.data),
                           Ldadisc1(training.data),
                           Ldadisc2(training.data),
                           Ldadisc3(training.data),
                           Ldadisc4(training.data),
                           Ldadisc5(training.data),
                           Ldadisc6(training.data),
                           Ldadisc7(training.data),
                           Ldadisc8(training.data),
                           Ldadisc9(training.data)),1,which.max)
TestPredicted=apply(cbind(Ldadisc0(test.data),
                          Ldadisc1(test.data),
                          Ldadisc2(test.data),
                          Ldadisc3(test.data),
                          Ldadisc4(test.data),
                          Ldadisc5(test.data),
                          Ldadisc6(test.data),
                          Ldadisc7(test.data),
                          Ldadisc8(test.data),
                          Ldadisc9(test.data)),1,which.max)
#compute the correct rates
PCorrectTrain=sum((TrainPredicted-1)==training.label)/length(training.label)
PCorrectTest=sum((TestPredicted-1)==test.label)/length(test.label)
#get PCorrectTrain=0.9125, PCorrectTest=0.8975
#i.e., correction rate on training data set=0.9125, correction rate on test data set=0.8975