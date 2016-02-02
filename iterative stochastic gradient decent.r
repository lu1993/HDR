# Iterative Stochastic Gradient Descent
#divide training set into 400 training exampls and 100 validation examples for each class #run the following algrithom to get the optimal lambda #that minimized the average error rate over five runs on the 100*10 validation data set
#initialize
lambda=1e-07 
wt=matrix(0,10,400)#matrix for intial values of Wk(k=1,2,...,10) pCorrectTest.SVMb=matrix(0,ncol=7,nrow=5)#to contain correct rate of test set under each lambda
set.seed(123)
# run five times
for(i in 1:5){
  train_ind0=sample((nrow(training.data[rownames(training.data)=="0",])), size=400,replace = FALSE, prob = NULL) #sample rows train0=training.data[rownames(training.data)=="0",][train_ind0, ]#get the training set of class 0 test0=training.data[rownames(training.data)=="0",][-train_ind0, ]#get the test set of class 0
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
  #let lambda=1e-6,1e-5,1e-4,...,1, get the correct rates on test set
  for(k in 1:7){
    lambda=lambda*10 
    eta0=1/(1+lambda)# set eta0=1/(1+lambda) #for each class
    for(j in 0:9){ #sweep the data set 100 times
      for(s in 1:100){
        etat=eta0/s #randomly shuffle training data
        train_ind=sample((nrow(train)),size=4000,replace = FALSE, prob = NULL)
        train_s=train[train_ind, ] #for examples of class j,set Yn=1, otherwise Yn=-1
        y=(-1)^((rownames(train_s)==j)+1)
        for(n in 1:5000){
          if(y[n]*train_s[n,]%*%wt[j+1,]>1){wt[j+1,]=(1-etat*lambda)*wt[j+1,]}
          else{wt[j+1,]=(1-etat*lambda)*wt[j+1,]+etat*train_s[n,]*y[n]}
        }
      }
    } #now we get W for 10 classes #use W to classify test set
    w=t(wt)
    test.w=test%*%w
    testPredict.SVMb=apply(test.w, 1, which.max)
    #compute correct rates
    pCorrectTest.SVMb[i,k]=sum((testPredict.SVMb-1)==test.label)/length(test.label)
  }
}
#compare the average correction rates under each lambda #get the optimal lambda that maximizes the average correction rate
lambda.optimal=10^(apply(colMeans(pCorrectTest.SVMb),1,which.max)-1)
# plot the average error rates versus lambda
error.rate.SVMb=1-colMeans(pCorrectTest.SVMb)
l=c(1e-06,1e-05,1e-04,1e-03,1e-02,1e-01,1)
plot(c(1:7),error.rate.SVMb,ylab="error rate",xlab="lambda",type="b",xaxt = 'n')
axis(1,at=1:7,labels=l)
# The above plot shows the average error rates on test set versus lambda,
# we get the optimal lambda=0.001

#Apply this lambda and perform stochastic GD on the whole training and test data
lambda=1e-03 eta0=1/(1+lambda)# set eta0=1/(1+lambda) wt=matrix(0,10,400)#matrix for intial values of Wk(k=1,2,...,10)
set.seed(123)
#for each class
for(j in 0:9){ #sweep the data 100 times
  for(s in 1:100){
    etat=eta0/s #randomly shuffle training data
    train_ind=sample((nrow(training.data)),size=5000,replace = FALSE, prob = NULL)
    train_s=training.data[train_ind, ] #for examples of class j,set Yn=1, otherwise Yn=-1
    y=(-1)^((rownames(train_s)==j)+1)
    for(n in 1:5000){
      if(y[n]*train_s[n,]%*%wt[j+1,]>1){wt[j+1,]=(1-etat*lambda)*wt[j+1,]}
      else{wt[j+1,]=(1-etat*lambda)*wt[j+1,]+etat*train_s[n,]*y[n]}
    }
  }
} #now we get W for 10 classes #use W to classify test set
w=t(wt)
train.w=training.data%*%w
test.w=test.data%*%w
trainPredict.SVMb=apply(train.w,1,which.max)
testPredict.SVMb=apply(test.w, 1, which.max)
#compute correct rates
pCorrectTrain.SVMb=sum((trainPredict.SVMb-1)==training.label)/length(training.label)
pCorrectTest.SVMb=sum((testPredict.SVMb-1)==test.label)/length(test.label)
print(paste0("Accuracy(Precision):", pCorrectTest.SVMb))
#get correction rate on the test data set=0.915