#divide training set into 400 training exampls and 100 validation examples for each class #run the following algrithom to get the optimal cost parameter C #that minimized the average error rate over five runs on the 100*10 validation data set
library("e1071") error.rate.SVM=matrix(0,ncol=11,nrow=5) #to contain error rates in each iteration
#run five times
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
  train=rbind(train0,train1,train2,train3,train4,train5,train6,train7,train8,train9)
  test=rbind(test0,test1,test2,test3,test4,test5,test6,test7,test8,test9) #combine the 10 samples together again
  labels.train=as.numeric(rownames(train)) #extract the class of each data
  labels.test=as.numeric(rownames(test))
  train=as.data.frame(cbind(labels.train,train))#combine class factor and data as data frame
  test=as.data.frame(cbind(labels.test,test)) train[,1]=as.factor(train[,1]) #treat class as factor
  test[,1]=as.factor(test[,1]) colnames(train)=c("Y",paste("x.",1:400,sep="")) #name the columns of data frame
  colnames(test)=c("Y",paste("x.",1:400,sep=""))
  #let the cost parameter c be different values and compute the average error rate under each value to get the c that minimizes the error rate
  for(j in 0:10){
    c=10^(j-5)
    model.svm=svm(train$Y~.,method="class",data=train,cost=c)
    prediction.SVM=predict(model.svm,newdata=test,type="class")
    error.rate.SVM[i,j+1]=sum(test$Y!=prediction.SVM)/nrow(test)
  }
}
avg.error.rate.SVM=colMeans(error.rate.SVM)
#plot the average error rates versus cost parameter c and get optimal c
c=c(1e-05,1e-04,1e-03,1e-02,1e-01,1,1e01,1e02,1e03,1e04,1e05)
plot(seq(1,11),avg.error.rate.SVM,ylab="error rate",xlab="lambda",type="b",xaxt = 'n')
axis(1,at=1:11,labels=c)
c.optimal=10^(apply(t(avg.error.rate.SVM),1,which.min)-1)

# The above figure shows error rates versus values of cost parameter and
# we get the optimal c=10

#apply SVM with the optimal C=10 on the whole training and test set #prepare data
a=cbind(training.label,training.data)
b=cbind(test.label,test.data)
dataset.train=as.data.frame(a)
dataset.test=as.data.frame(b)
dataset.train[,1]=as.factor(dataset.train[,1])
dataset.test[,1]=as.factor(dataset.test[,1])
colnames(dataset.train)=c("Y",paste("x.",1:400,sep=""))
colnames(dataset.test)=c("Y",paste("x.",1:400,sep=""))
#do SVM
pc=proc.time()
model.svm=svm(dataset.train$Y~.,method="class",data=dataset.train,cost=c.optimal)
proc.time()=pc
#perform on test data set to get the correction rate
prediction.SVM=predict(model.svm,newdata=dataset.test,type="class")
error.rate.SVM=sum(dataset.test$Y!=prediction.SVM)/nrow(dataset.test)
print(paste0("Accuracy(Precision):",1-error.rate.SVM))
#we get correction rate on test data set =0.9109