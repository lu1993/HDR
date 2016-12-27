#reshape the dataset
dim(training.data) <- c(num.class * num.training, d) # Reshape training data to 2-dim matrix
dim(test.data) <- c(num.class * num.test, d) # Same for test.
training.label <- rep(0:9, num.training) # Labels of training data.
test.label <- rep(0:9, num.test) # Labels of test data

#install glmnet packages
install.packages("glmnet", repos = "http://cran.us.r-project.org")
library("glmnet")

#divide training data into training and validation parts to compute optimal lambda
set.seed(123)
accuracy=matrix(0,nrow=5,ncol=86)
lambda=matrix(0,nrow=5,ncol=86)
for(i in 1:5){
  
  train_ind0=sample((nrow(training.data[rownames(training.data)=="0",])), 
                    size=400,replace = FALSE, prob = NULL) #sample rows
  train0=training.data[train_ind0, ]#get the training set of class 0
  test0=training.data[-train_ind0, ]#get the test set of class 0
  
  train_ind1=sample((nrow(training.data[rownames(training.data)=="1",])), 
                    size=400,replace = FALSE, prob = NULL) #sample rows
  train1=training.data[train_ind1, ]#get the training set of class 1
  test1=training.data[-train_ind1, ]#get the test set of class 1
  
  train_ind2=sample((nrow(training.data[rownames(training.data)=="2",])), 
                    size=400,replace = FALSE, prob = NULL) 
  train2=training.data[train_ind2, ]
  test2=training.data[-train_ind2, ]
  
  train_ind3=sample((nrow(training.data[rownames(training.data)=="3",])), 
                    size=400,replace = FALSE, prob = NULL) 
  train3=training.data[train_ind3, ]
  test3=training.data[-train_ind3, ]
  
  train_ind4=sample((nrow(training.data[rownames(training.data)=="4",])), 
                    size=400,replace = FALSE, prob = NULL) 
  train4=training.data[train_ind4, ]
  test4=training.data[-train_ind4, ]
  
  train_ind5=sample((nrow(training.data[rownames(training.data)=="5",])), 
                    size=400,replace = FALSE, prob = NULL) 
  train5=training.data[train_ind5, ]
  test5=training.data[-train_ind5, ]
  
  train_ind6=sample((nrow(training.data[rownames(training.data)=="6",])), 
                    size=400,replace = FALSE, prob = NULL) 
  train6=training.data[train_ind6, ]
  test6=training.data[-train_ind6, ]
  
  train_ind7=sample((nrow(training.data[rownames(training.data)=="7",])), 
                    size=400,replace = FALSE, prob = NULL) 
  train7=training.data[train_ind7, ]
  test7=training.data[-train_ind7, ]
  
  train_ind8=sample((nrow(training.data[rownames(training.data)=="8",])), 
                    size=400,replace = FALSE, prob = NULL) 
  train8=training.data[train_ind8, ]
  test8=training.data[-train_ind8, ]
  
  train_ind9=sample((nrow(training.data[rownames(training.data)=="9",])), 
                    size=400,replace = FALSE, prob = NULL) 
  train9=training.data[train_ind9, ]
  test9=training.data[-train_ind9, ]
  
  train=rbind(train0,train1,train2,train3,train4,train5,train6,train7,train8,train9)
  test=rbind(test0,test1,test2,test3,test4,test5,test6,test7,test8,test9)
  
  #do logistic regression on the sampled training set
  model=glmnet(train,rownames(train),"multinomial")
  lambda[i,]=t(model$lambda)
  for(j in 1:length(model$lambda)){
    yHat=predict(model,test,model$lambda[j],"class")
    accuracy[i,j]=mean(yHat==rownames(test))
  }
}
pCorrection.test.LR=colMeans(accuracy)
lambda.optimal=mean(lambda[,apply(t(pCorrection.test.LR),1,which.max)])
plot(log(colMeans(lambda)),pCorrection.test.LR,type="l")
#get lambda.optimal=0.002615186 

#do logistic regression on the whole training data
model=glmnet(training.data,training.label,"multinomial")

#do test on test data
yHat.train=predict(model,training.data,lambda.optimal,"class")
yHat.test=predict(model,test.data,lambda.optimal,"class")
accuracy.test=mean((yHat.test==test.label))#accuracy.test=0.8678
accuracy.train=mean((yHat.train==training.label))#accuracy.train=0.9192
  