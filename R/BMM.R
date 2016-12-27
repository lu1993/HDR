#perform the following procedure 5 times for each class k to get the optimal m value for each class
M.cop=rep(0,10) #to contain m values pCorrectTest.BMM=array(0,dim=c(10,4,5)) #to contain correct rates in each iteration
for(j in 0:9){
  train_indj=sample((nrow(training.data[rownames(training.data)==j,])), size=400,replace = FALSE, prob = NULL) #sample rows trainj=training.data[rownames(training.data)==j,][train_ind0, ]#get the training set of class j testj=training.data[rownames(training.data)==j,][-train_ind0, ]#get the test set of class j
  for (m in 2:5){
    for (i in 1:5){ #assign each sample at random to one of the M components
      rownames(trainj)=sample(1:m,size=400,replace=TRUE)
      rownames(testj)=sample(1:m,size=100,replace=TRUE)
      #compute initial values of p and pi
      p=matrix(0,d,m)
      for(k in 1:m){
        p[,k]=(colSums(trainj[rownames(trainj)==k,])+1)/(nrow(trainj[rownames(trainj)==k,])+2)
      }
      pi=matrix(0,1,m)
      for(l in 1:m){
        pi[,l]=(nrow(trainj[rownames(trainj)==l,])+1)/(nrow(trainj)+m)
      }
      #prepare for EM
      loglike=numeric(0)
      #Perform EM.
      iter=0
      while (iter<=1 || abs(diff(tail(loglike, 2))/tail(loglike, 1))>0.001) { # E-step.
        log.q=matrix(0,nrow(trainj),m)
        log.q=as.numeric(log(pi))+trainj%*%log(p)+(1-trainj)%*%log(1-p)
        # Compute log joint likelihood.
        M=apply(log.q, 1, max)
        loglike=c(loglike, sum(log(rowSums(exp(log.q - M))) + M)) # Normalize q.
        q=exp(log.q - M)
        q=q/rowSums(q) # M-step.
        pi=(colSums(q)+1)/(nrow(trainj)+m)
        p=t(apply((t(trainj)%*%q+1), 1, "/", (colSums(q)+2)))
        iter=iter + 1
      }
      #use joint density to classify
      log.q.test=as.numeric(log(pi))+testj%*%log(p)+(1-testj)%*%log(1-p)
      testPredict.BMM=apply(log.q.test, 1, which.max)
      #compute correct rates
      pCorrectTest.BMM[j+1,m-1,i]=sum(testPredict.BMM==rownames(testj))/nrow(testj)
    }
  } 
  #compute average correct rates for each m and get the m that maximizes the average correct rates
  M.cop[j+1]=apply(t(rowMeans(pCorrectTest.BMM[j+1,,])),1,which.max)
} 

#plot correct rates versus M values for each class
plot(seq(1,5),c(1,rowMeans(pCorrectTest.BMM[1,,])),ylab="correction rate",xlab="M value",type="b",xaxt = 'n')
axis(1,at=1:5,labels=seq(1,5))
points(seq(1,5),c(1,rowMeans(pCorrectTest.BMM[2,,])),pch=12)
points(seq(1,5),c(1,rowMeans(pCorrectTest.BMM[3,,])),pch=15)
points(seq(1,5),c(1,rowMeans(pCorrectTest.BMM[4,,])),pch="+")
points(seq(1,5),c(1,rowMeans(pCorrectTest.BMM[5,,])),pch="*")
points(seq(1,5),c(1,rowMeans(pCorrectTest.BMM[6,,])),pch=16)
points(seq(1,5),c(1,rowMeans(pCorrectTest.BMM[7,,])),pch=17)
points(seq(1,5),c(1,rowMeans(pCorrectTest.BMM[8,,])),pch=25)
points(seq(1,5),c(1,rowMeans(pCorrectTest.BMM[9,,])),pch=14)
points(seq(1,5),c(1,rowMeans(pCorrectTest.BMM[10,,])),pch=13)

'''
We can see that for each class, for m>1(m=2,3,4,5), 
the correct rates on the test data set decrease sharply and are all below 0.5. 
Therefore, for each class, we choose m=1 and get the optimal the optimal M=10.
'''

#perform the algorithm on the whole training and test data sets with optimal M=10
#reshape the training data set
dim(training.data)=c(num.class,num.training,d)
#compute initial values of p and pi
p=matrix(0,d,num.class)
for(i in 1:num.class){
  p[,i]=(colSums(training.data[i,,])+1)/(nrow(training.data[i,,])+2)
}
pi=matrix(0,1,num.class)
for(i in 1:num.class){
  pi[,i]=(nrow(training.data[i,,])+1)/(num.class * num.training+10)
}
#prepare for EM
loglike=numeric(0)
#Perform EM.
iter=0
while (iter<=1 || abs(diff(tail(loglike, 2))/tail(loglike, 1))>0.001) { # E-step.
  log.q=matrix(0,num.class * num.training,num.class)
  dim(training.data)=c(num.class * num.training, d)#reshape the training data
  log.q=as.numeric(log(pi))+training.data%*%log(p)+(1-training.data)%*%log(1-p) # Compute log joint likelihood.
  M=apply(log.q, 1, max)
  loglike=c(loglike, sum(log(rowSums(exp(log.q - M))) + M)) # Normalize q.
  q=exp(log.q - M)
  q=q/rowSums(q) # M-step.
  pi=(colSums(q)+1)/(num.class * num.training+10)
  p=t(apply((t(training.data)%*%q+1), 1, "/", (colSums(q)+2)))
  iter=iter + 1
}
#use joint density to classify
trainPredict.BMM=apply(log.q, 1, which.max)
log.q.test=as.numeric(log(pi))+test.data%*%log(p)+(1-test.data)%*%log(1-p)
testPredict.BMM=apply(log.q.test, 1, which.max)
#compute correct rates
pCorrectTrain.BMM=sum((trainPredict.BMM-1)==training.label)/length(training.label)
pCorrectTest.BMM=sum((testPredict.BMM-1)==test.label)/length(test.label)
#pCorrectTrain.BMM=0.7625,pCorrectTest.BMM=0.7125
#i.e., the correction rate on the training data is 0.7625, the correction rate on the test data set is 0.7125
#reshape p(d,m) as a 20*20 array and show the images
dim(p)
dim(p)=c(20,20,num.class)
image(p[,,1],col=gray(seq(0, 1, length.out=256)),axes=FALSE,asp=1)
image(p[,,2],col=gray(seq(0, 1, length.out=256)),axes=FALSE,asp=1)
image(p[,,3],col=gray(seq(0, 1, length.out=256)),axes=FALSE,asp=1)
image(p[,,4],col=gray(seq(0, 1, length.out=256)),axes=FALSE,asp=1)
image(p[,,5],col=gray(seq(0, 1, length.out=256)),axes=FALSE,asp=1)
image(p[,,6],col=gray(seq(0, 1, length.out=256)),axes=FALSE,asp=1)
image(p[,,7],col=gray(seq(0, 1, length.out=256)),axes=FALSE,asp=1)
image(p[,,8],col=gray(seq(0, 1, length.out=256)),axes=FALSE,asp=1)
image(p[,,9],col=gray(seq(0, 1, length.out=256)),axes=FALSE,asp=1)
image(p[,,10],col=gray(seq(0, 1, length.out=256)),axes=FALSE,asp=1)

'''
Show the probability vectors obtained and  
We can see that for m=1,2,...,10, the images of the arrays 
are just images of numbers 0,1,2,3,4,5,6,7,8,9
'''