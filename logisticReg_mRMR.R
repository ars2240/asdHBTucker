library(caret)
library(mRMRe)
library(pROC)
library(R.matlab)

nfeat=100 # select number of features

# load data set
asdData <- read.csv("~/Documents/Northwestern/Research/Code/asdHBTucker_gam0.1.csv", header=F, stringsAsFactors=F)
colnames(asdData)[1]="asd"
nDocs=nrow(asdData)

set.seed(12345)
#train_ind <- sample(seq_len(nDocs), size = smp_size)

data <- readMat("~/Documents/Northwestern/Research/Code/cvInd.mat")
train_ind <- which(data$ind==1)

# take those in training set
asdData=asdData[train_ind,]

# format as numerical
for(i in 1:ncol(asdData)){
  asdData[,i]=as.numeric(asdData[,i])
}

#create folds for cross validation
nFolds=10;
folds=createFolds(1:length(train_ind),k=nFolds)

AUC=c();
AUCtr=c();

for(i in 1:nFolds){
  #split data into training and testing sets
  asdTest=asdData[folds[[i]],]
  asdTrain=asdData[!(1:nDocs %in% folds[[i]]),]
  
  dd <- mRMR.data(data = asdTrain)
  filter <- mRMR.classic(data = dd, target_indices = c(1), feature_count = nfeat)  #run mRMR
  
  # get columns and subset
  cols <- c(solutions(filter)$'1')
  asdTrain <- asdTrain[,c(1,cols)]
  asdTest <- asdTest[,c(1,cols)]
  
  #logistic regression training
  #Note: this likely will throw a warning about a rank-deficient matrix
  #this is fine (rows of matrix must sum to 1, which we know)
  asdModel=glm(asd ~.,family=binomial(link='logit'),data=asdTrain)
  
  #use model as predictor on test set
  asdPred=predict(asdModel,asdTest)
  asdPredtr=predict(asdModel,asdTrain)
  
  #calculate ROC curve and AUC
  asdROC=roc(asdTest$asd,asdPred)
  asdROCtr=roc(asdTrain$asd,asdPredtr)
  AUC[i]=auc(asdROC)
  AUCtr[i]=auc(asdROCtr)
}

#print summaries
sink(paste("~/Documents/Northwestern/Research/Code/mRMR_", toString(nfeat), ".txt", sep=""))
print("Validation")
print(mean(AUC))
print(sd(AUC))
print(t.test(AUC,mu=0.5)$p.value)
print("Train")
print(mean(AUCtr))
print(sd(AUCtr))
print(t.test(AUCtr,mu=0.5)$p.value)
sink()
