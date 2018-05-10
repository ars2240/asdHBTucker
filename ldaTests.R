#load packages
library(caret);
library(ldatuning);
library(pROC);
library(tidytext);
library(tm);
library(topicmodels);
library(R.matlab);

#load data
asdSparse <- read.csv("~/Documents/Northwestern/Research/Code/asdSparse.csv", row.names = 1, header= T, check.names=F); #import from csv

#split data into genetic variant and pathway data
#then remove duplicate rows
geneVar=asdSparse[,c(1,2,4)];
geneVar=unique(geneVar);
pathways=asdSparse[,c(1,3,4)];
pathways=aggregate(value ~ row + pathway, pathways, sum);

#concatonate genetic variant and pathway columns
asdSparse$concat=paste(asdSparse[,2],asdSparse[,3],sep="-");

#cast matrices as DocumentTopicMatrix
gvDTM=cast_dtm(geneVar,row,col,value);
pwDTM=cast_dtm(pathways,row,pathway,value);
cDTM=cast_dtm(asdSparse,row,concat,value);

#find number of topics
#Warning: takes a few hours
minTop=2;
maxTop=40;
step=2;
gvNTop = FindTopicsNumber(gvDTM, topics = seq(minTop, maxTop, by = step),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs", control = list(), mc.cores = 2L, verbose = FALSE)
pwNTop = FindTopicsNumber(pwDTM, topics = seq(minTop, maxTop, by = step),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs", control = list(), mc.cores = 2L, verbose = FALSE)
cNTop = FindTopicsNumber(cDTM, topics = seq(minTop, maxTop, by = step),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs", control = list(), mc.cores = 2L, verbose = FALSE)

#plot
jpeg('geneticVariantNumberOfTopics.jpg');
FindTopicsNumber_plot(gvNTop);
dev.off();
jpeg('pathwaysNumberOfTopics.jpg');
FindTopicsNumber_plot(pwNTop);
dev.off();
jpeg('concatenateNumberOfTopics.jpg');
FindTopicsNumber_plot(cNTop);
dev.off();

#perform LDA with set number of topics
nTopics=20;
gvLDA=LDA(gvDTM,nTopics);
pwLDA=LDA(pwDTM,nTopics);
cLDA=LDA(cDTM,nTopics);

#get topic proportion/probabilities
#Note: this may take a few minutes
gvPos=posterior(gvLDA)$topics;
pwPos=posterior(pwLDA)$topics;
cPos=posterior(cLDA)$topics;

#re-order
gvPos=data.frame(gvPos[order(as.numeric(rownames(gvPos))),]);
pwPos=data.frame(pwPos[order(as.numeric(rownames(pwPos))),]);
cPos=data.frame(cPos[order(as.numeric(rownames(cPos))),]);
write.csv(gvPos, paste("~/Documents/Northwestern/Research/Code/gvLDA_", nTopics, ".csv", sep=""));
write.csv(pwPos, paste("~/Documents/Northwestern/Research/Code/pwLDA_", nTopics, ".csv", sep=""));
write.csv(cPos, paste("~/Documents/Northwestern/Research/Code/cLDA_", nTopics, ".csv", sep=""));
# gvPos=read.csv(paste("~/Documents/Northwestern/Research/Code/gvLDA_", nTopics, ".csv", sep=""), row.names = 1);
# pwPos=read.csv(paste("~/Documents/Northwestern/Research/Code/pwLDA_", nTopics, ".csv", sep=""), row.names = 1);
# cPos=read.csv(paste("~/Documents/Northwestern/Research/Code/cLDA_", nTopics, ".csv", sep=""), row.names = 1);
c2Pos=cbind(gvPos,pwPos);
colnames(c2Pos)=paste("X",1:ncol(c2Pos), sep="");

#add binary ASD identifier
nDocs=length(unique(asdSparse[,1]));
gvPos$asd=rep(c(1,0),nDocs/2);
pwPos$asd=rep(c(1,0),nDocs/2);
cPos$asd=rep(c(1,0),nDocs/2);
c2Pos$asd=rep(c(1,0),nDocs/2);

# 70% of the sample size
#smp_size <- floor(0.7 * nDocs)

# set the seed to make your partition reproductible
set.seed(12345)
#train_ind <- sample(seq_len(nDocs), size = smp_size)

data <- readMat("~/Documents/Northwestern/Research/Code/cvInd.mat")
train_ind <- which(data$ind==1)

# take those in training set
gvPos=gvPos[train_ind,];
pwPos=pwPos[train_ind,];
cPos=cPos[train_ind,];
c2Pos=c2Pos[train_ind,];

#create folds for cross validation
nFolds=10;
folds=createFolds(1:smp_size,k=nFolds);

#initialize AUC vectors
gvAUC=c();
pwAUC=c();
cAUC=c();
c2AUC=c();

for(i in 1:nFolds){
  #split data into training and testing sets
  gvTest=gvPos[folds[[i]],];
  pwTest=pwPos[folds[[i]],];
  cTest=cPos[folds[[i]],];
  c2Test=c2Pos[folds[[i]],];
  gvTrain=gvPos[!(1:nDocs %in% folds[[i]]),];
  pwTrain=pwPos[!(1:nDocs %in% folds[[i]]),];
  cTrain=cPos[!(1:nDocs %in% folds[[i]]),];
  c2Train=c2Pos[!(1:nDocs %in% folds[[i]]),];
  
  #logistic regression training
  #Note: this likely will throw a warning about a rank-deficient matrix
  #this is fine (rows of matrix must sum to 1, which we know)
  gvModel=glm(asd ~.,family=binomial(link='logit'),data=gvTrain);
  pwModel=glm(asd ~.,family=binomial(link='logit'),data=pwTrain);
  cModel=glm(asd ~.,family=binomial(link='logit'),data=cTrain);
  c2Model=glm(asd ~.,family=binomial(link='logit'),data=c2Train);
  
  #use model as predictor on test set
  gvPred=predict(gvModel,gvTest);
  pwPred=predict(pwModel,pwTest);
  cPred=predict(cModel,cTest);
  c2Pred=predict(c2Model,c2Test);
  
  #calculate ROC curve and AUC
  gvROC=roc(gvTest$asd,gvPred);
  pwROC=roc(pwTest$asd,pwPred);
  cROC=roc(cTest$asd,cPred);
  c2ROC=roc(c2Test$asd,c2Pred);
  gvAUC[i]=auc(gvROC);
  pwAUC[i]=auc(pwROC);
  cAUC[i]=auc(cROC);
  c2AUC[i]=auc(c2ROC);
}

#print summaries
sink(paste("~/Documents/Northwestern/Research/Code/ldaTests_", toString(nTopics), ".txt", sep=""));
print("Genetic Variants");
print(summary(gvAUC));
print(sd(gvAUC));
print(t.test(gvAUC,mu=0.5));
print("Pathways");
print(summary(pwAUC));
print(sd(pwAUC));
print(t.test(pwAUC,mu=0.5));
print("Concatenation");
print(summary(cAUC));
print(sd(cAUC));
print(t.test(cAUC,mu=0.5));
print("Concatenation v2");
print(summary(c2AUC));
print(sd(c2AUC));
print(t.test(c2AUC,mu=0.5));
sink();
