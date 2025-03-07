asdSparse=csvread('cancerSparse.csv',1,1);
asd=sptensor(asdSparse(:,1:3),asdSparse(:,4));
%asd=sptensor(asdSparse(:,1:3),ones(size(asdSparse,1),1));

pTest=.3; %percent of data in test
rng(12345); %seed RNG
nPat=size(asd,1); %number of patients
ind=crossvalind('HoldOut',nPat,pTest); %split data into test & train sets
ind=double(ind);

nFolds=2; %set number of folds
nTrain=sum(ind); %size of training set
cvInd=crossvalind('Kfold',nTrain,nFolds); %split data into k folds
ind(ind~=0)=cvInd;

save('cancerCVInd_2.mat','ind');
csvwrite('cancerCVInd_2.csv',ind);
