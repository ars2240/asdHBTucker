load('cancerHBTuckerGenData.mat');

asd=sptensor(sparse(:,1:2),sparse(:,4),[max(sparse(:,1)), ...
    max(sparse(:,2))],@max);
% asd=sptensor(sparse(:,1:3),sparse(:,4));
% asd=sptenmat(asd,1);
% asd=double(asd);

% save('cancerGenNumberBoth', 'asd')
csvwrite('cancerGenNumber.csv', double(asd));

pTest=.3; %percent of data in test
rng(12345); %seed RNG
nPat=size(asd,1); %number of patients
ind=crossvalind('HoldOut',nPat,pTest); %split data into test & train sets

ind=int8(ind);

nFolds=10; %set number of folds
nTrain=sum(ind); %size of training set
cvInd=crossvalind('Kfold',nTrain,nFolds); %split data into k folds
ind(ind~=0)=cvInd;

csvwrite('cancerGenCVInd.csv', ind);
