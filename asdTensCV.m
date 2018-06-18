asdSparse=csvread('asdSparse.csv',1,1);
asd=sptensor(asdSparse(:,1:3),asdSparse(:,4));

pTest=.3; %percent of data in test
rng(12345); %seed RNG
nPat=size(asd,1); %number of patients
ind=crossvalind('HoldOut',nPat,pTest); %split data into test & train sets

%split data based on index into training and testing sets
asd=asd(ind,:,:);

nFolds=10; %set number of folds
nTrain=sum(ind); %size of training set
cvInd=crossvalind('Kfold',nTrain,nFolds); %split data into k folds

options=init_options();
% mex drawZscPar.c CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp";
% options.L = 3;
options.maxIter = 500;
options.freq = 20;
options.treeReps = 2;
options.btReps = 2;
% options.topicModel = 'PAM';

phi=cell(nFolds+1,1);
for f=1:nFolds
    b=cvInd==f; %logical indices of test fold
    ind=find(b);
    
    [phi{f}, ~, ~]=asdHBTucker3(asd(ind,:,:),options);
end
[phi{nFolds+1}, ~, ~]=asdHBTucker3(asd,options);
save('asdHBTuckerCV.mat','phi');