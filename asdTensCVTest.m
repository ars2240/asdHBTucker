load('asdHBTuckerCV.mat'); %load tensor

asdSparse=csvread('asdSparse.csv',1,1);
asd=sptensor(asdSparse(:,1:3),asdSparse(:,4));

pTest=.3; %percent of data in test
rng(12345); %seed RNG
nPat=size(asd,1); %number of patients
ind=crossvalind('HoldOut',nPat,pTest); %split data into test & train sets

%split data based on index into training and testing sets
asd=asd(find(ind),:,:);

nFolds=10; %set number of folds
nTrain=sum(ind); %size of training set
cvInd=crossvalind('Kfold',nTrain,nFolds); %split data into k folds

options=init_options();
% mex drawZscPar.c CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp";
% options.L = 3;
options.maxIter = 1000;
options.freq = 10;
options.treeReps = 1;
options.btReps = 1;
% options.par=0;
% options.topicModel = 'PAM';

testPhi=cell(nFolds,1);
for f=1:nFolds
    b=cvInd==f; %logical indices of test fold
    
    testPhi{f} = asdHBTuckerNew(asd, psi{f}, samples{f}, paths{f}, tree{f}, b, options);
end
save('asdHBTuckerCVTest.mat', 'testPhi');