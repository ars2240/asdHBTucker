asdSparse=csvread('cancerSparseGenes.csv',1,1);
asdTens=sptensor(asdSparse(:,1:2),asdSparse(:,3));
asdTens=double(asdTens);

%run only once, keep constant
%or use seed
pTest=.3; %percent of data in test
rng(12345); %seed RNG
nPat=size(asdTens,1); %number of patients
ind=crossvalind('HoldOut',nPat,pTest); %split data into test & train sets

%phiMat=zscore(phiMat); %normalize
asd=csvread('cancerLabel.csv',1,1); %cancer class

%split data based on index into training and testing sets
trainASD=asd(ind);
testASD=asd(~ind);

nFolds=10; %set number of folds
nTrain=sum(ind); %size of training set
cvInd=crossvalind('Kfold',nTrain,nFolds); %split data into k folds

phi=cell(nFolds,1);
testPhi=cell(nFolds,1);
cvTestASD=cell(nFolds,1);
cvTrainASD=cell(nFolds,1);

for i=1:nFolds
    b=cvInd==i; %logical indices of test fold
    
    phi{i}=asdTens(~b,:);
    testPhi{i}=asdTens(b,:);
    
    %split data based on index into training and testing sets
    cvTestASD{i}=trainASD(b,:);
    cvTrainASD{i}=trainASD(~b,:);
    
end
save('cancerHBTuckerCVData_noDecomp.mat','phi', 'testPhi', 'cvTestASD','cvTrainASD');