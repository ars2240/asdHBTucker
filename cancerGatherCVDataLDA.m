asdSparse=csvread('cancerSparse.csv',1,1);
asdTens=sptensor(asdSparse(:,1:3),asdSparse(:,4));

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

cvTestASD=cell(nFolds,1);
cvTrainASD=cell(nFolds,1);
phi=cell(nFolds,1);
testPhi=cell(nFolds,1);

for i=1:nFolds
    b=cvInd==i; %logical indices of test fold
    
    phi{i}=csvread(['data/cancer_gvLDA_20_', int2str(i), '_train.csv'],1,1);
    testPhi{i}=csvread(['data/cancer_gvLDA_20_', int2str(i), '_valid.csv'],1,1);
    
    %split data based on index into training and testing sets
    cvTestASD{i}=trainASD(b,:);
    cvTrainASD{i}=trainASD(~b,:);
    
end
save('cancerHBTuckerCVDataLDA.mat','phi', 'testPhi', 'cvTestASD','cvTrainASD');