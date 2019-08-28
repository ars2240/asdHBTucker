asdSparse=csvread('cancerSparseGenes.csv',1,1);
asdTens=sptensor(asdSparse(:,1:2),asdSparse(:,3));
asdTens=double(asdTens);

load('cancerCVInd.mat'); %load indices

%phiMat=zscore(phiMat); %normalize
asd=csvread('cancerLabel.csv',1,1); %cancer class

%split data based on index into training and testing sets
asdTens=asdTens(ind~=0,:);
trainASD=asd(ind~=0);

cvInd=ind(ind~=0);

nFolds=10; %set number of folds
nTop=20; %number of topics

phi=cell(nFolds,1);
testPhi=cell(nFolds,1);
cvTestASD=cell(nFolds,1);
cvTrainASD=cell(nFolds,1);

for i=1:nFolds
    b=cvInd==i; %logical indices of test fold
    
    phi{i}=csvread(['data/cancer_py_gvLDA_', int2str(nTop), '_', ...
        int2str(i), '_cheating_train.csv'],1,1);
    testPhi{i}=csvread(['data/cancer_py_gvLDA_', int2str(nTop), '_', ...
        int2str(i), '_cheating_valid.csv'],1,1);
    
    phi{i}=phi{i}(~b,:);
    
    %split data based on index into training and testing sets
    cvTestASD{i}=trainASD(b,:);
    cvTrainASD{i}=trainASD(~b,:);
    
end
save('cancerHBTuckerCVData_gvLDA_cheating.mat','phi', 'testPhi', 'cvTestASD','cvTrainASD');