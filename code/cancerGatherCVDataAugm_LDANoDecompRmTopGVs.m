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
asdTens=asdTens(ind,:);
trainASD=asd(ind);
testASD=asd(~ind);

nFolds=10; %set number of folds
threshold=0; %threshold for considering topic
nGVs=5; %top # of GVs to remove
nTop=20; %number of LDA topics
nTrain=sum(ind); %size of training set
cvInd=crossvalind('Kfold',nTrain,nFolds); %split data into k folds

cvTestASD=cell(nFolds,1);
cvTrainASD=cell(nFolds,1);
phi=cell(nFolds,1);
testPhi=cell(nFolds,1);

for i=1:nFolds
    b=cvInd==i; %logical indices of test fold
    
    phiT=csvread(['data/cancer_py_gvLDA_', int2str(nTop), '_', ...
        int2str(i), '_train.csv'],1,1);
    testPhiT=csvread(['data/cancer_py_gvLDA_', int2str(nTop), '_', ...
        int2str(i), '_valid.csv'],1,1);
    topics=csvread(['data/cancer_py_gvLDA_', int2str(nTop), '_', ...
        int2str(i), '_genes.csv'],1,1)';
    
    if nGVs>0
        [~,topTops]=sort(topics,'descend');
        topTops=topTops(1:nGVs,:);
        topTops=reshape(topTops,[nGVs*nTop,1]);
        topTops=unique(topTops);

        gvs=setdiff(1:size(asdTens,2),topTops);
    else
        gvs=1:size(asdTens,2);
    end
    
    phiT=phiT>threshold;
    testPhiT=testPhiT>threshold;
    
    phi{i}=[asdTens(~b,gvs),phiT];
    testPhi{i}=[asdTens(b,gvs),testPhiT];
    
    %split data based on index into training and testing sets
    cvTestASD{i}=trainASD(b,:);
    cvTrainASD{i}=trainASD(~b,:);
    
end
save('cancerHBTuckerCVDataAugm_LDANoDecompRmTopGVs.mat','phi', 'testPhi', 'cvTestASD','cvTrainASD');
