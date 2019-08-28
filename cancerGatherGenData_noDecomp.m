asdTens=csvread('cancerGenNumber.csv');

load('cancerCVInd.mat'); %load indices

%phiMat=zscore(phiMat); %normalize
asd=csvread('cancerHBTuckerGenDataLabel_3.csv',1,1); %cancer class

%split data based on index into training and testing sets
asdTens=asdTens(ind~=0,:);
trainASD=asd(ind~=0);

cvInd=ind(ind~=0);

nFolds=10; %set number of folds

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
save('cancerHBTuckerGenData_noDecomp.mat','phi', 'testPhi', 'cvTestASD','cvTrainASD');