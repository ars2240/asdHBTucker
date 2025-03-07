asdTens=csvread('cancerHBTGenNumberGV.csv');

ind = csvread('cancerGenCVInd.csv');

%phiMat=zscore(phiMat); %normalize
asd=csvread('cancerHBTGenGVLabel.csv',1,1); %cancer class

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
    
    phi{i}=sparse(asdTens(~b,:));
    testPhi{i}=sparse(asdTens(b,:));
    
    %split data based on index into training and testing sets
    cvTestASD{i}=trainASD(b,:);
    cvTrainASD{i}=trainASD(~b,:);
    
end
save('cancerHBTGenGV_noDecomp.mat','phi', 'testPhi', 'cvTestASD','cvTrainASD');
