ind = csvread('cancerGenCVInd.csv');

%phiMat=zscore(phiMat); %normalize
asd=csvread('cancerLDAGenDataLabel.csv',1,1); %cancer class

% rmClass = [];  %classes to be removed
% numClass = max(asd)+1; % number of old classes
% nClass = -1*ones(1,numClass);
% nClass(not(ismember(1:numClass,rmClass)))=(1:(numClass-length(rmClass)))-1;

%split data based on index into training and testing sets
trainASD=asd(ind~=0);

cvInd=ind(ind~=0);

nFolds=10; %set number of folds
nTop=20; %number of topics
head='data/cancer_py_gen_gvLDA_'; %file header

phi=cell(nFolds,1);
testPhi=cell(nFolds,1);
cvTestASD=cell(nFolds,1);
cvTrainASD=cell(nFolds,1);

for i=1:nFolds
    b=cvInd==i; %logical indices of test fold
    
    phi{i}=csvread(['data/', head, int2str(nTop), '_',int2str(i), ...
        '_train.csv'],1,1);
    testPhi{i}=csvread(['data/', head, int2str(nTop), '_', int2str(i), ...
        '_valid.csv'],1,1);
    
    %split data based on index into training and testing sets
    cvTestASD{i}=trainASD(b,:);
    cvTrainASD{i}=trainASD(~b,:);
    
    %remove classes
%     keep = not(ismember(cvTrainASD{i},rmClass));
%     phi{i}=phi{i}(keep,:);
%     cvTrainASD{i}=nClass(cvTrainASD{i}(keep)+1);
%     keep = not(ismember(cvTestASD{i},rmClass));
%     testPhi{i}=testPhi{i}(keep,:);
%     cvTestASD{i}=nClass(cvTestASD{i}(keep)+1);
    
end
save('cancerLDACVData_gvLDAGen.mat','phi', 'testPhi', 'cvTestASD','cvTrainASD');
