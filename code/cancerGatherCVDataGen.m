load('cancerCVInd.mat'); %load indicess

%phiMat=zscore(phiMat); %normalize
asd=csvread('cancerHBTuckerGenDataLabel_3.csv',1,1); %cancer class

% rmClass = [];  %classes to be removed
% numClass = max(asd)+1; % number of old classes
% nClass = -1*ones(1,numClass);
% nClass(not(ismember(1:numClass,rmClass)))=(1:(numClass-length(rmClass)))-1;

%split data based on index into training and testing sets
trainASD=asd(ind~=0);

cvInd=ind(ind~=0);

nFolds=10; %set number of folds
nTop=5; %number of topics

cvTestASD=cell(nFolds,1);
cvTrainASD=cell(nFolds,1);
cphi = cell(nFolds,1);
ctestPhi = cell(nFolds,1);

for i=1:nFolds
    load(['data/cancerHBTuckerCVGen_L3_gam1_', int2str(i) ,'_trees.mat']); %load tensor
    
    cphi{i} = sum(double(phi),3);
    ctestPhi{i} = sum(double(testPhi),3);
    
    
    b=cvInd==i; %logical indices of test fold
    
    %split data based on index into training and testing sets
    cvTestASD{i}=trainASD(b,:);
    cvTrainASD{i}=trainASD(~b,:);
    
    %remove classes
%     keep = find(not(ismember(cvTrainASD{i},rmClass)));
%     cphi{i}=cphi{i}(keep,:,:);
%     cvTrainASD{i}=nClass(cvTrainASD{i}(keep)+1);
%     keep = find(not(ismember(cvTestASD{i},rmClass)));
%     ctestPhi{i}=ctestPhi{i}(keep,:,:);
%     cvTestASD{i}=nClass(cvTestASD{i}(keep)+1);
    
end

phi = cphi;
testPhi = ctestPhi;

save('cancerHBTuckerCVDataGenGV.mat','phi', 'testPhi', 'cvTestASD','cvTrainASD');
