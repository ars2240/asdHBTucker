%load('cancerHBTuckerCV.mat'); %load tensor

asdSparse=csvread('cancerSparse.csv',1,1);
asd=sptensor(asdSparse(:,1:3),asdSparse(:,4));
%asd=sptensor(asdSparse(:,1:3),ones(size(asdSparse,1),1));

pTest=.3; %percent of data in test
rng(12345); %seed RNG
nPat=size(asd,1); %number of patients
ind=crossvalind('HoldOut',nPat,pTest); %split data into test & train sets

%split data based on index into training and testing sets
asd=asd(find(ind),:,:);

nFolds=10; %set number of folds
nTrain=sum(ind); %size of training set
cvInd=crossvalind('Kfold',nTrain,nFolds); %split data into k folds

npats=1000; %number of artificial patients
LL=zeros(nFolds,1); %initialize log-likelihood

L=options.L;
    
%adjustment if using constant L across dims
if length(L)==1
    L=repelem(L,2);
end

for f=1:nFolds
    b=cvInd==f; %logical indices of test fold
    
    r=cell(2,1);
    r{1}=unique(paths{1}(:,1:L(1)));
    r{2}=unique(paths{1}(:,(L(1)+1):(sum(L))));
    
    LL(f)=logLikelihood(asd(find(~b),:,:), asd(find(b),:,:), npats, 1, ...
        1/(size(asd,2)*size(asd,3)), psi{f}, r, paths{f}, tree{f}, options);
end

output_header=sprintf('%13s %13s','mean','stDev');
fprintf('%s\n',output_header);
fprintf('%13.4e %13.4e\n', mean(LL), std(LL));
