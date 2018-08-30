asdSparse=csvread('asdSparse.csv',1,1);
%asd=sptensor(asdSparse(:,1:3),asdSparse(:,4));
asd=sptensor(asdSparse(:,1:3),ones(size(asdSparse,1),1));

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
% options.maxIter = 100;
options.freq = 10;
% options.treeReps = 1;
% options.btReps = 1;
% options.topicModel = 'PAM';
% options.par = 0;
% options.collapsed = 1;

phi=cell(nFolds,1);
testPhi=cell(nFolds,1);
psi=cell(nFolds,1);
tree=cell(nFolds,1);
samples=cell(nFolds,1);
paths=cell(nFolds,1);
for f=1:nFolds
    b=cvInd==f; %logical indices of test fold
    ind=find(~b);
    fprintf('Fold # %6i\n',f);
    [phi{f}, psi{f}, tree{f}, samples{f}, paths{f}]=asdHBTucker3(asd(ind,:,:),options);
    testPhi{f} = asdHBTuckerNew(asd, psi{f}, samples{f}, paths{f}, tree{f}, b, options);
end
save('asdHBTuckerCV.mat','phi', 'testPhi', 'psi', 'tree', 'samples', 'paths', 'options');