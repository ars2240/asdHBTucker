asdSparse=csvread('cancerSparseGenes.csv',1,1);
asd=sptensor([asdSparse(:,1:2),ones(size(asdSparse,1),1)],asdSparse(:,3));
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

options=init_options();
% mex drawZscPar.c CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp";
% options.L = 3;
options.L = [2,1];
options.maxIter = 1000;
options.freq = 10;
% options.treeReps = 1;
% options.btReps = 1;
% options.topicModel = 'PAM';
% options.par = 0;
options.collapsed = 0;

phi=cell(nFolds,1);
testPhi=cell(nFolds,1);
psi=cell(nFolds,1);
tree=cell(nFolds,1);
samples=cell(nFolds,1);
paths=cell(nFolds,1);
LL=zeros(nFolds,1);
ms=zeros(nFolds,1);
for f=1:nFolds
    b=cvInd==f; %logical indices of test fold
    ind=find(~b);
    fprintf('Fold # %6i\n',f);
    [phi{f}, psi{f}, tree{f}, samples{f}, paths{f}, LL(f), ms(f)] = ...
        asdHBTucker3(asd(ind,:,:),options);
    testPhi{f} = asdHBTuckerNew(asd, psi{f}, samples{f}, paths{f}, ...
        tree{f}, b, options);
end

output_header=sprintf('%6s %13s %10s','fold','loglikelihood', ...
        'model size');
fprintf('%s\n',output_header);
for f=1:nFolds
    fprintf('%6i %13.2e %10i\n',...
                    f, LL(f), ms(f));
end
save('cancerHBTuckerCV_HLDA.mat','phi', 'testPhi', 'psi', 'tree', 'samples', 'paths', 'options');
