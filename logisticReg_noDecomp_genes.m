asdSparse=csvread('asdSparseGenes.csv',1,1);
asdSparse(:,4)=ones(size(asdSparse,1),1);

%run only once, keep constant
%or use seed
t=.01;
test='logistic regression'; % 'logistic regression' or 'exact
pTest=.3; %percent of data in test
rng(12345); %seed RNG
nPat=max(asdSparse(:,1)); %number of patients
ind=crossvalind('HoldOut',nPat,pTest); %split data into test & train sets
%save('cvInd.mat','ind'); %save indices
%load('cvInd.mat'); %load indices

asd=logical(repmat([1;0],nPat/2,1)); %binary whether or not patient has ASD

%split data based on index into training and testing sets
trainASD=asd(ind);
trainASDsp = asdSparse(ind(asdSparse(:,1)),:);

nFolds=10; %set number of folds
nTrain=sum(ind); %size of training set
cvInd=crossvalind('Kfold',nTrain,nFolds); %split data into k folds
AUC=zeros(nFolds,1); %initialize AUC vector
AUCtr=zeros(nFolds,1); %initialize AUC vector

cvTrainPhi=cell(nFolds,1);
cvTestPhi=cell(nFolds,1);

%disable certain warnings
warning off stats:glmfit:IterationLimit;
warning off stats:glmfit:IllConditioned;
warning off stats:glmfit:PerfectSeparation;
warning off stats:LinearModel:RankDefDesignMat;
warning off MATLAB:nearlySingularMatrix;

for i=1:nFolds
    b=cvInd==i; %logical indices of test fold
    
    %split data based on index into training and testing sets
    [cvTrainPhi{i}, cvTestPhi{i}]=asdGeneSelectCV2(trainASDsp, trainASD(~b), t, ...
        ind, b, test);
    cvTestASD=trainASD(b,:);
    cvTrainASD=trainASD(~b,:);
    
    %logistic regression
    logReg=glmfit(cvTrainPhi{i},cvTrainASD,'binomial');
    
    %prediction
    predtr=glmval(logReg,cvTrainPhi{i},'logit');
    pred=glmval(logReg,cvTestPhi{i},'logit');
    
    %compute AUC of ROC curve
    [~,~,~,AUCtr(i)]=perfcurve(cvTrainASD,predtr,1);
    [~,~,~,AUC(i)]=perfcurve(cvTestASD,pred,1);
end

%re-enable certain warnings
warning on stats:glmfit:IterationLimit;
warning on stats:glmfit:IllConditioned;
warning on stats:glmfit:PerfectSeparation;
warning on stats:LinearModel:RankDefDesignMat;
warning on MATLAB:nearlySingularMatrix;

%t-test that mean AUC = 0.5
[~,p]=ttest(AUC,.5);
[~,ptr]=ttest(AUCtr,.5);

%print values
fprintf('Set\t Mean\t StDev\t P-value\n');
fprintf('Valid\t %1.4f\t %1.4f\t %1.4f\n',mean(AUC),std(AUC),p);
fprintf('Train\t %1.4f\t %1.4f\t %1.4f\n',mean(AUCtr),std(AUCtr),ptr);

save('asdNoDecomp_1s.mat','cvTrainPhi','cvTestPhi');