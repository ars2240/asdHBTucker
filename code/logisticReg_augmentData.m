load('asdNoDecomp.mat');
load('asdHBTuckerCVData.mat');

nFolds=size(phi,1);

AUC=zeros(nFolds,1); %initialize AUC vector
AUCtr=zeros(nFolds,1); %initialize AUC vector


%disable certain warnings
warning off stats:glmfit:IterationLimit;
warning off stats:glmfit:IllConditioned;
warning off stats:glmfit:PerfectSeparation;
warning off stats:LinearModel:RankDefDesignMat;
warning off MATLAB:nearlySingularMatrix;

for i=1:nFolds
    phi{i}=double(tenmat(phi{i},1));
    testPhi{i}=double(tenmat(testPhi{i},1));
    phi{i}=[phi{i} cvTrainPhi{i}];
    testPhi{i}=[testPhi{i} cvTestPhi{i}];
    
    %logistic regression
    logReg=glmfit(phi{i},cvTrainASD{i},'binomial');
    
    %prediction
    predtr=glmval(logReg,phi{i},'logit');
    pred=glmval(logReg,testPhi{i},'logit');
    
    %compute AUC of ROC curve
    [~,~,~,AUCtr(i)]=perfcurve(cvTrainASD{i},predtr,1);
    [~,~,~,AUC(i)]=perfcurve(cvTestASD{i},pred,1);
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
