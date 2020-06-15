asdSparse=csvread('cancerSparseND4.csv',1,1);
asdTens=sptensor(asdSparse(:,1:3),asdSparse(:,4));

load('cancerCVInd.mat'); %load indices

%phiMat=zscore(phiMat); %normalize
asd=csvread('cancerLabel.csv',1,1); %cancer class

%split data based on index into training and testing sets
asdTens=asdTens(find(ind~=0),:,:);

cvInd=ind(ind~=0);

options=init_options();
% tpl=10; % topics per level
options.gam = .1;
options.L = 2;
% options.topicModel = 'PAM';
% options.par = 0;
options.maxIter = 1000;
% options.topicsPerLevel{1}=tpl;
% options.topicsPerLevel{2}=tpl;
% options.collapsed = 0;

nFolds=10; %set number of folds

normTr=zeros(nFolds,1);
normVa=zeros(nFolds,1);

for f=1:nFolds
    b=cvInd==f; %logical indices of test fold

    phiO=asdTens(find(~b),:,:);
    testPhiO=asdTens(find(b),:,:);

    load(['data/cancerHBTuckerCVND_L', int2str(options.L), '_tpl', ...
        num2str(options.gam), '_', int2str(f), '_', ...
        options.topicType, '_Trees.mat']);

    p = 0;
    for i=1:sum(~b)
        p=p+sum(sum(double((phiO(i,:,:)-...
            full(ttensor(phi(i,:,:),psi))).^2)));
    end
    normTr(f)=sqrt(p);
    
    p = 0;
    for i=1:sum(b)
        p=p+sum(sum(double((testPhiO(i,:,:)-...
            full(ttensor(testPhi(i,:,:),psi))).^2)));
    end
    normVa(f)=sqrt(p);

end

fprintf('Train Norm= %5.4f\n',mean(normTr));
fprintf('Valid Norm= %5.4f\n',mean(normVa));