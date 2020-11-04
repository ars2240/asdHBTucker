% asdSparse=csvread('cancerSparseND4.csv',1,1);
% asdTens=sptensor(asdSparse(:,1:3),asdSparse(:,4));
load('cancerHBTuckerGenData_L3_IndepTrees_KB.mat');
asd=sptensor(sparse(:,1:3),sparse(:,4));

ind=csvread('cancerGenCVInd.csv');

%phiMat=zscore(phiMat); %normalize

%split data based on index into training and testing sets
asd=asd(find(ind~=0),:,:);

cvInd=ind(ind~=0);

options=init_options();
% tpl=10; % topics per level
options.gam = .1;
options.L = [5, 6];
options.topicModel = 'None';
% options.par = 0;
options.maxIter = 1000;
% options.topicsPerLevel{1}=tpl;
% options.topicsPerLevel{2}=tpl;
% options.collapsed = 0;

% asdG=collapse(asd,3,@max);
% asdGC=collapse(asdG>0,1);
% gG=find(asdGC>400 & asdGC<1000);
% asd=asd(:,gG,:);
% % remove zero pathways
% asdP=collapse(asd,[1,2]);
% gP=find(asdP>0);
% asd=asd(:,:,gP);
% asdGP=collapse(asd,1);
% [~,gP,~]=unique(double(asdGP)', 'rows');
% asd=asd(:,:,gP);

nFolds=10; %set number of folds

normTr=zeros(nFolds,1);
normVa=zeros(nFolds,1);

for f=1:nFolds
    b=cvInd==f; %logical indices of test fold

    phiO=asd(find(~b),:,:);
    testPhiO=asd(find(b),:,:);

    load(['data/cancerHBTuckerCVGenKB2_L', int2str(options.L), '_', ...
            int2str(f), '_',  options.topicModel, '.mat']);

    p = 0;
    for i=1:sum(~b)
        t=squeeze(phi(i,:,:))'*psi{1}'; t=t'*psi{2}';
        t0=full(squeeze(phiO(i,:,:)));
        p=p+sum(sum(double((t0-t).^2)));
    end
    normTr(f)=sqrt(p);
    
    p = 0;
    for i=1:sum(b)
        t=squeeze(testPhi(i,:,:))'*psi{1}'; t=t'*psi{2}';
        t0=full(squeeze(testPhiO(i,:,:)));
        p=p+sum(sum(double((t0-t).^2)));
    end
    normVa(f)=sqrt(p);

end

fprintf('Train Norm= %5.4f\n',mean(normTr));
fprintf('Valid Norm= %5.4f\n',mean(normVa));