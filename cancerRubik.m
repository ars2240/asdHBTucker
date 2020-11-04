% try
    %load('cancerHBTuckerGenData_L3_IndepTrees_KB.mat');
    asdSparse=csvread('cancerSparseND4.csv',1,1);
    asd=sptensor(sparse(:,1:3),sparse(:,4));
    %asd=sptensor(asdSparse(:,1:3),ones(size(asdSparse,1),1));

    K=5; % topics
    
    pTest=.3; %percent of data in test
    rng(12345); %seed RNG
    nPat=size(asd,1); %number of patients
    ind=crossvalind('HoldOut',nPat,pTest); %split data into test & train sets

    %split data based on index into training and testing sets
    asd=asd(find(ind),:,:);

    nFolds=10; %set number of folds
    nBest=10;
    nTrain=sum(ind); %size of training set
    cvInd=crossvalind('Kfold',nTrain,nFolds); %split data into k folds
    
%     % remove bad genes
%     asdG=collapse(asd,3,@max);
%     asdGC=collapse(asdG>0,1);
%     gG=find(asdGC>400 & asdGC<1000);
%     asd=asd(:,gG,:);
%     % remove zero pathways
%     asdP=collapse(asd,[1,2]);
%     gP=find(asdP>0);
%     asd=asd(:,:,gP);
%     asdGP=collapse(asd,1);
%     [~,gP,~]=unique(double(asdGP)', 'rows');
%     asd=asd(:,:,gP);

    T = rubik_simple(asd,K,zeros(size(asd,2),1));
    lambda=T.lambda; phi=T.U{1};
    psi=cell(2,1); psi{1}=T.U{2}; psi{2}=T.U{3};

    %save data
    save(['data/cancerRubik_K', int2str(K), '.mat'],'phi','psi','lambda');
    csvwrite(['data/cancerRubik_K', int2str(K), '.csv'],phi);
        
% catch e
%     display(e.identifier);
%     display(e.message);
%     for i=1:size(e.stack,1)
%         display(e.stack(i,1));
%     end
% end