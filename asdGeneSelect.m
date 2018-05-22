function x = asdGeneSelect(xSparse, t)
    %performs logistic regression on genes to reduce tensor size
    %xSparse = sparse matrix to be turned into tensor
    %t = p-value threshold
    
    x=sptensor(xSparse(:,1:3),xSparse(:,4));
    
    xM=sptensor(xSparse(:,1:2),xSparse(:,4),max(xSparse(:,1:2)),@max);
    xM=full(tenmat(xM,1));
    xM=xM(:,:);
    
    nGenes = size(xM,2);
    
    pTest=.3; %percent of data in test
    rng(12345); %seed RNG
    nPat=size(x,1); %number of patients
    ind=crossvalind('HoldOut',nPat,pTest); %split data into test & train sets

    xM=xM(:,sum(xM,1)>0); %remove columns of all zeros
    asd=logical(repmat([1;0],nPat/2,1)); %binary whether or not patient has ASD

    %split data based on index into training and testing sets
    trainX=xM(ind,:);
    trainASD=asd(ind);

    pval=ones(1,nGenes);
    
    %disable certain warnings
    warning off stats:glmfit:IterationLimit;
    warning off stats:glmfit:IllConditioned;
    warning off stats:glmfit:PerfectSeparation;
    warning off MATLAB:nearlySingularMatrix;

    for i=1:nGenes
        %logistic regression
        [~,~,stats]=glmfit(trainX(:,i),trainASD,'binomial');
        
        pval(i)=stats.p(2);
    end

    %re-enable certain warnings
    warning on stats:glmfit:IterationLimit;
    warning on stats:glmfit:IllConditioned;
    warning on stats:glmfit:PerfectSeparation;
    warning on MATLAB:nearlySingularMatrix;
    
    ind = pval<t;
    fprintf('Threashold= %5.4f\n',t);
    fprintf('Number of Genes in Threshold= %5d\n',sum(ind));
    
    %subset x by genes in threashold
    x = x(:,find(ind),:);
    
    %remove pathways with 0 count
    xM = tenmat(x, 3);
    xM = xM(:,:);
    pCount=sum(xM,2);
    ind = pCount>0;
    fprintf('Number of Pathways= %5d\n',sum(ind));
    x = x(:,:,find(ind));
    
end