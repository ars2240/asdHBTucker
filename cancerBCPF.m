try
    asdSparse=csvread('cancerSparseND4.csv',1,1);
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
    save('cancerCVInd','cvInd');
    
    LL=zeros(nFolds,1); %initialize log-likelihood
    
    % remove bad genes
    asdG=collapse(asd,3,@max);
    asdGC=collapse(asdG>0,1);
    gG=find(asdGC>400 & asdGC<1000);
    asd=asd(:,gG,:);
    % remove zero pathways
    asdP=collapse(asd,[1,2]);
    gP=find(asdP>0);
    asd=asd(:,:,gP);
    asdGP=collapse(asd,1);
    [~,gP,~]=unique(double(asdGP)', 'rows');
    asd=asd(:,:,gP);

    model = BCPF(asd);
        
    %save data
    save('data/cancerBCPF.mat','model');
    csvwrite('data/cancerBCPF.csv',model.Z{1});

    % print LL info
    output_header=sprintf('%13s %13s','mean','stDev');
    fprintf('%s\n',output_header);
    fprintf('%13.6e %13.6e\n', mean(LL), std(LL));
catch e
    display(e.identifier);
    display(e.message);
    for i=1:size(e.stack,1)
        display(e.stack(i,1));
    end
end