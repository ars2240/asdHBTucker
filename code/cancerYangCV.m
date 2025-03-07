try
    asdSparse=csvread('cancerSparse.csv',1,1);
    asd=sparse(asdSparse(:,1),asdSparse(:,2),ones(size(asdSparse,1),1));
    asd1=int8(full(asd)>0);
    asd=sparse(asdSparse(:,1),asdSparse(:,3),ones(size(asdSparse,1),1));
    asd2=int8(full(asd)>0);
    x=[asd1,asd2]+1;
    y=csvread('cancerLabel.csv',1,1); %cancer class

    pTest=.3; %percent of data in test
    rng(12345); %seed RNG
    nPat=size(asd,1); %number of patients
    ind=crossvalind('HoldOut',nPat,pTest); %split data into test & train sets

    %split data based on index into training and testing sets
    x=x(ind,:);
    y=y(ind);

    nFolds=10; %set number of folds
    nTrain=sum(ind); %size of training set
    cvInd=crossvalind('Kfold',nTrain,nFolds); %split data into k folds

    for f=1:nFolds
        b=cvInd==f; %logical indices of test fold
        fprintf('Fold # %6i\n',f);
        [phi,ypred] = yang(x, y, b);
        
        %save data
        save(['data/cancerYangCV_', int2str(f), '_PAM.mat'], 'phi',...
            'ypred');
    end
catch e
    display(e.identifier);
    display(e.message);
    for i=1:size(e.stack,1)
        display(e.stack(i,1));
    end
end
