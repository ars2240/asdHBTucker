try
    asdSparse=csvread('cancerSparse.csv',1,1);
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
    tops=[10,25,50,100,150,200,250]; %number of CP topics
    
    for n=tops
        fprintf('%6i Topics\n',n);
        for f=1:nFolds
            b=cvInd==f; %logical indices of test fold
            ind=find(~b);
            fprintf('Fold # %6i\n',f);
            [P,~,info]=cp_als(asd(ind,:,:),n,'printitn',0,'tol',1e-6,...
                'maxiters',100,'init','nvecs'); % train
            phi=P.U{1};

            U=P.U;
            ind=find(b);
            U{1}=repelem(mean(U{1}), length(ind),1);
            Pt=cp_als(asd(ind,:,:),n,'init',U,'printitn',0,'tol',1e-6,...
                'maxiters',100); % test
            testPhi=Pt.U{1};

            %save data
            save(['data/cancerCP_', int2str(n), '_', int2str(f), '_nvecs.mat'], ...
                'phi', 'testPhi', 'P', 'Pt', 'info');
        end
    end
catch e
    display(e.identifier);
    display(e.message);
    for i=1:size(e.stack,1)
        display(e.stack(i,1));
    end
end