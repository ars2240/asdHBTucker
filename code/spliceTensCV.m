%try
    load('splice.mat');
    asd=sptensor(asdSparse,ones(size(asdSparse,1),1));
    %asd=sptensor(asdSparse(:,1:3),ones(size(asdSparse,1),1));

    pTest=.3; %percent of data in test
    rng(12345); %seed RNG
    nPat=size(asd,1); %number of patients
    ind=crossvalind('HoldOut',nPat,pTest); %split data into test & train sets

    %split data based on index into training and testing sets
    asdSparse=asdSparse(ismember(asdSparse(:,1),find(ind)),:);
    [~,~,asdSparse(:,1)]=unique(asdSparse(:,1));
    asd=sptensor(asdSparse,ones(size(asdSparse,1),1));

    nFolds=10; %set number of folds
    nTrain=sum(ind); %size of training set
    cvInd=crossvalind('Kfold',nTrain,nFolds); %split data into k folds

    options=init_options();
    %mex drawZscPar.c CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp";
    %mex drawZsCollapsedPar.c CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp";
    %mex drawZscSparsePar.c CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp";
    tpl=10; % topics per level
    options.gam = 1;
    options.L = 2;
    % options.topicModel = 'PAM';
    options.par = 0;
    options.maxIter = 1000;
    options.topicsPerLevel{1}=tpl;
    options.topicsPerLevel{2}=tpl;
    % options.collapsed = 0;
    npats=1000; %number of articificial patients
    
    disp(options); %print options
    
    LL=zeros(nFolds,1); %initialize log-likelihood

    L=options.L;

    %adjustment if using constant L across dims
    if length(L)==1
        L=repelem(L,2);
    end

    for f=1:nFolds
        b=cvInd==f; %logical indices of test fold
        ind=find(~b);
        trSparse=asdSparse(ismember(asdSparse(:,1),find(ind)),:);
        [~,~,trSparse(:,1)]=unique(trSparse(:,1));
        asdTr=sptensor(trSparse,ones(size(trSparse,1),1));
        fprintf('Fold # %6i\n',f);
        [phi, psi, tree, samples, paths,prob, ~,~] = ...
            asdHBTucker3(asdTr,options);
        testPhi = asdHBTuckerNew(asd, psi, samples, paths, tree, prob, ...
            b, options);
        
        %save data
        save(['data/spliceHBTuckerCV_L', int2str(options.L), '_tpl', ...
            num2str(tpl), '_', int2str(f), '_', ...
            options.topicType, '_PAM.mat'],'phi', 'testPhi', ...
            'psi', 'tree', 'samples', 'paths', 'prob', 'options');
    
        r=cell(2,1);
        r{1}=unique(paths(:,1:L(1)));
        r{2}=unique(paths(:,(L(1)+1):(sum(L))));

        %compute LL
        teSparse=asdSparse(~ismember(asdSparse(:,1),find(ind)),:);
        [~,~,teSparse(:,1)]=unique(teSparse(:,1));
        asdTe=sptensor(teSparse,ones(size(teSparse,1),1));
        LL(f)=logLikelihood(asdTr, asdTe, npats, 1, ...
            1/(size(asd,2)*size(asd,3)), psi, r, paths, tree, prob, ...
            samples, options);
    end

    % print LL info
    output_header=sprintf('%13s %13s','mean','stDev');
    fprintf('%s\n',output_header);
    fprintf('%13.6e %13.6e\n', mean(LL), std(LL));
% catch e
%     display(e.identifier);
%     display(e.message);
%     for i=1:size(e.stack,1)
%         display(e.stack(i,1));
%     end
% end
