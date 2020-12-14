% try
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
    nBest=10;
    nTrain=sum(ind); %size of training set
    cvInd=crossvalind('Kfold',nTrain,nFolds); %split data into k folds

    options=init_options();
    % mex drawZscPar.c CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp";
    tpl=10; % topics per level
    gam0 = 1;
    options.L = 2;
    %options.topicType = 'Level';
	%options.topicModel = 'PAM';
    options.par = 0;
    options.maxIter = 100;
    options.pType = 0;
    % options.treeReps = 5;
    % options.btReps = 5;
    options.topicsPerLevel{1}=tpl;
    options.topicsPerLevel{2}=tpl;
    % options.collapsed = 0;
    options.keepBest = 1;
    options.time = 0;
    options.print = 1;
    % options.cutoff = 0.1;
    % options.sparse = 0;
    options.topicsgoal = 200;
    
    disp(options); %print options
    
    LL=zeros(nFolds,1); %initialize log-likelihood

    L=options.L;

    %adjustment if using constant L across dims
    if length(L)==1
        L=repelem(L,2);
    end
    
    % remove bad genes
    asdG=collapse(asd,3,@max);
    asdGC=collapse(asdG>0,1);
    %asdGC=collapse(asdG,1);
    gG=find(asdGC>200 & asdGC<2000);
    asd=asd(:,gG,:);
    % remove zero pathways
    asdP=collapse(asd,[1,2]);
    gP=find(asdP>0);
    asd=asd(:,:,gP);
    asdGP=collapse(asd,1);
    [~,gP,~]=unique(double(asdGP)', 'rows');
    asd=asd(:,:,gP);
    
    % multiply by factor
    %asd=asd*10;

    for f=1:nFolds
        b=cvInd==f; %logical indices of test fold
        ind=find(~b);
        fprintf('Fold # %6i\n',f);
        KB.LL=-inf;
        options.gam=gam0;
        for k=1:nBest
            [~, ~, ~, ~, ~, options, ll,~] = ...
                asdHBTucker3(asd(ind,:,:),options);
            fprintf('%13.6e, %13.6e\n',ll, options.gam);
            if ll>KB.LL && ll~=0
                KB = options.best;
            end
        end
        fprintf('Best LL: %13.6e\n',KB.ll);
        phi=KB.phi; psi=KB.psi; tree=KB.tree; samples=KB.samples;
        paths=KB.paths; options.gam=KB.gamma;
        testPhi = asdHBTuckerNew(asd, psi, samples, paths, tree, ...
            b, options);
        
        %save data
        save(['data/cancerHBTCV3KB', int2str(nBest), '_L',...
            int2str(options.L), '_gamVar_', ...
            int2str(f), '_', options.topicType, '_', ...
            options.topicModel, '.mat'],'phi', 'testPhi', 'psi', ...
            'tree', 'samples', 'paths', 'options');
    
        r=cell(2,1);
        r{1}=unique(paths(:,1:L(1)));
        r{2}=unique(paths(:,(L(1)+1):(sum(L))));

        %compute LL
        LL(f)=logLikelihood(asd(find(~b),:,:), asd(find(b),:,:), ...
            1, 1/(size(asd,2)*size(asd,3)), psi, paths, tree, samples, ...
            options);
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