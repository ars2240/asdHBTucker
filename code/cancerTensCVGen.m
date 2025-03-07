% try
    load('cancerHBTuckerGenData_L3_IndepTrees.mat');
    asd=sptensor(sparse(:,1:3),sparse(:,4));
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

    options=init_options();
    % mex drawZscPar.c CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp";
    % tpl=10; % topics per level
    options.gam = .1;
    options.L = 3;
	% options.topicModel = 'None';
    options.par = 0;
    options.maxIter = 1000;
    % options.treeReps = 5;
    % options.btReps = 5;
    % options.topicsPerLevel{1}=tpl;
    % options.topicsPerLevel{2}=tpl;
    % options.collapsed = 0;
    % options.print = 0;
    % options.sparse = 0;
    
    disp(options); %print options
    
    LL=zeros(nFolds,1); %initialize log-likelihood

    L=options.L;

    %adjustment if using constant L across dims
    if length(L)==1
        L=repelem(L,2);
    end
    
    disp(sum(asd.values));
    
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

    for f=1:nFolds
        b=cvInd==f; %logical indices of test fold
        ind=find(~b);
        fprintf('Fold # %6i\n',f);
        [phi, psi, tree, samples, paths, ll,~] = ...
            asdHBTucker3(asd(ind,:,:),options);
        testPhi = asdHBTuckerNew(asd, psi, samples, paths, tree, ...
            b, options);
        
        %save data
        save(['data/cancerHBTuckerCVGen5_L', int2str(options.L), '_', ...
            int2str(f), '_',  options.topicModel, '.mat'],'phi', ...
            'testPhi', 'psi', 'tree', 'samples', 'paths', 'options');

        %compute LL
        LL(f)=ll;
%         LL(f)=logLikelihood(asd(find(~b),:,:), asd(find(b),:,:), ...
%             1, 1/(size(asd,2)*size(asd,3)), psi, paths, tree, options);
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
