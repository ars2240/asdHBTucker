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
    gam0 = 0.5;
    options.L = [2, 1];
    % options.topicType = 'Level';
	% options.topicModel = 'None';
    options.par = 0;
    options.maxIter = 20;
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
    options.topicsgoal = 500;
    dom = 'Genes';
    
    disp(options); %print options
    
    LL=zeros(nFolds,1); %initialize log-likelihood
    
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
    
    if strcmp(dom,'Pthwy')
        asd=permute(asd,[1 3 2]);
    end
    asd=collapse(asd,3,@max);
    asd = sptensor([asd.subs,ones(size(asd.subs,1),1)], asd.vals, [asd.size, 1]);
    
    % multiply by factor
    %asd=asd*10;

    for f=10:nFolds
        b=cvInd==f; %logical indices of test fold
        ind=find(~b);
        fprintf('Fold # %6i\n',f);
        KB.LL=-inf;
        for k=1:nBest
            options.gam=gam0;
            [~, ~, ~, ~, ~, options, ll, ~] = ...
                asdHBTucker3(asd(ind,:,:),options);
            %fprintf('%13.6e, %13.6e\n',ll, options.gam(1));
            fprintf('%13.6e %2i\n',ll, options.best.iter);
            if ll>KB.LL && ll~=0
                KB = options.best;
            end
        end
        fprintf('Best LL: %13.6e\n',KB.LL);
        phi=KB.phi; psi=KB.psi; tree=KB.tree; samples=KB.samples;
        paths=KB.paths; options.gam=KB.gamma;
        testPhi = asdHBTuckerNew(asd, psi, samples, paths, tree, ...
            b, options);
        
        %save data
        save(['data/cancerHBTCV3KB', int2str(nBest), '_L',...
            int2str(options.L(1)), '_tpl', int2str(tpl),'_', ...
            int2str(f), '_hLDA.mat'],'phi', 'testPhi', 'psi', ...
            'tree', 'samples', 'paths', 'options');

        %compute LL
        %LL(f)=logLikelihood(asd(find(~b),:,:), asd(find(b),:,:), ...
        %    1, 1/(size(asd,2)*size(asd,3)), psi, paths, tree, samples, ...
        %    options);
    end

    % print LL info
    %output_header=sprintf('%13s %13s','mean','stDev');
    %fprintf('%s\n',output_header);
    %fprintf('%13.6e %13.6e\n', mean(LL), std(LL));
% catch e
%     display(e.identifier);
%     display(e.message);
%     for i=1:size(e.stack,1)
%         display(e.stack(i,1));
%     end
% end