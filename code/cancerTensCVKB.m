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
    gam0 = 0.1;
    options.L = [2 1];
    % options.topicType = 'CP';
	% options.topicModel = 'PAM';
    options.par = 0;
    options.maxIter = 100;
    options.pType = 0;
    % options.treeReps = 5;
    % options.btReps = 5;
    options.topicsPerLevel{1}=tpl;
    options.topicsPerLevel{2}=tpl;
    % options.collapsed = 0;
    options.keepBest = 3;
    options.time = 1;
    options.print = 1;
    % options.cutoff = 0.1;
    % options.sparse = 0;
    % options.weights = [48, 48];
    options.topicsgoal = 500;
    options.save = 0;
    % dom = 'Genes';
    dom = 'Pwy';
    options.coh.measure = 'umass';
    % tail = '_weighted48x_48x_cohmass';
    tail = '_cohmass';
    
    if strcmp(options.topicType,'CP')
        options.topicsgoal = sqrt(options.topicsgoal);
        options.maxTop = 50;
    end
    
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
    
    % for hLDA
    ind = asd.subs; ind(:,2)=1;
    v = asd.values;
    asd = sptensor(ind,v,max(ind),@max);
    
    if strcmp(dom,'Pwy')
        asd=permute(asd,[1 3 2]);
    end
    asd=collapse(asd,3,@max);
    asd=sptensor([asd.subs,ones(size(asd.subs,1),1)], asd.vals, [asd.size, 1]);
    
    % multiply by factor
    %asd=asd*10;

    for f=1:1%:nFolds
        b=cvInd==f; %logical indices of test fold
        ind=find(~b);
        fprintf('Fold # %6i\n',f);
        KB.cm=-inf;
        for k=1:nBest
            options.gam=gam0;
            if strcmp(options.topicModel,'PAM')
                [~, ~, ~, ~, ~, prob, options, ~, ~] = ...
                asdHBTucker3(asd(ind,:,:),options);
            else
                [~, ~, ~, ~, ~, options, ~, ~] = ...
                asdHBTucker3(asd(ind,:,:),options);
            end
            cm = options.best.cm;
            %fprintf('%13.6e, %13.6e\n',ll, options.gam(1));
            fprintf('%13.6e %2i\n',cm, options.best.iter);
            if cm>KB.cm && cm~=0
                KB = options.best;
                if strcmp(options.topicModel,'PAM')
                    KB.prob = prob;
                end
            end
        end
        fprintf('Best CM: %13.6e\n',KB.cm);
        phi=KB.phi; psi=KB.psi; samples=KB.samples;
        paths=KB.paths; options.gam=KB.gamma;
        if strcmp(options.topicModel,'PAM')
            prob=KB.prob;
            testPhi = asdHBTuckerNew(asd, psi, samples, paths, prob, ...
                b, options);
            if options.save == 1
                save(['data/cancerHBTCV3KB', int2str(nBest), '_L',...
                    int2str(options.L(1)), '_tpl', int2str(tpl), '_', ...
                    int2str(f), '_', options.topicModel, '_', ...
                    options.topicType, '_', dom, tail, '.mat'],'phi', ...
                    'testPhi', 'psi', 'prob', 'samples', 'paths', 'options');
            end
        else
            tree=KB.tree;
            testPhi = asdHBTuckerNew(asd, psi, samples, paths, tree, ...
                b, options);
            %save data
            if options.save == 1
                save(['data/cancerHBTCV3KB', int2str(nBest), '_L',...
                    int2str(options.L(1)), '_tpl', int2str(tpl), '_', ...
                    int2str(f), '_', options.topicModel, '_', ...
                    options.topicType, '_', dom, tail, '.mat'],'phi', ...
                    'testPhi', 'psi', 'tree', 'samples', 'paths', 'options');
            end
        end

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
