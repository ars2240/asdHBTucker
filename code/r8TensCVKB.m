% try
    asdSparse=csvread('r8p_sparse.csv');
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
    % options.weights = [.25, 1];
    options.topicsgoal = 50;
    options.save = 0;
    % dom = 'Genes';
    dom = 'Pwy';
    options.coh.measure = 'umass';
    % tail = '_weighted.25x_1x_t50_cohmass';
    tail = '_cohmass';
    
    if strcmp(options.topicType,'CP')
        options.topicsgoal = sqrt(options.topicsgoal);
        options.maxTop = 50;
    end
    
    disp(options); %print options
    
    LL=zeros(nFolds,1); %initialize log-likelihood
    
    % remove bad genes
    asdP=collapse(asd,[1,2],@sum);
    gP=find(asdP>200 & asdP<2000);
    asd=asd(:,:,gP);

    %subset phrases by reverting to "none"
    asdG=collapse(asd,3,@max);
    asdGC=collapse(asdG>0,1);
    s = asd.subs;
    noneP=mode(asd.subs(:,2));
    gG=find(asdGC<=10);
    rows = ismember(s(:,2), gG);
    s(:,2) = s(:,2) + size(asd,3);
    s(rows,2) = s(rows,3);
    rows = find(s(:,2) == noneP + size(asd,3));
    s(rows,2) = s(rows,3);
    %s(:,2) = 1;
    asd=sptensor(s,asd.vals);

    asdG=collapse(asd,3,@max);
    asdGC=collapse(asdG>0,1);
    %asdGC=collapse(asdG,1);
    gG=find(asdGC>0);
    if length(gG) < size(asd,2)
        asd=asd(:,gG,:);
    end

    % remove zero pathways
    % asdGP=collapse(asd,1);
    % [~,gP,~]=unique(double(asdGP)', 'rows');
    % asd=asd(:,:,gP);

    % bad=[0, 29, 64, 81, 180, 186, 194, 224, 234, 242, 244, 245, 246, 247] + 1;
    bad = csvread('r8_badP.csv') + 1;
    good=setdiff(1:size(asd,3),bad);
    asd=asd(:,:,good);
    good=setdiff(1:size(asd,2),bad);
    asd=asd(:,good,:);

    % csvwrite('r8p_sparse3.csv',[asd.subs, asd.vals]);
    
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
                save(['data/r8p2HBTCV3KB', int2str(nBest), '_L',...
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
                save(['data/r8p2HBTCV3KB', int2str(nBest), '_L',...
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
