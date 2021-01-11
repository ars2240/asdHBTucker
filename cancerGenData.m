% try
    asdSparse=csvread('cancerSparseND4.csv',1,1);
    asd=sptensor(asdSparse(:,1:3),asdSparse(:,4));

    options=init_options();
    % mex drawZscPar.c CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp";
    % tpl=10; % topics per level
    options.gam = .1;
    options.L = 3;
    options.topicModel = 'IndepTrees';
    options.par = 0;
    options.maxIter = 1000;
    % options.btReps = 1;
    % options.map = 0;
    % options.topicsPerLevel{1}=tpl;
    % options.topicsPerLevel{2}=tpl;
    % options.collapsed = 0;
    % options.print = 1;
    % options.sparse = 0;
    
    disp(options); %print options

    L=options.L;

    %adjustment if using constant L across dims
    if length(L)==1
        L=repelem(L,2);
    end
    
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

    [phi, psi, tree, samples, paths, ~,~] = asdHBTucker3(asd,options);
    
    %save data
    if length(options.L)==1
        Ls = int2str(options.L);
    else
        Ls = mat2str(options.L);
    end
    save(['data/cancerHBTuckerGenData_L', Ls,...
        '_', options.topicModel, '_4.mat'],'phi', 'psi', 'tree', ...
            'samples', 'paths', 'options');
        
    disp(mean(phi(sum(sum(phi,3),2)>0,1,1)));
    
    options.npats=size(asd,1); %number of articificial patients
    % options.npats=50000; %number of articificial patients
        
    sparse=generatePatients(asd, 1/prod(L), psi, paths, tree, samples, options);

    save(['cancerHBTuckerGenData_L', Ls,...
        '_', options.topicModel, '_4.mat'],'sparse','options');
% catch e
%     display(e.identifier);
%     display(e.message);
%     for i=1:size(e.stack,1)
%         display(e.stack(i,1));
%     end
% end