try
    load('cancerHBTuckerGenData.mat');
    asd=sptensor(sparse(:,1:3),sparse(:,4));
    %asd=sptensor(asdSparse(:,1:3),ones(size(asdSparse,1),1));

    options=init_options();
    % mex drawZscPar.c CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp";
    tpl=10; % topics per level
    options.gam = 1;
    options.L = 3;
    % options.topicModel = 'PAM';
    % options.par = 0;
    % options.maxIter = 10;
    options.topicsPerLevel{1}=tpl;
    options.topicsPerLevel{2}=tpl;
    % options.collapsed = 0;
    npats=1000; %number of articificial patients
    
    disp(options); %print options

    [phi, psi, tree, samples, paths, ~,~] = ...
        asdHBTucker3(asd,options);

    %save data
    save(['data/cancerHBTuckerGen_L', int2str(options.L), '_gam', ...
        num2str(options.gam), '_trees.mat'],'phi', ...
        'psi', 'tree', 'samples', 'paths', 'options');
    
    disp(psi);

catch e
    display(e.identifier);
    display(e.message);
    for i=1:size(e.stack,1)
        display(e.stack(i,1));
    end
end