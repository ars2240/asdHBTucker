try
    load('data/cancerHBTuckerGenData.mat');
    
    oPsi=psi;  %compute product
    
    iters=10000;  %number of Metropolis?Hastings iterations

    options=init_options();
    % mex drawZscPar.c CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp";
    options.gam = 1;
    options.L = 3;
    % options.topicModel = 'PAM';
    % options.par = 0;
    % options.maxIter = 10;
    % options.collapsed = 0;

    %load data
    load(['data/cancerHBTuckerGen_L', int2str(options.L), '_gam', ...
        num2str(options.gam), '_trees.mat']);
    
    rotation = cell(2,1);
    translation = cell(2,1);
    lrms = zeros(2,1);
    n = zeros(2,1);
    
    for i=1:2
        oPsiG = size(oPsi{i},2);
        psiG = size(psi{i},2);
        if oPsiG>psiG
            psiT = psi{i};
            pcs = pca(oPsi{i});
            oPsiT = oPsi{i}*pcs(:,1:psiG);
        elseif oPsiG<psiG
            pcs = pca(psi{i});
            psiT = psi{i}*pcs(:,1:oPsiG);
            oPsiT = oPsi{i};
        else
            psiT = psi{i};
            oPsiT = oPsi{i};
        end
        psiT=psiT';
        oPsiT=oPsiT';
        [rotation{i}, translation{i}, lrms(i)] = Kabsch(psiT, oPsiT);
        n(i)=norm(rotation{i}*psiT-oPsiT-translation{i});
    end

    disp(n);
    disp(lrms);
    save('data/cancerHBTuckerCVGen_tuckerPCA_rot.mat','rotation', ...
        'translation', 'lrms');

catch e
    display(e.identifier);
    display(e.message);
    for i=1:size(e.stack,1)
        display(e.stack(i,1));
    end
end