try
    load('data/cancerHBTuckerGenData.mat');
    
    oPsi=psi;  %store psi
    
    iters=1000;  %number of Metropolis?Hastings iterations

    options=init_options();
    % mex drawZscPar.c CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp";
    options.gam = 1;
    options.L = 3;
    % options.topicModel = 'PAM';
    % options.par = 0;
    % options.maxIter = 10;
    % options.collapsed = 0;

    diff1=zeros(nFolds,1);
    diff2=zeros(nFolds,1);
    order1=cell(nFolds,1);
    order2=cell(nFolds,1);

    for f=1:nFolds
        %load data
        load(['data/cancerHBTuckerCVGen_L', int2str(options.L), '_gam', ...
            num2str(options.gam), '_', int2str(f), '_trees.mat']);
        
        [diff1(f), order1{f}] = psiMH(oPsi{1}, psi{1}, iters);
        [diff2(f), order2{f}] = psiMH(oPsi{12}, psi{12}, iters);
   
    end

    
catch e
    display(e.identifier);
    display(e.message);
    for i=1:size(e.stack,1)
        display(e.stack(i,1));
    end
end