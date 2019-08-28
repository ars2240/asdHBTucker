try
    load('data/cancerHBTuckerGenData.mat');
    
    oTuck=ttensor(phi,{speye(size(phi,1)),psi{1},psi{2}});  %compute product
    
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
    
    tuck=ttensor(phi,{speye(size(phi,1)),psi{1},psi{2}});  %compute product

    [diff, order] = tuckerMH(oTuck, tuck, iters);
    disp(diff);
    save('data/cancerHBTuckerCVGen_tuckerMH_order.mat','order');

catch e
    display(e.identifier);
    display(e.message);
    for i=1:size(e.stack,1)
        display(e.stack(i,1));
    end
end