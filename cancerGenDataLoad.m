try
    asdSparse=csvread('cancerSparse.csv',1,1);
    asd=sptensor(asdSparse(:,1:3),asdSparse(:,4));

    options=init_options();
    options.gam = 1;
    options.L = 3;
    options.topicModel = 'IndepTrees';
    % options.par = 0;
    % options.maxIter = 10;
    % options.collapsed = 0;
    npats=size(asd,1); %number of articificial patients
    
    disp(options); %print options

    L=options.L;

    %adjustment if using constant L across dims
    if length(L)==1
        L=repelem(L,2);
    end
    
    %save data
    load('data/cancerHBTuckerGenData.mat');
        
    r=cell(2,1);
    r{1}=unique(paths(:,1:L(1)));
    r{2}=unique(paths(:,(L(1)+1):(sum(L))));
        
    sparse=generatePatients(asd, npats, 1, psi, r, paths, tree, options);

    save('cancerHBTuckerGenData2.mat','sparse');
catch e
    display(e.identifier);
    display(e.message);
    for i=1:size(e.stack,1)
        display(e.stack(i,1));
    end
end