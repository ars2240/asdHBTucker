%try
    rng(123);

    %constants
    p=200; %number of features
    N=4000; %total size
    n=600; %training size
    ep=max(6,floor(log(n)/log(4))); %expected maximum number of predictors
    d=4; %number of categories for each features
    d0=1; %prior for Dirichlet Distribution
    c=0.3;
    pM=[1-3*c*ep/p,c*ep/p,c*ep/p,c*ep/p]; %prior probability for kj 
    np=0; %number of predictors included in the model

    % generate data
    x=zeros(N,p);
    for i=1:p
        x(:,i)=randsample(d,N,true);
    end
    % y=(rand(N,1)<(1./(1+(exp(2*(-2*(x(:,9)==1)+(x(:,9)==2)-0.5*(x(:,9)==5)-...
    %     (x(:,11)==3)+2*(x(:,11)==5)+2*(x(:,13)==1)-...
    %     1.5*(x(:,15)==4)-1.5*(x(:,17)==4)+2*(x(:,19)==3)))))));
    A0=rand(4,4,4);
    A=tensor(A0.^2./(A0.^2+(1-A0).^2));
    y=(rand(N,1)<A(x(:,[9,11,13])));

    train=randsample(N,n);
    X0=x(:,[9,11,13]);
    Y0=y;
    x=x(train,[9,11,13]);
    Y=y(train);

    options=init_options();
    %mex drawZscPar.c CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp";
    %mex drawZsCollapsedPar.c CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp";
    %mex drawZscSparsePar.c CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp";
    tpl=10; % topics per level
    options.gam = .5;
    options.L = 2;
    options.topicModel = 'PAM';
    options.par = 0;
    options.maxIter = 1000;
    options.topicsPerLevel{1}=tpl;
    options.topicsPerLevel{2}=tpl;
    % options.collapsed = 0;
    npats=1000; %number of articificial patients
    
    disp(options); %print options

    L=options.L;

    %adjustment if using constant L across dims
    if length(L)==1
        L=repelem(L,2);
    end

    asdTr=sptensor([(1:n)',x],ones(size(x,1),1));
    asd = sptensor([(1:N)',X0],ones(size(X0,1),1));
    b = setdiff(1:N, train);
    [phi, psi, tree, samples, paths, prob, ~,~] = ...
        asdHBTucker3(asdTr,options);
    testPhi = asdHBTuckerNew(asd, psi, samples, paths, tree, prob ...
        b, options);

    %save data
    save(['data/yangHBTuckerCV_L', int2str(options.L), '_gam', ...
        num2str(tpl), '_', options.topicType, '_PAM.mat'],'phi', ...
        'testPhi', 'psi', 'tree', 'samples', 'paths', 'prob', 'options');

    r=cell(3,1);
    r{1}=unique(paths(:,1:L(1)));
    r{2}=unique(paths(:,(L(1)+1):(L(1)+L(2))));
    r{3}=unique(paths(:,(L(1)+L(2)+1):sum(L)));

    %compute LL
    teSparse=[(1:N)',X0];
    teSparse=teSparse(b,:);
    teSparse(:,1)=1:(N-n);
    asdTe=sptensor(teSparse,ones(size(teSparse,1),1));
    s=size(asd);
    LL=logLikelihood(asdTr, asdTe, npats, 1,  1/(prod(s(2:end))), ...
        psi, r, paths, tree, prob, samples, options);

    % print LL info
    fprintf('LL: %13.6e\n', LL);
% catch e
%     display(e.identifier);
%     display(e.message);
%     for i=1:size(e.stack,1)
%         display(e.stack(i,1));
%     end
% end