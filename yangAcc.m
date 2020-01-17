try
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
    tpl=2; % topics per level
    options.gam = 1;
    options.L = 2;
    options.topicModel = 'IndepTrees';
    options.topicType = 'Level';
    options.par = 0;
    options.maxIter = 1000;
    options.topicsPerLevel{1}=tpl;
    options.topicsPerLevel{2}=tpl;
    options.topicsPerLevel{3}=tpl;
    % options.collapsed = 0;
    npats=1000; %number of articificial patients
    
    disp(options); %print options

    %save data
    if strcmp(options.topicModel,'PAM')
        load(['data/yangHBTuckerCV_L', int2str(options.L), '_tpl', ...
            num2str(tpl), '_', options.topicType, '_PAM.mat']);
    else
        load(['data/yangHBTuckerCV_L', int2str(options.L), '_gam', ...
            num2str(options.gam), '_trees.mat']);
    end
    
    %fit logistic regression
    x=double(tenmat(phi,1));
    nzcol=sum(x)>0;
    mod = mnrfit(x(:,nzcol),Y+1);
    
    %test error
    xt=double(tenmat(phi,1));
    ypred = mnrval(mod,xt(:,nzcol));
    [~,yhat]=min(Y0(indpred),[],2);
    mean(yhat~=(Y0(indpred)+1))

catch e
    display(e.identifier);
    display(e.message);
    for i=1:size(e.stack,1)
        display(e.stack(i,1));
    end
end