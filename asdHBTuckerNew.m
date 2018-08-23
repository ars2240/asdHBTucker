function phi = asdHBTuckerNew(asdTens, psi, oSamples, oPaths, tree, b, options)
    %performs 3-mode condition probablility Bayesian Tucker decomposition 
    %on a counting tensor
    %P(mode 2, mode 3 | mode 1)
    % = P(mode 2 | topic 2) P(mode 3| topic 3) P(topic 2, topic 3| mode 1)
    %x = counting tensor to be decomposed
    %options = 
    % par = whether or not z's are computed in parallel
    % time = whether or not time is printed
    % print = whether or not loglikelihood & perplexity are printed
    % freq = how frequent loglikelihood & perplexity are printed
    % maxIter = number of Gibbs sample iterations
    % gam = hyper parameter(s) of CRP
    % L = levels of hierarchical trees
    
    tStart=tic;
    
    rng('shuffle'); %seed RNG
    
    x=asdTens(find(b),:,:);
    
    dims=size(x); %dimensions of tensor
    
    gam=options.gam;
    L=options.L;
    LL=0; %initialize log-likelihood
    ent=0; %initialize entropy
    
    %adjustment if using constant gam across dims
    if length(gam)==1
        gam=repelem(gam,2);
    end
    
    %adjustment if using constant L across dims
    if length(L)==1
        L=repelem(L,2);
    end
    
    %calculate L1 norm of tensor
    l1NormX=sum(x.vals);
    if l1NormX==0
        error('empty array');
    end
    
    %initialize sample matrix
    sStart=tic;
    samples=zeros(l1NormX,5);
    s=find(x>0); %find nonzero elements
    v=x(s(:,1),s(:,2),s(:,3)); %get nonzero values
    v=v.vals; %extract values
    samples(:,1:3)=repelem(s,v,1); %set samples
    %[~,xStarts]=unique(samples(:,1));
    %xEnds=[xStarts(2:length(xStarts))-1;size(samples,1)];
    sampTime=toc(sStart);
    
    %initialize tree
    treeStart=tic;
    switch options.topicModel
        case 'IndepTrees'
            [paths,r] = newTreePathsInit(oPaths,oSamples,tree,b,L);
        case 'PAM'
            error('Error. \nPAM code not written yet');
        case 'None'
            paths=repmat([1:L(1),1:L(2)],dims(1),1);
            r=cell(2,1); %initialize
            r{1}=1:L(1);
            r{2}=1:L(2);
            tree=cell(2,1); %initialize
        otherwise
            error('Error. \nNo topic model type selected');
    end
    treeTime=toc(treeStart);
    
    %calculate dimensions of core
    coreDims=zeros(1,3);
    coreDims(1)=dims(1);
    for i=1:2
       %set core dimensions to the number of topics in each mode
       coreDims(i+1)=length(r{i});
    end
    
    %draw core tensor p(z|x)
    coreStart=tic;
    [phi,p]=drawCoreUni(paths,coreDims,L,r,options);
    LL=LL+sum(p);
    ent=ent+entropy(exp(p));
    coreTime=toc(coreStart);
    
    %save('asd.mat','phi','psi','r','samples');
    %draw latent topic z's
    zStart=tic;
    switch options.par
        case 1
            [samples,p]=drawZscPar(samples,phi,psi,r);
            LL=LL+sum(log(p));
            ent=ent+entropy(p);
        otherwise
            [samples,p]=drawZsc(samples,phi,psi,r);
            LL=LL+sum(log(p));
            ent=ent+entropy(p);
    end
    zTime=toc(zStart);
    
    if options.print==1
        output_header=sprintf('%6s %13s %10s','iter','loglikelihood', ...
            'entropy');
        fprintf('%s\n',output_header);
        fprintf('%6i %13.2e %10.2e\n',0,LL,ent);
    end
    
    %gibbs sampler
    cont=1;
    nIter=0;
    while cont==1
        LL=0; %reset log-likelihood
        ent=0; %reset entropy
        
        for btIt=1:options.btReps

            %redraw core tensor p(z|x)
            %subset to get samples with x
            coreStart=tic;
            %redraw core tensor p(z|x)
            [phi,p]=drawCoreCon(samples,paths,coreDims,L,r,options);
            if btIt==options.btReps
                LL=LL+sum(p);
                ent=ent+entropy(exp(p));
            end
            coreTime=coreTime+toc(coreStart);

            %redraw latent topic z's
            zStart=tic;
            switch options.par
                case 1
                    [samples,p]=drawZscPar(samples,phi,psi,r);
                    LL=LL+sum(log(p));
                    ent=ent+entropy(p);
                otherwise
                    [samples,p]=drawZsc(samples,phi,psi,r);
            end
            if btIt==options.btReps
                LL=LL+sum(log(p));
                ent=ent+entropy(p);
            end
            zTime=zTime+toc(zStart);
        end
        
        %redraw tree
        treeStart=tic;
        for treeIt=1:options.treeReps
            switch options.topicModel
                case 'IndepTrees'
                    paths=newTreePaths(asdTens,oSamples,samples,oPaths,...
                        tree,b,L,r,options);
                case 'PAM'
                    error('Error. \nPAM code not written yet');
                case 'None'
                otherwise
                    error('Error. \nNo topic model type selected');
            end
        end
        treeTime=treeTime+toc(treeStart);
        
        %increment iteration counter
        nIter=nIter+1;
        
        %print loglikelihood & entropy
        if options.print==1
            if mod(nIter,options.freq)==0
                fprintf('%6i %13.2e %10.2e\n',...
                    nIter, LL, ent);
            end
        end
        
        %check if to continue
        if nIter>=(options.maxIter/10)
            cont=0;
        end
    end
    tTime=toc(tStart);
    
    %print times
    if options.time==1
        fprintf('Sample Init time= %5.2f\n',sampTime);
        fprintf('Core time= %5.2f\n',coreTime);
        fprintf('Z time= %5.2f\n',zTime);
        fprintf('Tree time= %5.2f\n',treeTime);
        fprintf('Total time= %5.2f\n',tTime);
    end
end