function [phi, psi, tree, samples, paths] = asdHBTucker3(x,options)
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
            [paths,tree,r,LLtree,entTree]=initializeTree(L,dims,gam);
        case 'PAM'
            [paths,tpl,tree,r,LLtree,entTree]=initializePAM(L,dims,options);
        case 'None'
            paths=repmat([1:L(1),1:L(2)],dims(1),1);
            r=cell(2,1); %initialize
            r{1}=1:L(1);
            r{2}=1:L(2);
            tree=cell(2,1); %initialize
            LLtree=0;
            entTree=0;
        otherwise
            error('Error. \nNo topic model type selected');
    end
    LL=LL+LLtree;
    ent=ent+entTree;
    treeTime=toc(treeStart);
    
    %calculate dimensions of core
    coreDims=zeros(1,3);
    coreDims(1)=dims(1);
    for i=1:2
       %set core dimensions to the number of topics in each mode
       coreDims(i+1)=length(r{i});
    end
    
    %draw matrices p(y|z)
    matStart=tic;
    psi=cell(2,1); %initialize
    for i=2:3
        %draw values from dirichlet distribution with uniform prior
        switch options.pType
            case 0
                prior=repelem(1/dims(i),dims(i));
            case 1
                prior=repelem(1,dims(i));
            otherwise
                error('Error. \nNo prior type selected');
        end
        [psiT,p]=drchrnd(prior,coreDims(i),options);
        psi{i-1}=psiT';
        LL=LL+sum(p);
        ent=ent+entropy(exp(p));
    end
    matTime=toc(matStart);
    
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
            %recalculate dimensions of core
            for i=1:2
               %set core dimensions to the number of topics in each mode
               coreDims(i+1)=length(r{i});
            end

            %matrices
            psi{1}=zeros(dims(2),coreDims(2));
            psi{2}=zeros(dims(3),coreDims(3));

            %redraw matrices p(y|z)
            matStart=tic;
            for i=2:3
                [u,~,ir]=unique(samples(:,2+i));
                samps=accumarray(ir,1:size(samples,1),[],@(w){samples(w,:)});
                dim=dims(i);
                [~,loc]=ismember(r{i-1},u);
                psiT=zeros(dim,coreDims(i));
                for j=1:coreDims(i)
                    %draw values from dirichlet distribution with uniform prior
                    %plus counts of occurances of both y & z
                    switch options.pType
                        case 0
                            prior=repelem(1/dim,dim);
                        case 1
                            prior=repelem(1,dim);
                        otherwise
                            error('Error. \nNo prior type selected');
                    end
                    if loc(j)~=0
                        prior=prior+histc(samps{loc(j)}(:,i)',1:dim);
                    end
                    [psiT(:,j),p]=drchrnd(prior,1,options);
                    if btIt==options.btReps
                        LL=LL+sum(p);
                        ent=ent+entropy(exp(p));
                    end
                end
                psi{i-1}=psiT;
            end
            matTime=matTime+toc(matStart);

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
                    [paths,tree,r,LLtree,entTree]=redrawTree(dims,...
                        samples,paths,L,tree,r,gam);
                case 'PAM'
                    [paths,tree,LLtree,entTree]=redrawPAM(dims,samples,...
                        paths,tpl,tree,L,options);
                case 'None'
                    LLtree=0;
                    entTree=0;
                otherwise
                    error('Error. \nNo topic model type selected');
            end
        end
        LL=LL+LLtree;
        ent=ent+entTree;
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
        if nIter>=options.maxIter
            cont=0;
        end
    end
    tTime=toc(tStart);
    
    %print times
    if options.time==1
        fprintf('Sample Init time= %5.2f\n',sampTime);
        fprintf('Matrix time= %5.2f\n',matTime);
        fprintf('Core time= %5.2f\n',coreTime);
        fprintf('Z time= %5.2f\n',zTime);
        fprintf('Tree time= %5.2f\n',treeTime);
        fprintf('Total time= %5.2f\n',tTime);
    end
end