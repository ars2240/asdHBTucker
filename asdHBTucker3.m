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
    
    if options.collapsed==1
        
        %initialize zero counts
        dimsM=zeros(2,1);
        dimsM(1)=length(r{1});
        dimsM(2)=length(r{2});
        cphi=zeros(dims(1),dimsM(1),dimsM(2));
        cpsi=cell(2,1);
        cpsi{1}=zeros(dims(2),dimsM(1));
        cpsi{2}=zeros(dims(3),dimsM(2));
        
        %draw latent topic z's
        zStart=tic;
        switch options.par
            case 1
                [samples,p]=drawZsCollapsedPar(samples,cphi,cpsi,paths,...
                    L,options.prior);
                LL=LL+sum(log(p));
                ent=ent+entropy(p);
            otherwise
                [samples,p]=drawZsCollapsed(samples,cphi,cpsi,paths,L,...
                    options.prior);
                LL=LL+sum(log(p));
                ent=ent+entropy(p);
        end
        zTime=toc(zStart);
        
        %new counts
        cStart=tic;
        [cphi,cpsi,~] = counts(samples, dims, r);
        cTime=toc(cStart);
    else
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
        
        cTime=0;
    end
    
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
            if options.collapsed==1 && nIter<(options.maxIter-1)
                pad=max(r{1})-size(cphi,2);
                if pad>0
                   cphi=padarray(cphi,[0 pad 0],'post');
                   cpsi{1}=padarray(cpsi{1},[0 pad],'post');
                end
                pad=max(r{2})-size(cphi,3);
                if pad>0
                   cphi=padarray(cphi,[0 0 pad],'post');
                   cpsi{2}=padarray(cpsi{2},[0 pad],'post');
                end
                cphi=cphi(:,r{1},r{2});
                cpsi{1}=cpsi{1}(:,r{1});
                cpsi{2}=cpsi{2}(:,r{2});
                %draw latent topic z's
                zStart=tic;
                switch options.par
                    case 1
                        [samples,p]=drawZsCollapsedPar(samples,cphi,...
                            cpsi,paths,L,options.prior);
                        LL=LL+sum(log(p));
                        ent=ent+entropy(p);
                    otherwise
                        [samples,p]=drawZsCollapsed(samples,cphi,cpsi,...
                            paths,L,options.prior);
                        LL=LL+sum(log(p));
                        ent=ent+entropy(p);
                end
                zTime=toc(zStart);
            else
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
        end
        
        %redraw tree
        if nIter<(options.maxIter-1)
            
            %new counts
            cStart=tic;
            switch options.collapsed
                case 1
                    [cphi,cpsi,ctree] = counts(samples, dims, r);
                otherwise
                    [~,cpsi,ctree] = counts(samples, dims, r);
            end
            cTime=cTime+toc(cStart);
            
            treeStart=tic;
            for treeIt=1:options.treeReps
                switch options.topicModel
                    case 'IndepTrees'
                        [paths,tree,r,LLtree,entTree]=redrawTree(dims,...
                            cpsi,ctree,paths,L,tree,r,options);
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
            treeTime=treeTime+toc(treeStart);
        end
        LL=LL+LLtree;
        ent=ent+entTree;
        
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
        fprintf('Sample Init time= %5.2f sec\n',sampTime);
        if options.collapsed~=1
            fprintf('Matrix time= %5.2f sec\n',matTime);
            fprintf('Core time= %5.2f sec\n',coreTime);
        end
        hrs = floor(zTime/3600);
        zTime = zTime - hrs * 3600;
        min = floor(zTime/60);
        zTime = zTime - min * 60;
        fprintf('Z time= %2i hrs, %2i min, %4.2f sec \n', hrs, min, ...
            zTime);
        if options.collapsed==1
            fprintf('Count time= %5.2f sec\n',cTime);
        end
        hrs = floor(treeTime/3600);
        treeTime = treeTime - hrs * 3600;
        min = floor(treeTime/60);
        treeTime = treeTime - min * 60;
        fprintf('Tree time= %2i hrs, %2i min, %4.2f sec \n', hrs, min, ...
            treeTime);
        hrs = floor(tTime/3600);
        tTime = tTime - hrs * 3600;
        min = floor(tTime/60);
        tTime = tTime - min * 60;
        fprintf('Total time= %2i hrs, %2i min, %4.2f sec \n', hrs, min, ...
            tTime);
    end
end