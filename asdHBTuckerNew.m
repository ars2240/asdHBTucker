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
            paths=zeros(dims(1),sum(L));
            if L(1)~=L(2)
                error("Error. \nLevels do not match");
            end

            %reformat topicsPerLevel as cell of vectors of correct length
            if iscell(options.topicsPerLevel)
                tpl=options.topicsPerLevel;
                if length(tpl)~=2
                    error("Error. \nNumber of cells !=2");
                end
                if length(tpl{1})==1
                    tpl{1}=[1,repelem(tpl{1}(1),L(1)-1)];
                elseif length(tpl{1})~=(L(1)-1)
                    error("Error. \nInvalid length of topics per level");
                end
                if length(tpl{2})==1
                    tpl{2}=repelem(tpl{2}(1),L(2));
                elseif length(tpl{2})~=(L(2)-1)
                    error("Error. \nInvalid length of topics per level");
                end
            else
                tplV=options.topicsPerLevel;
                tpl=cell(2,1);
                if length(tplV)==1
                    tpl{1}=[1,repelem(tplV(1),L(1)-1)];
                    tpl{2}=repelem(tplV(1),L(2));
                elseif length(tplV)==2
                    tpl{1}=[1,repelem(tplV(1),L(1)-1)];
                    tpl{2}=repelem(tplV(2),L(2));
                elseif length(tplV)==L(1)
                    tpl{1}=tplV;
                    tpl{2}=tplV;
                elseif length(tplV)==2*(L(1))
                    tpl{1}=tplV(1:L(1));
                    tpl{2}=tplV((L(1)+1):2*L(1));
                else
                    error("Error. \nInvalid length of topics per level");
                end
            end

            %initialize restaurant list
            r=cell(2,1);
            r{1}=1:(sum(tpl{1}));
            r{2}=1:(sum(tpl{2}));
            
            [paths,prob,~,~]=newPAM(dims,oSamples,paths,tpl,prob,L);
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
    
    %old counts
    cStart=tic;
    switch options.collapsed
        case 1
            [ocphi,ocpsi,~] = counts(oSamples, [max(oSamples(:,1)), ...
                dims(2:3)], r);
        otherwise
            [~,ocpsi,~] = counts(oSamples, [max(oSamples(:,1)), ...
                dims(2:3)], r);
    end
    cTime=toc(cStart);
    
    %calculate dimensions of core
    coreDims=zeros(1,3);
    coreDims(1)=dims(1);
    for i=1:2
       %set core dimensions to the number of topics in each mode
       coreDims(i+1)=length(r{i});
    end
    
    if options.collapsed==1
        
        %initialize zero counts
        dimsM=zeros(2,1);
        dimsM(1)=size(r{1},1);
        dimsM(2)=size(r{2},1);
        cphi=zeros(dims(1),dimsM(1),dimsM(2));
        
        %draw latent topic z's
        zStart=tic;
        switch options.par
            case 1
                [samples,p]=drawZsCollapsedPar(samples,cphi,ocpsi,paths,...
                    L,options.prior);
                LL=LL+sum(log(p));
                ent=ent+entropy(p);
            otherwise
                [samples,p]=drawZsCollapsed(samples,cphi,ocpsi,paths,...
                    L,options.prior);
                LL=LL+sum(log(p));
                ent=ent+entropy(p);
        end
        zTime=toc(zStart);
        
        %new counts
        cStart=tic;
        [cphi,cpsi,~] = counts(samples, dims, r);
        cTime=cTime+toc(cStart);
        
        coreTime=0;
        
    else

        %draw core tensor p(z|x)
        coreStart=tic;
        [phi,p]=drawCoreUni(paths,coreDims,L,r,options);
        if ndims(phi) < 3
            phi(end, end, 2) = 0; 
        end
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
            if options.collapsed==1 && nIter<(options.maxIter/10-1)
                
                tcpsi=cpsi;
                tcpsi{1}=cpsi{1}+ocpsi{1};
                tcpsi{2}=cpsi{2}+ocpsi{2};

                %draw latent topic z's
                zStart=tic;
                switch options.par
                    case 1
                        [samples,p]=drawZsCollapsedPar(samples,cphi,tcpsi,...
                            paths,L,options.prior);
                        LL=LL+sum(log(p));
                        ent=ent+entropy(p);
                    otherwise
                        [samples,p]=drawZsCollapsed(samples,cphi,tcpsi,...
                            paths,L,options.prior);
                        LL=LL+sum(log(p));
                        ent=ent+entropy(p);
                end
                zTime=toc(zStart);
            else
                %redraw core tensor p(z|x)
                %subset to get samples with x
                coreStart=tic;
                %redraw core tensor p(z|x)
                [phi,p]=drawCoreCon(samples,paths,coreDims,L,r,options);
                if ndims(phi) < 3
                    phi(end, end, 2) = 0; 
                end
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
        
        %new counts
        cStart=tic;
        switch options.collapsed
            case 1
                [cphi,cpsi,ctree] = counts(samples, dims, r);
            otherwise
                [~,~,ctree] = counts(samples, dims, r);
        end
        cTime=cTime+toc(cStart);
        
        %redraw tree
        treeStart=tic;
        for treeIt=1:options.treeReps
            switch options.topicModel
                case 'IndepTrees'
                    paths=newTreePaths(asdTens,ocpsi,ctree,oPaths,...
                        tree,b,L,options);
                case 'PAM'
                    [paths,prob,~,~]=newPAM(dims,oSamples,paths,tpl,prob,L);
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
        if options.collapsed~=1
            fprintf('Core time= %5.2f sec\n',coreTime);
        end
        fprintf('Z time= %5.2f\n',zTime);
        if options.collapsed==1
            fprintf('Count time= %5.2f sec\n',cTime);
        end
        fprintf('Tree time= %5.2f\n',treeTime);
        fprintf('Total time= %5.2f\n',tTime);
    end
end