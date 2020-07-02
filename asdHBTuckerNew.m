function phi = asdHBTuckerNew(asdTens, psi, oSamples, oPaths, tree, varargin)
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
    
    if nargin==7
        b=varargin{1};
        options=varargin{2};
    elseif nargin==8
        prob=varargin{1};
        b=varargin{2};
        options=varargin{3};
    else
        error("Error. \nIncorrect number of inputs.");
    end
    
    tStart=tic;
    
    rng('shuffle'); %seed RNG
    
    modes=length(tree); %number of dependent modes
    dims=size(asdTens); %dimensions of tensor
    cts=collapse(asdTens,[2,3]);
    bo = b;
    b = bo & cts>0;
    dims(1)=sum(b);
    ind=asdTens.subs;
    ind=ind(b(ind(:,1)),:);
    vals=asdTens(tensIndex2(ind,size(asdTens)));
    [~,~,ind(:,1)]=unique(ind(:,1));
    
    %calculate L1 norm of tensor
    l1NormX=sum(vals);
    if l1NormX==0
        error('empty array');
    end
    
    % reshape x
    x=sptensor(ind,vals);
    
    L=options.L;
    LL=0; %initialize log-likelihood
    ent=0; %initialize entropy
    
    %adjustment if using constant L across dims
    if length(L)==1
        L=repelem(L,modes);
    end
    
    
    %initialize sample matrix
    sStart=tic;
    samples=zeros(l1NormX,1+2*modes);
    s=x.subs; %find nonzero elements
    v=x.vals; %get nonzero values
    samples(:,1:(1+modes))=repelem(s,v,1); %set samples
    %[~,xStarts]=unique(samples(:,1));
    %xEnds=[xStarts(2:length(xStarts))-1;size(samples,1)];
    sampTime=toc(sStart);
    
    %initialize tree
    treeStart=tic;
    switch options.topicModel
        case 'IndepTrees'
            [paths,r] = newTreePathsInit(oPaths,tree,b,L);
            
            %old counts
            cStart=tic;
            [~,ocpsi,~] = counts(oSamples, ...
                [max(oSamples(:,1)), dims(2:end)], r, paths, [0,1,0], options);
            cTime=toc(cStart);
        case 'PAM'
            [tpl, r]=initPAM(dims,options);
            paths=ones(dims(1),sum(L));
            
            %old counts
            cStart=tic;
            [~,ocpsi,~] = counts(oSamples, ...
                [max(oSamples(:,1)), dims(2:end)], r, paths, [0,1,0], options);
            cTime=toc(cStart);
            
            ctree=cell(modes,1);
            for i=1:modes
                ctree{i}=zeros(dims(1),dims(i+1),length(r{i}));
            end
            
            [paths,~,~]=newPAM(dims,ocpsi,ctree,paths,tpl,prob,options);
        case 'None'
            r=cell(modes,1); %initialize
            path=zeros(1,sum(L));
            for i=1:modes
                r{i}=1:L(i);
                path(1+sum(L(1:(i-1))):sum(L(1:i)))=1:L(i);
            end         
            paths=repmat(path,options.npats,1);
            tree=cell(modes,1); %initialize
            
            %old counts
            cStart=tic;
            [~,ocpsi,~] = counts(oSamples, ...
                [max(oSamples(:,1)), dims(2:3)], r, paths, [0,1,0], options);
            cTime=toc(cStart);
            
        otherwise
            error('Error. \nNo topic model type selected');
    end
    treeTime=toc(treeStart);

    %calculate dimensions of core
    coreDims=zeros(1,1+modes);
    coreDims(1)=dims(1);
    for i=1:modes
       %set core dimensions to the number of topics in each mode
       coreDims(i+1)=max(r{i});
    end
    
    if options.collapsed==1
        
        %initialize zero counts
        if options.sparse
            cphi=zeros([coreDims(1),L]);
        else
            cphi=zeros(coreDims);
        end
        
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
        [cphi,cpsi,~] = counts(samples, dims, r, paths, [1,1,0], options);
        cTime=cTime+toc(cStart);
        
        coreTime=0;
        
    else

        %draw core tensor p(z|x)
        coreStart=tic;
        [phi,p]=drawCoreUni(paths,coreDims,r,options);
        if ndims(phi) < 3
            phi(end, end, 2) = 0; 
        end
        LL=LL+sum(p);
        ent=ent+entropy(exp(p));
        coreTime=toc(coreStart);

        %save('asd.mat','phi','psi','r','samples');
        %draw latent topic z's
        zStart=tic;
        switch options.sparse
            case 0
                switch options.par
                    case 1
                        [samples,~]=drawZscPar(samples,phi,psi,r);
                    otherwise
                        [samples,~]=drawZsc(samples,phi,psi,r);
                end
            otherwise
                switch options.par
                    case 1
                        [samples,~] = drawZscSparsePar(samples,phi,psi,...
                            paths,L);
                    otherwise
                        [samples,~] = drawZscSparse(samples,phi,psi,...
                            paths,L);
                end
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
                for i=1:modes
                    tcpsi{i}=cpsi{i}+ocpsi{i};
                end

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
                [phi,p]=drawCoreCon(samples,paths,coreDims,r,options);
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
                switch options.sparse
                    case 0
                        switch options.par
                            case 1
                                [samples,p]=drawZscPar(samples,...
                                    phi,psi,r);
                            otherwise
                                [samples,p]=drawZsc(samples,...
                                    phi,psi,r);
                        end
                    otherwise
                        switch options.par
                            case 1
                                [samples,p] = drawZscSparsePar(samples,...
                                    phi,psi,paths,L);
                            otherwise
                                [samples,p] = drawZscSparse(samples,...
                                    phi,psi,paths,L);
                        end
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
                [cphi,cpsi,ctree] = counts(samples, dims, r, paths, options);
            otherwise
                [~,~,ctree] = counts(samples, dims, r, paths, [0,0,1], options);
        end
        cTime=cTime+toc(cStart);
        
        %redraw tree
        treeStart=tic;
        for treeIt=1:options.treeReps
            switch options.topicModel
                case 'IndepTrees'
                    paths=newTreePaths(asdTens,ocpsi,ctree,oPaths,...
                        tree,b,options);
                case 'PAM'
                    [paths,~,~]=newPAM(dims,ocpsi,ctree,paths,tpl,prob,...
                        options);
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
    
    res=cell(modes,1);
    %reformat phi
    if options.sparse~=0
        phiT=sptensor([],[],coreDims);
        for i=1:coreDims(1)
            for j=1:modes
                res{j}=paths(i,(1+sum(L(1:(j-1))):sum(L(1:j))));
            end
            switch options.topicType
                case 'Cartesian'
                    len = prod(L);
                    subs=[repmat(i,[len,1]),tensIndex(res)];
                    vals=reshape(phi(i,:),[len,1]);
                case 'Level'
                    subs=zeros(L(1),1+modes);
                    subs(:,1)=i;
                    for j=1:modes
                        subs(:,j+1)=res{j};
                    end
                    vals=squeeze(phi(i,:));
                    vals=vals(vals>0)';
                otherwise
                    error('Error. \nNo topic type selected');
            end   
            phiT=phiT+sptensor(subs,vals,coreDims);
        end
        phi=phiT;
    end

    % add zeros
    phio=phi;
    sph=size(phio);
    phi=zeros([sum(bo),sph(2:end)]);
    phi(cts(bo)>0,:,:)=phio;
    
    %print times
    if options.time==1
        fprintf('Test\n');
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