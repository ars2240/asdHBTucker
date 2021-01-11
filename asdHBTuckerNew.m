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
    
    if isnumeric(options.rng)
        options.rng=options.rng+1;
    end
    rng(options.rng); %seed RNG
    
    modes=size(tree,1); %number of dependent modes
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
        L=repelem(L,modes); options.L=L;
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
            
        case 'PAM'
            [tpl, r]=initPAM(dims,options);
            paths=ones(dims(1),sum(L));
            
            ctree=cell(modes,1);
            for i=1:modes
                ctree{i}=zeros(dims(1),dims(i+1),length(r{i}));
            end
            
        case 'None'
            r=cell(modes,1); %initialize
            path=zeros(1,sum(L));
            for i=1:modes
                r{i}=1:L(i);
                path(1+sum(L(1:(i-1))):sum(L(1:i)))=1:L(i);
            end         
            paths=repmat(path,dims(1),1);
            tree=cell(modes,1); %initialize
            
        otherwise
            error('Error. \nNo topic model type selected');
    end
    
    %old counts
    cStart=tic;
    [~,ocpsi,~] = counts(oSamples, [max(oSamples(:,1)), dims(2:3)], ...
        r, oPaths, [0,1,0], options);
    cTime=toc(cStart);
    if strcmp(options.topicModel, 'PAM')
        if ~exist('prob','var')
            prob=tree;
        end
        [paths,~,~]=newPAM(dims,ocpsi,ctree,paths,tpl,prob,options);
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
                    L,options.pType);
                LL=LL+sum(log(p));
                ent=ent+entropy(p);
            otherwise
                [samples,p]=drawZsCollapsed(samples,cphi,ocpsi,paths,...
                    L,options);
                LL=LL+sum(log(p));
                ent=ent+entropy(p);
        end
        zTime=toc(zStart);
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
    end
    
    if options.print==1 || options.keepBest > 0
        if options.map == 1
            phi = drawCoreMAP(samples,paths,coreDims,r,options);
        end
        phiS = sparsePhi(phi, coreDims, paths, options);
        if strcmp(options.topicModel,'PAM')
            LL2=logLikelihood(x, x, 1, 1/(size(x,2)*size(x,3)), psi, ...
                paths, {}, prob, samples, options);
            [LL, zLL, treeLL]=modelLL(phiS, psi, samples, paths, r, ...
                prob,options);
        else
            LL2=logLikelihood(x, x, 1, 1/(size(x,2)*size(x,3)), psi, ...
                paths, tree, samples, options);
            [LL, zLL, treeLL]=modelLL(phiS, psi, samples, paths, r, ...
                options);
        end
        if options.keepBest > 0
            if options.map == 1
                phi = drawCoreMAP(samples,paths,coreDims,r,options);
            end
            options.best.LL = LL; options.best.phi = phi;
            options.best.samples = samples;
            options.best.paths = paths;
        end
        if options.print
            fileID = fopen('verbose.txt','a');
            output_header=sprintf('%6s %13s %13s %10s %13s %13s',...
                'iter', 'loglikelihood', 'mixture LL', 'entropy',...
                'tree LL','z LL');
            fprintf(fileID,'%s\n',output_header);
            fprintf(fileID,...
                '%6i %13.2e %13.2e %10.2e %13.2e %13.2e\n',...
                0,LL,LL2,ent,treeLL,zLL);
            fclose(fileID);
        end
    end
    
    %gibbs sampler
    cont=1;
    nIter=0;
    while cont==1
        LL=0; %reset log-likelihood
        ent=0; %reset entropy
        
        %new counts
        cStart=tic;
        [~,~,ctree] = counts(samples, dims, r, paths, [0,0,1], options);
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
        
        for btIt=1:options.btReps
            if options.collapsed==1 && (nIter<(options.maxIter/10-1) ...
                    || options.map==1)
                
                cStart=tic;
                [cphi,cpsi,~] = counts(samples, dims, r, paths, [1,1,0], options);
                cTime=cTime+toc(cStart);
                
                tcpsi=cpsi;
                for i=1:modes
                    tcpsi{i}=cpsi{i}+ocpsi{i};
                end
                
                %draw latent topic z's
                zStart=tic;
                switch options.par
                    case 1
                        [samples,p]=drawZsCollapsedPar(samples,cphi,tcpsi,...
                            paths,L,options.pType);
                        LL=LL+sum(log(p));
                        ent=ent+entropy(p);
                    otherwise
                        ts = samples;
                        [samples,p]=drawZsCollapsed(samples,cphi,tcpsi,...
                            paths,L,options);
                        LL=LL+sum(log(p));
                        ent=ent+entropy(p);
                end
                zTime=toc(zStart);
                
                %MAP estimates
                if options.map==1 && nIter>=options.maxIter/10-1
                    psi = drawpsiMAP(samples, dims, r, paths, options);
                    coreDims=coreSize(modes, dims, r);
                    phi = drawCoreMAP(samples,paths,coreDims,r,options);
                end
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
        
        %increment iteration counter
        nIter=nIter+1;
        
        %print loglikelihood & entropy
        if (options.print==1 || options.keepBest > 0) && mod(nIter,options.freq)==0
            if options.map == 1
                phi = drawCoreMAP(samples,paths,coreDims,r,options);
            end
            phiS = sparsePhi(phi, coreDims, paths, options);
            if strcmp(options.topicModel,'PAM')
                LL2=logLikelihood(x, x, 1, 1/(size(x,2)*size(x,3)), psi, ...
                    paths, {}, prob, samples, options);
                [LL, zLL, treeLL]=modelLL(phiS, psi, samples, paths, r, ...
                    prob,options);
            else
                LL2=logLikelihood(x, x, 1, 1/(size(x,2)*size(x,3)), psi, ...
                    paths, tree, samples, options);
                [LL, zLL, treeLL]=modelLL(phiS, psi, samples, paths, r, ...
                    options);
            end
            %display(samples);
        	%disp([LL, treeLL, zLL]);
            if options.keepBest > 0 && options.best.LL < LL && LL~=0
                options.best.LL = LL; options.best.phi = phi;
                options.best.samples = samples;
                options.best.paths = paths;
            elseif options.keepBest == 1
                LL = options.best.LL; phi = options.best.phi;
                samples = options.best.samples;
                paths = options.best.paths;
            end
            if options.print == 1
                fileID = fopen('verbose.txt','a');
                fprintf(fileID,...
                    '%6i %13.2e %13.2e %10.2e %13.2e %13.2e\n',...
                    nIter,LL,LL2,ent,treeLL,zLL);
                fclose(fileID);
            end
        end
        
        %check if to continue
        if nIter>=(options.maxIter/10)
            cont=0;
        end
    end
    tTime=toc(tStart);
    
    if options.keepBest > 0
        phi = options.best.phi; paths = options.best.paths;
    end
    
    %reformat phi
    if options.sparse~=0
        phi = sparsePhi(phi, coreDims, paths, options);
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