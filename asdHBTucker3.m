function [phi, psi, tree, samples, paths, varargout] = asdHBTucker3(x,options)
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
    
    rng(options.rng); %seed RNG
    
    % store original tensor
    odims=size(x); %dimensions of tensor
    modes=length(odims)-1;  %number of dependent modes
    options.varRat=zeros(options.maxIter+1,1);
    
    % remove zeros
    cts=collapse(x,[2,3]);
    zind=cts==0;
    if sum(zind)>0
        ind=cell(modes+1,1);
        for i=1:modes
            ind{1+i}=1:odims(1+i);
        end
        ind{1}=find(cts>0)';
        x=x(tensIndex2(ind,odims));
        s=[length(ind{1}),odims(2:end)];
        x=reshape(x,s);
        x=sptensor(x);
        if ndims(x)<3
            x=sptensor([x.subs,ones(size(x.subs,1),1)],x.vals,s);
        end
    end

    dims=size(x); %dimensions of tensor
    
    gam=options.gam; L=options.L;
    
    %adjustment if using constant gam across dims
    if length(gam)==1
        gam=repelem(gam,modes);
        options.gam=gam;
    end
    
    %adjustment if using constant L across dims
    if length(L)==1
        L=repelem(L,modes);
        options.L=L;
    end
    
    if length(options.weights)==1
        options.weights=repelem(weights,modes);
    end
    
    %calculate L1 norm of tensor
    l1NormX=sum(x.vals);
    if l1NormX==0
        error('empty array');
    end
    
    %initialize sample matrix
    sStart=tic;
    samples=zeros(l1NormX,1+2*modes);
    s=find(x>0); %find nonzero elements
    v=x(s); %extract values
    samples(:,1:(1+modes))=repelem(s,v,1); %set samples
    %[~,xStarts]=unique(samples(:,1));
    %xEnds=[xStarts(2:length(xStarts))-1;size(samples,1)];
    sampTime=toc(sStart);
    
    %initialize tree
    treeStart=tic;
    switch options.topicModel
        case 'IndepTrees'
            [paths,tree,r,treeLL,treeEnt]=initializeTree(dims, options);
        case 'PAM'
            [paths,tpl,prob,r,treeLL,treeEnt]=initializePAM(dims,options);
            options.topicsPerLevel = tpl;
        case 'None'
            levels=ones(1,sum(L));
            r=cell(modes,1); %initialize
            tree=cell(modes,1); %initialize
            for i=1:modes
                levels((1+sum(L(1:(i-1)))):sum(L(1:i)))=1:L(i);
                r{i}=1:L(i);
            end
            paths=repmat(levels,dims(1),1);
            treeLL=0; treeEnt=0;
        otherwise
            error('Error. \nNo topic model type selected');
    end
    treeTime=toc(treeStart);
    
    if options.collapsed==1
        
        if isfield(options,'init') && isfield(options.init,'phi')
            t=tabulate(samples(:,1));
            cphi = options.init.phi.*t(:,2);
        else
            %initialize zero counts
            if strcmp(options.topicType,'CP')
                dimsM=length(r)*ones(modes,1);
            else
                dimsM=zeros(modes,1);
                for i=1:modes
                    dimsM(i)=length(r{i});
                end
            end

            if options.sparse==0
                cphi=zeros([dims(1),dimsM]);
            else
                cphi=zeros([dims(1),L]);
            end
        end
        
        if isfield(options,'init') && isfield(options.init,'psi')
            cpsi = options.init.psi;
            for i=1:modes
                t=tabulate(samples(:,i+1));
                cpsi{i}=options.init.psi{i}.*t(:,2);
            end
        else
            cpsi=cell(modes,1);
            for i=1:modes
                cpsi{i}=zeros(dims(i+1),dimsM(i));
            end
        end
        
        %draw latent topic z's
        zStart=tic;
        switch options.par
            case 1
                [samples,p]=drawZsCollapsedPar(samples,cphi,cpsi,paths,...
                    L,options.pType);
            otherwise
                [samples,p,var]=drawZsCollapsed2(samples,cphi,cpsi,paths,L,...
                    options);
                options.varRat(1)=mean(var(:,1)./var(:,2));
        end
        zLL=sum(log(p)); zEnt=entropy(p);
        zTime=toc(zStart);
        
        matTime=0; coreTime=0;
    else
        coreDims=coreSize(modes, dims, r);

        if isfield(options,'init') && isfield(options.init,'psi')
            psi = options.init.psi;
        else
            %draw matrices p(y|z)
            matStart=tic;
            psi=cell(modes,1); %initialize
            for i=2:(modes+1)
                %draw values from dirichlet distribution with uniform prior
                switch options.pType
                    case 0
                        prior=repelem(1/dims(i),dims(i));
                    case 1
                        prior=repelem(1,dims(i));
                    case 0
                        prior=repelem(2/dims(i),dims(i));
                    otherwise
                        error('Error. \nNo prior type selected');
                end
                [psiT,~]=drchrnd(prior,coreDims(i),options);
                psi{i-1}=psiT';
            end
            matTime=toc(matStart);
        end

        if isfield(options,'init') && isfield(options.init,'phi')
            phi = options.init.phi;
        else
            %draw core tensor p(z|x)
            coreStart=tic;
            [phi,~]=drawCoreUni(paths,coreDims,options);
            coreTime=toc(coreStart);
        end
        
        %draw latent topic z's
        zStart=tic;
        oSamples=samples;
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
        p=collapsedProb(samples, oSamples, dims, r, paths, options);
        zLL=sum(log(p)); zEnt=entropy(p);
        zTime=toc(zStart);
    end
    cTime=0;
    
    % sum LL & Entropy
    % LL=treeLL+matLL+coreLL+zLL; ent=treeEnt+matEnt+coreEnt+zEnt;
    LL=treeLL+zLL; ent=treeEnt+zEnt;
    
    if options.print==1 || options.keepBest > 0
        if options.topicsgoal>0
            coreDims=coreSize(modes, dims, r);
            nt=prod(coreDims(2:end));
        end
        if options.collapsed==1 && options.map == 1
            psi = drawpsiMAP(samples, dims, r, paths, options);
        elseif options.collapsed==1 || options.topicsgoal>0
            [psi,~,~]=drawpsi(dims, modes, samples, r, options);
        end
        c = zeros(1, modes); nu = zeros(1, modes);
        for i = 1:modes
            [c(i), nu(i)] = coh(collapse(x,4-i,@max), psi{i}, options);
        end
        cm = mean(c);
        if options.map == 1
            coreDims=coreSize(modes, dims, r);
            phi = drawCoreMAP(samples,paths,coreDims,r,options);
        end
        phiS = sparsePhi(phi, coreDims, paths, options);
        n = norm(x-ttm(tensor(phiS), psi, [2,3]));
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
                coreDims=coreSize(modes, dims, r);
                phi = drawCoreMAP(samples,paths,coreDims,r,options);
            end
            options.best.LL = -inf; options.best.phi = phi;
            options.best.psi = psi; options.best.samples = samples;
            options.best.paths = paths; options.best.r = r;
            options.best.gamma = options.gam; options.best.iter = 0;
            options.best.cm = -inf; options.best.c=c; options.best.nu=nu;
            if strcmp(options.topicModel,'PAM')
                options.best.prob = prob;
            else
                options.best.tree = tree;
            end
        end
        if options.print == 1 && options.topicsgoal == 0
            fileID = fopen('verbose.txt','w');
            output_header=sprintf('%6s %13s %13s %13s %13s %13s %13s',...
                'iter', 'loglikelihood', 'mixture LL', 'entropy',...
                'tree LL','z LL', 'norm');
            fprintf(fileID,'%s\n',output_header);
            fprintf(fileID,...
                '%6i %13.2e %13.2e %13.2e %13.2e %13.2e %13.2e\n',...
                0,LL,LL2,ent,treeLL,zLL,n);
            fclose(fileID);
        elseif options.print == 1
            div = diversity(psi);
            fileID = fopen('verbose.txt','w');
            output_header=sprintf('%6s %13s %13s %13s %13s %13s %6s %13s %13s %13s %13s %13s',...
                'iter', 'loglikelihood', 'mixture LL', 'entropy',...
                'tree LL','z LL', 'ntop', 'gamma', 'div1', 'div2', 'norm','cm');
            fprintf(fileID,'%s\n',output_header);
            fprintf(fileID,...
                '%6i %13.2e %13.2e %13.2e %13.2e %13.2e %6i %13.2e %13.2e %13.2e %13.2e %13.2e\n',...
                0,LL,LL2,ent,treeLL,zLL,nt,options.gam(1),div(1),div(2),n, cm);
            fclose(fileID);
        end
    end
    
    %gibbs sampler
    cont=1; nIter=0;
    while cont==1
        
        %new counts
        cStart=tic;
        [~,cpsi,ctree] = counts(samples, dims, r, paths, [0,1,1], options);
        cTime=cTime+toc(cStart);

        %redraw tree
        treeStart=tic;
        for treeIt=1:options.treeReps
            switch options.topicModel
                case 'IndepTrees'
                    [paths,tree,r,treeLL,treeEnt]=redrawTree(dims,...
                        cpsi,ctree,paths,tree,r,options);
                case 'PAM'
                    [paths,prob,treeLL,treeEnt]=redrawPAM(dims,...
                        cpsi,ctree,paths,tpl,prob,L,options);
                case 'None'
                    treeLL=0; treeEnt=0;
                otherwise
                    error('Error. \nNo topic model type selected');
            end
        end
        treeTime=treeTime+toc(treeStart);
        
        
        for btIt=1:options.btReps
            if options.collapsed==1 && (nIter<(options.maxIter-1) ...
                    || options.map==1)
                
                cStart=tic;
                [cphi,cpsi,~] = counts(samples, dims, r, paths, [1,1,0], options);
                cTime=cTime+toc(cStart);
                
                for i=1:modes
                    if strcmp(options.topicType,'CP')
                        pad=max(r)-size(cphi,i+1);
                    else
                        pad=max(r{i})-size(cphi,i+1);
                    end
                    if pad>0
                       if options.sparse==0
                           z=zeros(1,modes+1);
                           z(i+1)=pad;
                           cphi=padarray(cphi,z,'post');
                       end
                       cpsi{i}=padarray(cpsi{i},[0 pad],'post');
                    end
                end
                if options.sparse==0
                   ind=cell(length(r)+1,1);
                   for i=1:modes
                        if strcmp(options.topicType,'CP')
                            ind{1+i}=r;
                        else
                            ind{1+i}=r{i};
                        end
                   end
                   ind{1}=1:size(cphi,1);
                   cphi=cphi(tensIndex2(ind,size(cphi)));
                end
                for i=1:modes
                    if strcmp(options.topicType,'CP')
                        cpsi{i}=cpsi{i}(:,r);
                    else
                        cpsi{i}=cpsi{i}(:,r{i});
                    end
                end
                %draw latent topic z's
                zStart=tic;
                s = samples;
                switch options.par
                    case 1
                        [samples,p]=drawZsCollapsedPar(samples,cphi,...
                            cpsi,paths,L,options.pType);
                    otherwise
                        [samples,p,var]=drawZsCollapsed2(samples,cphi,cpsi,...
                            paths,L,options);
                        options.varRat(nIter+2)=mean(var(:,1)./var(:,2));
                end
                zLL=sum(log(p)); zEnt=entropy(p);
                zTime=toc(zStart);
                
                %MAP estimates
                if options.map==1 && nIter==options.maxIter-1
                    psi = drawpsiMAP(samples, dims, r, paths, options);
                    coreDims=coreSize(modes, dims, r);
                    phi = drawCoreMAP(samples,paths,coreDims,r,options);
                end
            else
                coreDims=coreSize(modes, dims, r);

                %redraw matrices p(y|z)
                matStart=tic;
                [psi,~,~]=drawpsi(dims, modes, samples, r, options);
                matTime=matTime+toc(matStart);

                %redraw core tensor p(z|x)
                %subset to get samples with x
                coreStart=tic;
                %redraw core tensor p(z|x)
                [phi,~]=drawCoreCon(samples,paths,coreDims,r,options);
                coreTime=coreTime+toc(coreStart);

                %redraw latent topic z's
                zStart=tic;
                oSamples=samples;
                switch options.sparse
                    case 0
                        switch options.par
                            case 1
                                [samples,~]=drawZscPar(samples,...
                                    phi,psi,r);
                            otherwise
                                [samples,~]=drawZsc(samples,...
                                    phi,psi,r);
                        end
                    otherwise
                        switch options.par
                            case 1
                                [samples,~] = drawZscSparsePar(samples,...
                                    phi,psi,paths,L);
                            otherwise
                                [samples,~] = drawZscSparse(samples,...
                                    phi,psi,paths,L);
                        end
                end
                if btIt==options.btReps
                    p=collapsedProb(samples, oSamples, dims, r, paths, options);
                    zLL=sum(log(p));
                    zEnt=entropy(p);
                end
                zTime=zTime+toc(zStart);
            end
        end
        
        % sum LL & Entropy
        % LL=treeLL+matLL+coreLL+zLL; ent=treeEnt+matEnt+coreEnt+zEnt;
        LL=treeLL+zLL; ent=treeEnt+zEnt;
        
        %increment iteration counter
        nIter=nIter+1;
        %display(samples);
        %disp([LL, treeLL, zLL]);
        
        if options.topicsgoal>0
            coreDims=coreSize(modes, dims, r);
            nt=prod(coreDims(2:end));
        end
        
        %print loglikelihood & entropy
        if (options.print==1 || options.keepBest > 0) && mod(nIter,options.freq)==0
            if options.map == 1
                psi = drawpsiMAP(samples, dims, r, paths, options);
            elseif (options.collapsed==1 || options.topicsgoal>0) && nIter<(options.maxIter-1)
                [psi,~,~]=drawpsi(dims, modes, samples, r, options);
            end
            c = zeros(1, modes); nu = zeros(1, modes);
            for i = 1:modes
                [c(i), nu(i)] = coh(collapse(x,4-i,@max), psi{i}, options);
            end
            cm = mean(c);
            if options.map == 1
                coreDims=coreSize(modes, dims, r);
                phi = drawCoreMAP(samples,paths,coreDims,r,options);
            end
            phiS = sparsePhi(phi, coreDims, paths, options);
            n = norm(x-ttm(tensor(phiS), psi, [2,3]));
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
            if options.keepBest > 0 && (options.keepBest <= 2 && ...
                    options.best.LL < LL && LL~=0) || ...
                    (options.keepBest > 2 && options.best.cm < cm && ...
                    cm ~=0)
                options.best.LL = LL; options.best.phi = phi;
                options.best.psi = psi; options.best.samples = samples;
                options.best.paths = paths; options.best.r = r;
                options.best.gamma = options.gam;
                options.best.iter = nIter; options.best.cm = cm;
                options.best.c=c; options.best.nu=nu;
                if strcmp(options.topicModel,'PAM')
                    options.best.prob = prob;
                else
                    options.best.tree = tree;
                end
            elseif mod(options.keepBest, 2) == 1
                LL = options.best.LL; phi = options.best.phi;
                psi = options.best.psi; samples = options.best.samples;
                paths = options.best.paths; r = options.best.r;
                options.gam = options.best.gamma;
                if strcmp(options.topicModel,'PAM')
                    prob = options.best.prob;
                else
                    tree = options.best.tree;
                end
            end
            if options.print == 1 && options.topicsgoal == 0
                fileID = fopen('verbose.txt','a');
                fprintf(fileID,...
                    '%6i %13.2e %13.2e %13.2e %13.2e %13.2e %13.2e\n',...
                    nIter,LL,LL2,ent,treeLL,zLL,n);
                fclose(fileID);
            elseif options.print == 1
                div = diversity(psi);
                fileID = fopen('verbose.txt','a');
                fprintf(fileID,...
                    '%6i %13.2e %13.2e %13.2e %13.2e %13.2e %6i %13.2e %13.2e %13.2e %13.2e %13.2e\n',...
                    nIter,LL,LL2,ent,treeLL,zLL,nt,options.gam(1),div(1),div(2),n,cm);
                fclose(fileID);
            end
        end
        
        %check if to continue
        if nIter>=options.maxIter
            cont=0;
        elseif options.topicsgoal>0 && strcmp(options.topicModel,'IndepTrees')
            %d=(modes*max(prod(L-1),1));
            d=(max(prod(L-1),1));
            ng=(options.topicsgoal/nt)^(1/d);
            ng=max(min(ng,2),0.5);
            options.gam=ng*options.gam;
        end
    end
    tTime=toc(tStart);
    
    if options.keepBest > 0
        LL = options.best.LL; phi = options.best.phi;
        psi = options.best.psi; samples = options.best.samples;
        paths = options.best.paths; r = options.best.r;
        options.gam = options.best.gamma;
        if strcmp(options.topicModel,'PAM')
            prob = options.best.prob;
            tree = prob;
        else
            tree = options.best.tree;
        end
        coreDims=coreSize(modes, dims, r);
    end
    
    %compute model size: log(n)k
    ln=log(dims(1)); %log of sample size
    %number of parameters from decomposition
    k=dims(1)*(prod(L)-1);
    for i=1:modes
        k=k+coreDims(i+1)*(dims(i+1)-1);
    end
    switch options.topicModel
        case 'IndepTrees'
            k=k+dims(1)*(sum(L)-modes);
        case 'PAM'
            k=k+dims(1)*(sum(L)-1);
            for i=1:(L(1)-1)
                for j=1:(modes-1)
                    k=k+tpl{j}(i)*(tpl{j+1}(i)-1);
                end
                k=k+tpl{modes}(i)*(tpl{1}(i+1)-1);
            end
            for j=1:(modes-1)
                k=k+tpl{j}(L(1))*(tpl{j+1}(L(1))-1);
            end
    end
    ms=ln*k;
    
    %reformat phi
    if options.sparse~=0
        phi = sparsePhi(phi, coreDims, paths, options);
    end
    
    % add zeros
    phio=phi;
    sph=size(phio);
    phi=zeros([odims(1),sph(2:end)]);
    phi(~zind,:,:)=phio;
    
    if options.keepBest > 0
        options.best.phi = phi;
    end
    
    if nargout==7
        varargout{1}=LL; varargout{2}=ms;
    elseif nargout==8
        if strcmp(options.topicModel,'PAM')
            varargout{1}=prob; varargout{2}=LL; varargout{3}=ms;
        else
            varargout{1}=options; varargout{2}=LL; varargout{3}=ms;
        end
    elseif nargout==9
        varargout{1}=prob; varargout{2}=options; varargout{3}=LL;
        varargout{4}=ms;
    else
        error("Error. \nIncorrect number of outputs.");
    end
        
    
    %print times
    if options.time==1
        fprintf('Sample Init time= %5.2f sec\n',sampTime);
        if options.collapsed~=1
            fprintf('Matrix time= %5.2f sec\n',matTime);
            fprintf('Core time= %5.2f sec\n',coreTime);
        end
        hrs = floor(zTime/3600);
        zTime = zTime - hrs * 3600;
        mins = floor(zTime/60);
        zTime = zTime - mins * 60;
        fprintf('Z time= %2i hrs, %2i min, %4.2f sec \n', hrs, mins, ...
            zTime);
        if options.collapsed==1
            hrs = floor(cTime/3600);
            cTime = cTime - hrs * 3600;
            mins = floor(cTime/60);
            cTime = cTime - mins * 60;
            fprintf('Count time= %2i hrs, %2i min, %4.2f sec \n', hrs, ...
                mins, cTime);
        end
        hrs = floor(treeTime/3600);
        treeTime = treeTime - hrs * 3600;
        mins = floor(treeTime/60);
        treeTime = treeTime - mins * 60;
        fprintf('Tree time= %2i hrs, %2i min, %4.2f sec \n', hrs, mins, ...
            treeTime);
        hrs = floor(tTime/3600);
        tTime = tTime - hrs * 3600;
        mins = floor(tTime/60);
        tTime = tTime - mins * 60;
        fprintf('Total time= %2i hrs, %2i min, %4.2f sec \n', hrs, mins, ...
            tTime);
    end
end