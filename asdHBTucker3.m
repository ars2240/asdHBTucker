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
    
    rng('shuffle'); %seed RNG
    
    % store original tensor
    odims=size(x); %dimensions of tensor
    modes=length(odims)-1;  %number of dependent modes
    
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
        x=reshape(x,[length(ind{1}),odims(2:end)]);
        x=sptensor(x);
    end
    
    dims=size(x); %dimensions of tensor
    
    gam=options.gam;
    L=options.L;
    LL=0; %initialize log-likelihood
    ent=0; %initialize entropy
    
    %adjustment if using constant gam across dims
    if length(gam)==1
        gam=repelem(gam,modes);
    end
    
    %adjustment if using constant L across dims
    if length(L)==1
        L=repelem(L,modes);
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
            if nargout==8
                error('Incorrect number of outputs for this topic model.');
            end
            [paths,tree,r,LLtree,entTree]=initializeTree(L,dims,gam);
        case 'PAM'
            [paths,tpl,tree,r,LLtree,entTree]=initializePAM(dims,options);
        case 'None'
            levels=ones(1,sum(L));
            r=cell(modes,1); %initialize
            tree=cell(modes,1); %initialize
            for i=1:modes
                levels((1+sum(L(1:(i-1)))):sum(L(1:i)))=1:L(i);
                r{i}=1:L(i);
            end
            paths=repmat(levels,dims(1),1);
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
        dimsM=zeros(modes,1);
        for i=1:modes
            dimsM(i)=length(r{i});
        end
        if options.sparse==0
            cphi=zeros([dims(1),dimsM]);
        else
            cphi=zeros([dims(1),L]);
        end
        cpsi=cell(modes,1);
        for i=1:modes
            cpsi{i}=zeros(dims(i+1),dimsM(i));
        end
        
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
        [cphi,cpsi,~] = counts(samples, dims, r, paths, [1,1,0], options);
        cTime=toc(cStart);
        
        matTime=0;
        coreTime=0;
    else
        coreDims=coreSize(modes, dims, r);

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
        [phi,p]=drawCoreUni(paths,coreDims,options);
        if ndims(phi) < 3
            phi(end, end, 2) = 0; 
        end
        LL=LL+sum(p);
        ent=ent+entropy(exp(p));
        coreTime=toc(coreStart);
        
        %draw latent topic z's
        zStart=tic;
        switch options.sparse
            case 0
                switch options.par
                    case 1
                        [samples,p]=drawZscPar(samples,phi,psi,r);
                    otherwise
                        [samples,p]=drawZsc(samples,phi,psi,r);
                end
            otherwise
                switch options.par
                    case 1
                        [samples,p] = drawZscSparsePar(samples,phi,psi,...
                            paths,L);
                    otherwise
                        [samples,p] = drawZscSparse(samples,phi,psi,...
                            paths,L);
                end
        end
        LL=LL+sum(p);
        ent=ent+entropy(exp(p));
        zTime=toc(zStart);
        
        cTime=0;
    end
    
    if options.print==1
        if options.collapsed==1
            [psi,~,~]=drawpsi(dims, modes, samples, r, options);
        end
        LL2=logLikelihood(x, x, 1, 1/(size(x,2)*size(x,3)), psi, paths, ...
            tree, options);
        fileID = fopen('verbose.txt','w');
        output_header=sprintf('%6s %13s %13s %10s', 'iter',...
            'loglikelihood', 'mixture LL', 'entropy');
        fprintf(fileID,'%s\n',output_header);
        fprintf(fileID,'%6i %13.2e %13.2e %10.2e\n',0,LL,LL2,ent);
        fclose(fileID);
    end
    
    %gibbs sampler
    cont=1;
    nIter=0;
    while cont==1
        LL=0; %reset log-likelihood
        ent=0; %reset entropy
        
        for btIt=1:options.btReps
            if options.collapsed==1 && nIter<(options.maxIter-1)
                for i=1:modes
                    pad=max(r{i})-size(cphi,i+1);
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
                     ind{1+i}=r{i};
                   end
                   ind{1}=1:size(cphi,1);
                   cphi=cphi(tensIndex2(ind,size(cphi)));
                end
                for i=1:modes
                    cpsi{i}=cpsi{i}(:,r{i});
                end
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
                coreDims=coreSize(modes, dims, r);

                %redraw matrices p(y|z)
                matStart=tic;
                [psi,LLp,entp]=drawpsi(dims, modes, samples, r, options);
                if btIt==options.btReps
                    LL=LL+LLp;
                    ent=ent+entp;
                end
                matTime=matTime+toc(matStart);

                %redraw core tensor p(z|x)
                %subset to get samples with x
                coreStart=tic;
                %redraw core tensor p(z|x)
                [phi,p]=drawCoreCon(samples,paths,coreDims,r,options);
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
        
        %redraw tree
        if nIter<(options.maxIter-1)
            
            %new counts
            cStart=tic;
            switch options.collapsed
                case 1
                    [cphi,cpsi,ctree] = counts(samples, dims, r, paths, options);
                otherwise
                    [~,cpsi,ctree] = counts(samples, dims, r, paths, [0,1,1], options);
            end
            cTime=cTime+toc(cStart);
            
            treeStart=tic;
            for treeIt=1:options.treeReps
                switch options.topicModel
                    case 'IndepTrees'
                        [paths,tree,r,LLtree,entTree]=redrawTree(dims,...
                            cpsi,ctree,paths,L,tree,r,options);
                    case 'PAM'
                        [paths,tree,prob,LLtree,entTree]=redrawPAM(dims,...
                            cpsi,ctree,paths,tpl,tree,L,options);
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
                if options.collapsed==1 && nIter<(options.maxIter-1)
                    [psi,~,~]=drawpsi(dims, modes, samples, r, options);
                end
                LL2=logLikelihood(x, x, 1, 1/(size(x,2)*size(x,3)), ...
                    psi, paths, tree, options);
                fileID = fopen('verbose.txt','a');
                fprintf(fileID,'%6i %13.2e %13.2e %10.2e\n',...
                    nIter, LL, LL2, ent);
                fclose(fileID);
            end
        end
        
        %check if to continue
        if nIter>=options.maxIter
            cont=0;
        end
    end
    tTime=toc(tStart);
    
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
        phiT=sptensor([],[],coreDims);
        for i=1:coreDims(1)
            res=cell(modes,1);
            for j=1:modes
                res{j}=paths(i,(1+sum(L(1:(j-1)))):sum(L(1:j)));
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
    phi=zeros([odims(1),sph(2:end)]);
    phi(~zind,:,:)=phio;
    
    if nargout==7
        varargout{1}=LL;
        varargout{2}=ms;
    elseif nargout==8
        varargout{1}=prob;
        varargout{2}=LL;
        varargout{3}=ms;
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