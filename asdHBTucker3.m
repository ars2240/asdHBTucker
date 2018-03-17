function [phi, psi ,tree] = asdHBTucker3(x,L,gam,options)
    %performs 3-mode condition probablility Bayesian Tucker decomposition 
    %on a counting tensor
    %P(mode 2, mode 3 | mode 1)
    % = P(mode 2 | topic 2) P(mode 3| topic 3) P(topic 2, topic 3| mode 1)
    %x = counting tensor to be decomposed
    %L = levels of hierarchical trees
    %gam = hyper parameter(s) of CRP
    %options = 
    % options.par = whether or not z's are computed in parallel
    % options.time = whether or not time is printed
    
    tStart=tic;
    dims=size(x); %dimensions of tensor
    
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
    samples=zeros(l1NormX,5+sum(L));
    s=find(x>0); %find nonzero elements
    v=x(s(:,1),s(:,2),s(:,3)); %get nonzero values
    v=v.vals; %extract values
    samples(:,1:3)=repelem(s,v,1); %set samples
    [~,xStarts,~]=unique(samples(:,1)); %find starting value of Xs
    xEnds = [xStarts(2:dims(1))-1;size(samples,1)];
    sampTime=toc(sStart);
    
    %initialize tree
    treeStart=tic;
    samples(:,6)=1; %sit at root table
    samples(:,6+L(1))=1; %sit at root table
    tree=cell(2,1); %initialize
    r=cell(2,1); %initialize
    for i=1:2
       tree{i}{1}=[];
       r{i}=1;
       col=5+(i-1)*L(1);
       for j=1:dims(1)
           s=samples;
           curRes=1; %restaurant
           ir=xStarts;
           for k=2:L(i)
               
               %get count of customers at table
               tab=histc(s(ir,col+k)',tree{i}{curRes});
               
               new=0; %set boolean for new table to false
               
               %check to see if nonzero number of customers
               if sum(tab)~=0
                   nT=crp(tab,gam(i)); %draw new table
                   
                   %check to see if table is unoccupied
                   if nT>length(tab)
                       new=1;
                   end
               else
                   new=1; %if restaurant is unoccupied, create new table
               end
               
               %create new table if table is unoccupied
               if new==1
                   %get label of new table
                   label=find(~ismember(1:max(r{i}+1),r{i}),1);
                   r{i}=[r{i} label];
                   
                   %add new table to tree
                   tree{i}{curRes}=[tree{i}{curRes} label];
                   tree{i}{label}=[]; %add new table to tree
               else
                   label=tree{i}{curRes}(nT); %get label of table
               end
               
               samples(xStarts(j):xEnds(j),col+k)=label; %sit at table
               
               curRes=label; %update restaurant
               
               if k~=L(i)
                   %get indices of samples in next restaurant
                   %and subset data sets
                   sub=s(ir,col+k)==label;
                   irEnd=[ir(2:size(ir))-1;size(ex,1)];
                   x=1:size(ir);
                   x=x(sub);
                   pos=elems(ir(x),irEnd(x));
                   s=s(pos,:);
                   [~,ir,~]=unique(s(:,1));
               end
           end
       end
    end
    treeTimeInit=toc(treeStart);
    treeTime=0;
    
    %calculate dimensions of core
    coreDims=zeros(1,3);
    coreDims(1)=dims(1);
    for i=1:2
       %set core dimensions to the number of topics in each mode
       coreDims(i+1)=length(r{i});
    end
    
    %initialize tucker decomposition
    %core tensor
    phi=zeros(coreDims(1),coreDims(2),coreDims(3));
    
    %draw matrices p(y|z)
    matStart=tic;
    psi=cell(2,1); %initialize
    for i=2:3
        %draw values from dirichlet distribution with uniform prior
        psi{i-1}=drchrnd(repelem(1/dims(i),dims(i)),coreDims(i))';
    end
    matTime=toc(matStart);
    
    coreStart=tic;
    s=samples(xStarts,:); %subset
    for i=1:dims(1)
        %draw core tensor p(z|x)
        phi(i,:,:)=drawCoreUni(s(i,:),coreDims,L,r);
    end
    coreTime=toc(coreStart);
    
    %save('asd.mat','phi','psi','r','samples');
    %draw latent topic z's
    zStart=tic;
    switch options.par
        case 1
            samples=drawZscPar(samples,phi,psi,r);
        otherwise
            samples=drawZsc(samples,phi,psi,r);
    end
    zTime=toc(zStart);
    
    %gibbs sampler
    cont=1;
    nIter=0;
    while cont==1
        %recalculate dimensions of core
        for i=1:2
           %set core dimensions to the number of topics in each mode
           coreDims(i+1)=length(r{i});
        end

        %reinitialize tucker decomposition
        %core tensor
        phi=zeros(coreDims(1),coreDims(2),coreDims(3));
        %matrices
        psi{1}=zeros(dims(2),coreDims(2));
        psi{2}=zeros(dims(3),coreDims(3));

        %redraw matrices p(y|z)
        matStart=tic;
        for i=2:3
            [u,~,ir]=unique(samples(:,2+i));
            samps=accumarray(ir,1:size(samples,1),[],@(r){samples(r,:)});
            dim=dims(i);
            [~,loc]=ismember(r{i-1},u);
            psiT=zeros(dim,coreDims(i));
            for j=1:coreDims(i)
                %draw values from dirichlet distribution with uniform prior
                %plus counts of occurances of both y & z
                pdf=repelem(1/dim,dim);
                if loc(j)~=0
                    pdf=pdf+histc(samps{loc(j)}(:,i)',1:dim);
                end
                psiT(:,j)=drchrnd(pdf,1);
            end
            psi{i-1}=psiT;
        end
        matTime=matTime+toc(matStart);
        
        %redraw core tensor p(z|x)
        %subset to get samples with x
        coreStart=tic;
        [~,~,ir]=unique(samples(:,1));
        samps=accumarray(ir,1:size(samples,1),[],@(r){samples(r,:)});
        for i=1:dims(1)
            %redraw core tensor p(z|x)
            phi(i,:,:)=drawCoreCon(samps{i},coreDims,L,r);
        end
        coreTime=coreTime+toc(coreStart);

        %redraw latent topic z's
        zStart=tic;
        switch options.par
            case 1
                samples=drawZscPar(samples,phi,psi,r);
            otherwise
                samples=drawZsc(samples,phi,psi,r);
        end
        zTime=zTime+toc(zStart);
        
        %redraw tree
        treeStart=tic;
        [samples,tree,r] = redrawTree(dims,samples,L,tree,r,gam,xStarts);
        treeTime=treeTime+toc(treeStart);
        
        nIter=nIter+1;
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
        fprintf('Tree Init time= %5.2f\n',treeTimeInit);
        fprintf('Tree time= %5.2f\n',treeTime);
        fprintf('Total time= %5.2f\n',tTime);
    end
end