function [phi, psi ,tree] = asdHBTucker2(x,L,gam)
    %performs 3-mode condition probablility Bayesian Tucker decomposition 
    %on a counting tensor
    %P(mode 2, mode 3 | mode 1)
    % = P(mode 2 | topic 2) P(mode 3| topic 3) P(topic 2, topic 3| mode 1)
    %x = counting tensor to be decomposed
    %L = levels of hierarchical trees
    %gam = hyper parameter(s) of CRP
    
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
    samples=zeros(l1NormX,5+sum(L));
    s=find(x>0); %find nonzero elements
    v=x(s(:,1),s(:,2),s(:,3)); %get nonzero values
    v=v.vals; %extract values
    samples(:,1:3)=repelem(s,v,1); %set samples
    [~,xStarts,~]=unique(samples(:,1)); %find starting value of Xs
    
    %initialize tree
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
           for k=2:L(i)
               %subset to get samples in restaurant
               s=s(s(:,col+k-1)==curRes,:);
               
               [~,ir,~]=unique(s(:,1)); %get rows with unique x's
               
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
               
               samples(samples(:,1)==j,col+k)=label; %sit at table
           end
       end
    end
    
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
    psi=cell(2,1); %initialize
    for i=2:3
        %draw values from dirichlet distribution with uniform prior
        psi{i-1}=drchrnd(repelem(1/dims(i),dims(i)),coreDims(i))';
    end
    
    [~,ir,~]=unique(samples(:,1)); %get rows with unique x's
    s=samples(ir,:); %subset
    for i=1:dims(1)
        %draw core tensor p(z|x)
        phi(i,:,:)=drawCoreUni(s(i,:),coreDims,L,r);
    end
    
    %draw latent topic z's
    samples=drawZsc(samples,phi,psi,r);
    
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
        
        %redraw core tensor p(z|x)
        %subset to get samples with x
        [~,~,ir]=unique(samples(:,1));
        samps=accumarray(ir,1:size(samples,1),[],@(r){samples(r,:)});
        for i=1:dims(1)
            %redraw core tensor p(z|x)
            phi(i,:,:)=drawCoreCon(samps{i},coreDims,L,r);
        end

        %redraw latent topic z's
        samples=drawZsc(samples,phi,psi,r);
        
        %redraw tree
        [samples,tree,r] = redrawTree(dims,samples,L,tree,r,gam,xStarts);
        
        
        nIter=nIter+1;
        if nIter>=10
            cont=0;
        end
    end
end

% function samp = drawZs(samp,phi,psi,r)
%     %draws latent topic z's
%     %samp = sample matrix
%     %phi = tucker decomposition tensor core tensor
%     %psi = tucker decomposition matrices
%     %r = restaurant list
%     
%     for i=1:size(samp,1)
%     	samp(i,:)=drawZc(samp(i,:),phi,psi,r);
%     end
% end