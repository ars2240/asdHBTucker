function [phi, psi ,tree] = asdHBTuckerPar(x,L,gam)
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
    parfor i=1:dims(1)
        %draw core tensor p(z|x)
        phi(i,:,:)=drawCoreUni(s(i,:),coreDims,L,r);
    end
    
    %draw latent topic z's
    samples=drawZs(samples,phi,psi,r);
    
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
            parfor j=1:coreDims(i)
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
        parfor i=1:dims(1)
            %redraw core tensor p(z|x)
            phi(i,:,:)=drawCoreCon(samps{i},coreDims,L,r);
        end

        %redraw latent topic z's
        samples=drawZs(samples,phi,psi,r);
        
        %redraw tree
        for i=1:dims(1)
           %whether or not sample has patient=i
           b=samples(:,1)==i;
           
           for j=1:2
               curRes=1; %set current restaurant as root
               col=5+(j-1)*L(1); %starting column
               
               %all samples, excluding the one being resampled
               ex=samples(~b,:);
               s=samples;
               
               for k=2:L(j)
                   %get label of new table
                   newRes=find(~ismember(1:max(r{j}+1),r{j}),1);
                   
                   new=0; %set boolean for new table to false
                   
                   if ~isempty(tree{j}{curRes})
                       %add new restaurant to list
                       rList=[tree{j}{curRes} newRes];
                       [rList, order]=sort(rList);

                       %get number of samples in that restaurant
                       [~,ir,~]=unique(ex(:,1));

                       %compute CRP part of pdf
                       pdf=histc(ex(ir,col+k)',rList);

                       %get counts
                       cts1=accumarray(ex(:,[1+j col+k]),1,[dims(1+j) max(rList)]);
                       cts1=cts1(:,rList);
                       cts2=accumarray(s(:,[1+j col+k]),1,[dims(1+j) max(rList)]);
                       cts2=cts2(:,rList);

                       %compute contribution to pdf
                       pdf=log(pdf); %take long to prevent overflow
                       pdf=pdf+gammaln(sum(cts1,1)+1);
                       pdf=pdf-sum(gammaln(cts1+1/dims(1+j)),1);
                       pdf=pdf+sum(gammaln(cts2+1/dims(1+j)),1);
                       pdf=pdf-gammaln(sum(cts2,1)+1);
                       pdf=exp(pdf);
                       [~,l]=max(order);
                       pdf(l)=gam(j);
                       
                       %pick new table
                       nextRes=rList(multi(pdf));

                       ex=ex(ex(:,col+k)==nextRes,:);
                       s=s(s(:,col+k)==nextRes,:);
                       if nextRes==newRes
                           new=1;
                       end
                   else
                       new=1; %set boolean for new table to false
                       nextRes=newRes;
                   end
                   
                   samples(b,col+k)=nextRes; %sit at table
                   
                   %if new table
                   if new==1
                       r{j}=[r{j} newRes]; %add to restaurant list
                   
                       %add new table to tree
                       tree{j}{curRes}=[tree{j}{curRes} newRes];
                       tree{j}{newRes}=[]; %add new table to tree
                   end
                   
                   curRes=nextRes; %cycle restaurants
                   
               end
               
               %handle abandoned tables
               rList=unique(samples(:,(col+1):(col+L(j))));
               in=ismember(r{j},rList);
               r{j}=r{j}(in);
               r{j}=sort(r{j});
               for k=find(~in)
                  tree{j}{k}=[];
               end
               for k=1:max(r{j})
                  tree{j}{k}=tree{j}{k}(ismember(tree{j}{k},r{j}));
               end
           end
        end
        
        
        nIter=nIter+1;
        if nIter>=10
            cont=0;
        end
    end
end

function samp = drawZs(samp,phi,psi,r)
    %draws latent topic z's
    %samp = sample matrix
    %phi = tucker decomposition tensor core tensor
    %psi = tucker decomposition matrices
    %r = restaurant list
    
    parfor i=1:size(samp,1)
    	samp(i,:)=drawZ(samp(i,:),phi,psi,r);
    end
end