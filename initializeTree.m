function [paths,tree,r,LL,ent]= initializeTree(L,dims,gam)
    %Initializes hierarchical tree from the CRP
    %Inputs
    % L = levels of hierarchical tree
    % dims = dimensions of tensor
    % gam = hyper parameter of CRP
    %Outputs
    % path = tree paths
    % tree = hierarchical tree
    % r = restaurant lists
    % LL = log-likelihood
    % ent = entropy
    
    LL=0; %initialize log-likelihood
    ent=0; %initialize entropy
    modes=length(L); %number of dependent modes
    
    paths=zeros(dims(1),sum(L));
    for i=1:modes
        paths(:,1+sum(L(1:(i-1))))=1; %sit at root table
    end
    tree=cell(modes,1); %initialize
    r=cell(modes,1); %initialize
    for i=1:modes
       tree{i}{1}=[];
       r{i}=1;
       col=(i-1)*L(1);
       for j=randperm(dims(1))
           curRes=1; %restaurant
           for k=2:L(i)
               
               %get count of customers at table
               tab=histc(paths(:,col+k)',tree{i}{curRes});
               
               new=0; %set boolean for new table to false
               
               %check to see if nonzero number of customers
               if sum(tab)~=0
                   [nT, p]=crp(tab,gam(i)); %draw new table
                   LL=LL+log(p); %increment log-likelihood
                   ent=ent+entropy(p); %increment entropy
                   
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
               
               paths(j,col+k)=label; %sit at table
               
               curRes=label; %update restaurant
               
           end
       end
    end
end