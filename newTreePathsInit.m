function [nPaths,r] = newTreePathsInit(paths,tree,ind,L)
    
    nPaths=ones(length(ind),sum(L));
    modes=length(L); %number of dependent modes
    r=cell(modes,1);
    
    for j=1:modes
        
        col=(j-1)*L(1); %starting column
        
        r{j}=unique(paths(:,(1+sum(L(1:(j-1)))):sum(L(1:j))))';
       
       for i=1:length(ind)
           curRes=1; %set current restaurant as root

           for k=2:L(j)

               %add new restaurant to list
               rList=tree{j}{curRes};
               rList=sort(rList);

               %compute CRP part of pdf
               pdf=histc(paths(:,col+k)',rList);

               %compute contribution to pdf
               pdf=pdf/sum(pdf); %normalize

               %pick new table
               next=multi(pdf);
               curRes=rList(next);

               nPaths(i,col+k)=curRes; %sit at table
           end
       end
    end
end