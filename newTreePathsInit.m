function [nPaths,r] = newTreePathsInit(paths,samples,tree,ind,L)
    
    nPaths=ones(sum(ind),sum(L));
    
    r=cell(2,1);
    
    for j=1:2
        
        col=(j-1)*L(1); %starting column
        
        r{j}=unique(samples(:,3+j));
       
       for i=1:sum(ind)
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