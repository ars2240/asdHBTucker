function [nPaths,r] = newTreePathsInit(paths,tree,ind,options)
    
    L=options.L;
    nPaths=ones(sum(ind),sum(L));
    modes=length(L); %number of dependent modes
    
    if strcmp(options.topicType,'CP')
        r=unique(paths(:,1:L(1)))';

        for i=1:sum(ind)
           curRes=1; %set current restaurant as root

           for k=2:L(1)

               %add new restaurant to list
               rList=tree{curRes};
               rList=sort(rList);

               %compute CRP part of pdf
               pdf=histc(paths(:,k)',rList);

               %compute contribution to pdf
               pdf=pdf/sum(pdf); %normalize

               %pick new table
               next=multi(pdf);
               curRes=rList(next);

               nPaths(i,k)=curRes; %sit at table
           end
        end
        nPaths=repmat(nPaths(:,1:L(1)),1,modes);
    else
        r=cell(modes,1);

        for j=1:modes

            col=(j-1)*L(1); %starting column

            r{j}=unique(paths(:,(1+sum(L(1:(j-1)))):sum(L(1:j))))';

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
end