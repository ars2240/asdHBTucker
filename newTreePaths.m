function nPaths = newTreePaths(asdTens,samples,paths,tree,ind,options)

    L=options.L;
    
    %adjustment if using constant L across dims
    if length(L)==1
        L=repelem(L,2);
    end
    
    nPaths=ones(sum(ind),sum(L));
    
    r=cell(2,1);
    
    dims=size(asdTens);
    
    %convert from logical to actual
    ind=find(ind);
    
    for j=1:2
        
        col=(j-1)*L(1); %starting column
        
        r{j}=unique(paths((col+1):(col+L(j))));
        
       %get counts
       ctsA=accumarray(samples(:,[1+j 3+j]),1,[dims(1+j),...
           max(max(r{j}),max(samples(:,3+j)))]);
       
       for i=1:length(ind)
           curRes=1; %set current restaurant as root
           
           t=tenmat(full(asdTens(ind(i),:,:)),1,2);
           t=t(:,:);
           cts=max(t,[],mod(j,2)+1);
           if j==2
               cts=cts';
           end

           for k=2:L(j)

               %add new restaurant to list
               rList=tree{j}{curRes};
               rList=sort(rList);

               %compute CRP part of pdf
               pdf=histc(paths(:,col+k)',rList);

               %get counts
               cts1=ctsA(:,rList);
               cts2=ctsA(:,rList)+cts;

               %compute contribution to pdf
               pdf=log(pdf); %take log to prevent overflow
               pdf=pdf+gammaln(sum(cts1,1)+1);
               pdf=pdf-sum(gammaln(cts1+1/dims(1+j)),1);
               pdf=pdf+sum(gammaln(cts2+1/dims(1+j)),1);
               pdf=pdf-gammaln(sum(cts2,1)+1);
               pdf=pdf-mean(pdf);
               pdf=exp(pdf);
               pdf=pdf/sum(pdf); %normalize

               %pick new table
               next=multi(pdf);
               nextRes=rList(next);

               nPaths(i,col+k)=nextRes; %sit at table
           end
       end
    end
end