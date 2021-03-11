function [paths,tree,r,LL,ent] = redrawTree(dims,cpsi,ctree,paths,tree,r,options)
    %dims = dimensions of tensor
    %ctree = output from counts
    %paths = tree paths
    %L = levels of hierarchical tree
    %tree = hierarchical tree
    %r = restaurant lists
    %options = 
    % gam = hyper parameter(s) of CRP

    gam=options.gam; L=options.L;
    modes=length(L); %number of dependent modes
    
    %adjustment if using constant gam across dims
    if length(gam)==1
        gam=repelem(gam,modes);
    end
    
    LL=0; %initialize log-likelihood
    ent=0; %initialize entropy
    
    if strcmp(options.topicType,'CP')
        cts=cell(modes,1); start=cell(modes,1);
        subs=cell(modes,1); vals=cell(modes,1);
        for j=1:modes
           cts{j}=ctree{j};
           if options.sparse==1
               subs{j}=cts{j}.subs;
               vals{j}=cts{j}.vals;
               [s,start{j},~]=unique(subs{j}(:,1));
               start{j}=[start{j}; nnz(cts{j})+1];
           else
               cts{j}=permute(cts{j},[2:(modes+1),1]);
           end
        end

        for i=randperm(dims(1))
            curRes=1; %set current restaurant as root

            new=0; %set boolean for new table to false

            for k=2:L(1)
                %get label of new table
                newRes=find(~ismember(1:max(r+1),r),1);

                if ~isempty(tree{curRes})
                    %add new restaurant to list
                    rList=[tree{curRes} newRes];
                    [rList, order]=sort(rList);
                    
                    %compute CRP part of pdf
                    c=histc([paths(1:(i-1),k); paths((i+1):end,k)]',rList);
                    b = c>0 | rList == newRes;
                    rList=rList(b); c=c(b); order=order(b);
                    [~,l]=max(order);
                    c(l)=gam(j);
                    lp=log(c);
                    
                    for j=1:modes
                        %get counts
                        ctsA=cpsi{j};

                        switch options.pType
                           case 0
                               prior=1/dims(1+j);
                           case 1
                               prior=1;
                           case 2
                               prior=2/dims(1+j);
                           otherwise
                               error('Error. \nNo prior type selected');
                        end

                        gcts = gammaln(ctsA+prior);

                        if newRes>size(ctsA,2)
                           cts{j}(:,newRes,:)=0;
                           ctsA(:,newRes)=0;
                           gcts(:,newRes)=gammaln(prior);
                        end

                        %get counts
                        if options.sparse==1
                           cts1=ctsA(:,rList);
                           tsubs=subs{j}(start{j}(i):(start{j}(i+1)-1),:);
                           tvals=vals{j}(start{j}(i):(start{j}(i+1)-1));
                           [incl,tsubs(:,3)]=ismember(tsubs(:,3),rList);
                           if sum(incl)>0
                               tsubs=tsubs(incl,:);
                               tvals=tvals(incl);
                               tsubs=sub2ind(size(cts1), tsubs(:,2), tsubs(:,3));
                               cts1(tsubs)=cts1(tsubs)-tvals;
                           end
                        else
                           cts1=ctsA(:,rList)-cts{j}(:,rList,i);
                        end
                        cts2=ctsA(:,rList);
                        gcts2=gcts(:,rList);

                        %compute contribution to pdf
                        lp = logPDF(lp, rList, cts1, cts2, gcts2, prior);
                    end
                    
                    %pick new table
                    pdf=adjPDF(lp);
                    next=multi(pdf);
                    nextRes=rList(next);
                    p=pdf(next);
                    LL=LL+log(p); %increment log-likelihood
                    ent=ent+entropy(p); %increment entropy

                    if nextRes==newRes
                       new=1;
                    end
               else
                   new=1; %set boolean for new table to false
                   nextRes=newRes;
               end

               paths(i,k)=nextRes; %sit at table

               %if new table
               if new==1
                   r=[r newRes]; %add to restaurant list

                   %add new table to tree
                   tree{curRes}=[tree{curRes} newRes];
                   tree{newRes}=[]; %add new table to tree
               end

               curRes=nextRes; %cycle restaurants

           end

        end

        %handle abandoned tables
        rList=reshape(paths(:,1:L(1)),[],1);
        in=ismember(r,rList);
        r=r(in);
        r=sort(r);
        for k=find(~in)
          tree{k}=[];
        end
        for k=1:max(r)
          tree{k}=tree{k}(ismember(tree{k},r));
        end

        r=unique(paths(:,1:L(1)))';
        l=length(r);
        tN=cell(l,1);
        for j=1:l
            tN{j}=arrayfun(@(x) find(r==x,1),tree{r(j)});
        end
        paths(:,1:L(1))=arrayfun(@(x)find(r==x,1),paths(:,1:L(1)));
        r=1:l; tree=tN;
        paths=repmat(paths(:,1:L(1)),1,modes);
    else
        for j=1:modes
        
           %get counts
           ctsA=cpsi{j};
           cts=ctree{j};
           if options.sparse==1
               subs=cts.subs;
               vals=cts.vals;
               [s,start,~]=unique(subs(:,1));
               start=[start; nnz(cts)+1];
           else
               cts=permute(cts,[2:(modes+1),1]);
           end


           col=(j-1)*L(1); %starting column

           switch options.pType
               case 0
                   prior=1/dims(1+j);
               case 1
                   prior=1;
               case 2
                   prior=2/dims(1+j);
               otherwise
                   error('Error. \nNo prior type selected');
           end

           gcts = gammaln(ctsA+prior);

           for i=randperm(dims(1))
               curRes=1; %set current restaurant as root

               new=0; %set boolean for new table to false

               for k=2:L(j)
                   %get label of new table
                   newRes=find(~ismember(1:max(r{j}+1),r{j}),1);

                   if newRes>size(ctsA,2)
                       %cts=padarray(cts,[0 1 0],'post');
                       %ctsA=padarray(ctsA,[0 1],'post');
                       %gcts=padarray(gcts,[0 1],glp,'post');
                       cts(:,newRes,:)=0;
                       ctsA(:,newRes)=0;
                       gcts(:,newRes)=gammaln(prior);
                   end

                   if ~isempty(tree{j}{curRes})
                       %add new restaurant to list
                       rList=[tree{j}{curRes} newRes];
                       [rList, order]=sort(rList);

                       %compute CRP part of pdf
                       c=histc([paths(1:(i-1),col+k); ...
                           paths((i+1):end,col+k)]',rList);
                       b = c>0 | rList == newRes;
                       rList=rList(b); c=c(b); order=order(b);

                       %get counts
                       if options.sparse==1
                           cts1=ctsA(:,rList);
                           tsubs=subs(start(i):(start(i+1)-1),:);
                           tvals=vals(start(i):(start(i+1)-1));
                           [incl,tsubs(:,3)]=ismember(tsubs(:,3),rList);
                           if sum(incl)>0
                               tsubs=tsubs(incl,:);
                               tvals=tvals(incl);
                               tsubs=sub2ind(size(cts1), tsubs(:,2), tsubs(:,3));
                               cts1(tsubs)=cts1(tsubs)-tvals;
                           end
                       else
                           cts1=ctsA(:,rList)-cts(:,rList,i);
                       end
                       cts2=ctsA(:,rList);
                       gcts2=gcts(:,rList);

                       %compute contribution to pdf
                       [~,l]=max(order);
                       c(l)=gam(j);
                       pdf = getPDF(c, rList, cts1, cts2, gcts2, prior);

                       %pick new table
                       next=multi(pdf);
                       nextRes=rList(next);
                       p=pdf(next);
                       LL=LL+log(p); %increment log-likelihood
                       ent=ent+entropy(p); %increment entropy

                       if nextRes==newRes
                           new=1;
                       end
                   else
                       new=1; %set boolean for new table to false
                       nextRes=newRes;
                   end

                   paths(i,col+k)=nextRes; %sit at table

                   %if new table
                   if new==1
                       r{j}=[r{j} newRes]; %add to restaurant list

                       %add new table to tree
                       tree{j}{curRes}=[tree{j}{curRes} newRes];
                       tree{j}{newRes}=[]; %add new table to tree
                   end

                   curRes=nextRes; %cycle restaurants

               end

           end

           %handle abandoned tables
           rList=reshape(paths(:,(col+1):(col+L(j))),[],1);
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

        [r, tree, paths] = sortTopics(r, tree, paths, L);
    end
end