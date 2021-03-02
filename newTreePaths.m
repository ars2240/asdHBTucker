function nPaths = newTreePaths(asdTens,ocpsi,ctree,paths,tree,ind,options)
    
    L=options.L;
    dims=size(asdTens);
    modes=length(dims)-1;  %number of dependent modes

    %adjustment if using constant L across dims
    if length(L)==1
        L=repelem(L,modes);
    end

    nPaths=ones(sum(ind),sum(L));
    
    if strcmp(options.topicType,'CP')
        for i=1:sum(ind)
           curRes=1; %set current restaurant as root

            for k=2:L(1)
                lp=0;
                
                %restaurant list
                rList=tree{curRes};
                rList=sort(rList);

                %compute CRP part of pdf
                pdf=histc(paths(:,k)',rList);

                for j=1:modes
                    %get counts
                    ctsA=ocpsi{j};
                    cts=ctree{j};
                    if options.sparse==1
                       subs=cts.subs;
                       vals=cts.vals;
                       [~,start,~]=unique(subs(:,1));
                       start=[start; nnz(cts)+1];
                    else
                       cts=permute(cts,[2:(modes+1),1]);
                    end

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

                    %get counts
                    cts1=ctsA(:,rList);
                    if options.sparse==1
                       cts2=ctsA(:,rList);
                       tsubs=subs(start(i):(start(i+1)-1),:);
                       tvals=vals(start(i):(start(i+1)-1));
                       [incl,tsubs(:,3)]=ismember(tsubs(:,3),rList);
                       if sum(incl)>0
                           tsubs=tsubs(incl,:);
                           tvals=tvals(incl);
                           tsubs=sub2ind(size(cts1), tsubs(:,2), tsubs(:,3));
                           cts2(tsubs)=cts2(tsubs)+tvals;
                       end
                    else
                       cts2=ctsA(:,rList)+cts(:,rList,i);
                    end
                    gcts2=gcts(:,rList);

                    %compute contribution to pdf
                    pdf = getPDF(pdf, rList, cts1, cts2, gcts2, prior);
                    lp = log(pdf) + lp;
                end
                
                %pick new table
                next=multi(exp(lp));
                curRes=rList(next);

                nPaths(i,k)=curRes; %sit at table
            end
        end
        nPaths=repmat(nPaths(:,1:L(1)),1,modes);
    else
        for j=1:modes

            col=(j-1)*L(1); %starting column

            %get counts
            ctsA=ocpsi{j};
            cts=ctree{j};
            if options.sparse==1
               subs=cts.subs;
               vals=cts.vals;
               [~,start,~]=unique(subs(:,1));
               start=[start; nnz(cts)+1];
            else
               cts=permute(cts,[2:(modes+1),1]);
            end

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

            for i=1:sum(ind)
               curRes=1; %set current restaurant as root

                for k=2:L(j)

                   %restaurant list
                   rList=tree{j}{curRes};
                   rList=sort(rList);

                   %compute CRP part of pdf
                   pdf=histc(paths(:,col+k)',rList);

                   %get counts
                   cts1=ctsA(:,rList);
                   if options.sparse==1
                       cts2=ctsA(:,rList);
                       tsubs=subs(start(i):(start(i+1)-1),:);
                       tvals=vals(start(i):(start(i+1)-1));
                       [incl,tsubs(:,3)]=ismember(tsubs(:,3),rList);
                       if sum(incl)>0
                           tsubs=tsubs(incl,:);
                           tvals=tvals(incl);
                           tsubs=sub2ind(size(cts1), tsubs(:,2), tsubs(:,3));
                           cts2(tsubs)=cts2(tsubs)+tvals;
                       end
                   else
                       cts2=ctsA(:,rList)+cts(:,rList,i);
                   end
                   gcts2=gcts(:,rList);

                   %compute contribution to pdf
                   pdf = getPDF(pdf, rList, cts1, cts2, gcts2, prior);

                   %pick new table
                   next=multi(pdf);
                   curRes=rList(next);

                   nPaths(i,col+k)=curRes; %sit at table

                end
            end
        end
    end
end