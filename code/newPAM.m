function [paths,LL,ent] = newPAM(dims,ocpsi,ctree,paths,tpl,prob,options)
    %dims = dimensions of tensor
    %oSampless = x, y, z values
    %paths = tree paths
    %oPaths = old tree paths
    %tpl = number of topics per level of PAM
    %prob = probability tree
    %L = levels of hierarchical tree
    
    L=options.L;
    LL=0; %initialize log-likelihood
    ent=0; %initialize entropy
    modes=length(dims)-1;  %number of dependent modes
    
    %adjustment if using constant L across dims
    if length(L)==1
        L=repelem(L,modes);
    end
    
    cts=cell(modes,1); ctsA=cell(modes,1);
    gcts=cell(modes,1); subs=cell(modes,1);
    vals=cell(modes,1); start=cell(modes,1); prior=cell(modes,1);
    for j=1:modes
       %get counts
       cts{j}=ctree{j};
       if options.sparse==0 || ~issparse(cts{j})
           cts{j}=permute(cts{j},[2:3,1]);
       else
           if ~issparse(cts{j})
               cts{j}=sparse(cts{j});
           end
           subs{j}=cts{j}.subs;
           vals{j}=cts{j}.vals;
           [~,start{j},~]=unique(subs{j}(:,1));
           start{j}=[start{j}; nnz(cts{j})+1];
       end
       ctsA{j}=ocpsi{j};
       
       switch options.pType
            case 0
               prior{j}=1/dims(1+j);
            case 1
               prior{j}=1;
            case 2
               prior{j}=2/dims(1+j);
            otherwise
               error('Error. \nNo prior type selected');
        end

        gcts{j} = gammaln(ctsA{j}+prior{j});
    end
    
    for p=1:dims(1)
        res=1;
        
        %mode 2:modes, level 1
        j=1;
        for i=2:modes
            %get restaurant list
            rStart=sum(tpl{i}(1:(j-1)))+1;
            rList=rStart:sum(tpl{i}(1:j));
            pdf=prob{mod(i-2,modes)+1,j-(i==1)}(res,:);

            %get counts
            cts1=ctsA{i}(:,rList);
            if options.sparse==0
                cts2=ctsA{i}(:,rList)+cts{i}(:,rList,p);
            else
                cts2=ctsA{i}(:,rList);
                if ~isempty(start{i})
                    tsubs=subs{i}(start{i}(i):(start{i}(i+1)-1),:);
                    tvals=vals{i}(start{i}(i):(start{i}(i+1)-1));
                    [incl,tsubs(:,3)]=ismember(tsubs(:,3),rList);
                    if sum(incl)>0
                        tsubs=tsubs(incl,:);
                        tvals=tvals(incl);
                        tsubs=sub2ind(size(cts2), tsubs(:,2), tsubs(:,3));
                        cts2(tsubs)=cts2(tsubs)+tvals;
                    end
                end
            end

            gcts2=gcts{i}(:,rList);

            %compute contribution to pdf
            pdf = getPDF(pdf, rList, cts1, cts2, gcts2, prior{i});

            %pick new table
            res=multi(pdf);
            top=res+rStart-1;
            paths(p,j+(i-1)*L(1))=top;
            LL=LL+log(pdf(res));
            ent=ent+entropy(pdf(res));
        end
        
        for j=2:L(1)
            for i=1:2
                %get restaurant list
                rStart=sum(tpl{i}(1:(j-1)))+1;
                rList=rStart:sum(tpl{i}(1:j));
                pdf=prob{mod(i-2,modes)+1,j-(i==1)}(res,:);
                
                %get counts
                cts1=ctsA{i}(:,rList);
                if options.sparse==0
                    cts2=ctsA{i}(:,rList)+cts{i}(:,rList,p);
                else
                    cts2=ctsA{i}(:,rList);
                    if ~isempty(start{i})
                        tsubs=subs{i}(start{i}(i):(start{i}(i+1)-1),:);
                        tvals=vals{i}(start{i}(i):(start{i}(i+1)-1));
                        [incl,tsubs(:,3)]=ismember(tsubs(:,3),rList);
                        if sum(incl)>0
                            tsubs=tsubs(incl,:);
                            tvals=tvals(incl);
                            tsubs=sub2ind(size(cts2), tsubs(:,2), tsubs(:,3));
                            cts2(tsubs)=cts2(tsubs)+tvals;
                        end
                    end
                end
                gcts2=gcts{i}(:,rList);

                %compute contribution to pdf
                pdf = getPDF(pdf, rList, cts1, cts2, gcts2, prior{i});
                
                %pick new table
                res=multi(pdf);
                top=res+rStart-1;
                paths(p,j+(i-1)*L(1))=top;
                LL=LL+log(pdf(res));
                ent=ent+entropy(pdf(res));
            end
        end
    end

end
