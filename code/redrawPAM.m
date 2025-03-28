function [paths,prob,varargout] = redrawPAM(dims,cpsi,ctree,paths,tpl,prob,L,options)
    %dims = dimensions of tensor
    %sampless = x, y, z values
    %paths = tree paths
    %tpl = number of topics per level of PAM
    %prob = probability tree
    %L = levels of hierarchical tree

    LL=0; %initialize log-likelihood
    ent=0; %initialize entropy

    modes=length(L); %number of dependent modes

    if options.collapsed==1
        %i=1:(modes-1);
        for i=1:(modes-1)
            for j=1:L(1)
                len=tpl{i+1}(j);
                parents=tpl{i}(j);
                switch options.pType
                    case 0
                        prior=1/len*ones(len,parents);
                    case 1
                        prior=ones(len,parents);
                    case 2
                        prior=2/len*ones(len,parents);
                    otherwise
                        error('Error. \nNo prior type selected');
                end
                si=[sum(tpl{i}(1:j)),sum(tpl{i+1}(1:j))];
                cts=accumarray(paths(:,[sum(L(1:(i-1)))+j,...
                    sum(L(1:i))+j]),1,si);
                cts=cts((size(cts,1)-parents+1):size(cts,1),...
                    (size(cts,2)-len+1):size(cts,2));
                prob{i,j}=prior+cts;
                prob{i,j}=prob{i,j}./sum(prob{i,j},2);
            end
        end

        %i=modes;
        for j=1:(L(1)-1)
            len=tpl{1}(j+1);
            parents=tpl{modes}(j);
            switch options.pType
                case 0
                    prior=1/len*ones(len,parents);
                case 1
                    prior=ones(len,parents);
                case 2
                    prior=2/len*ones(len,parents);
                otherwise
                    error('Error. \nNo prior type selected');
            end
            si=[sum(tpl{modes}(1:j)),sum(tpl{1}(1:(j+1)))];
            cts=accumarray(paths(:,[sum(L(1:(modes-1)))+j,j+1]),1,si);
            cts=cts((size(cts,1)-parents+1):size(cts,1),...
                (size(cts,2)-len+1):size(cts,2));
            prob{modes,j}=prior+cts;
            prob{modes,j}=prob{modes,j}./sum(prob{modes,j},2);
        end

    else
        %i=1:(modes-1);
        for i=1:(modes-1)
            for j=1:L(1)
                len=tpl{i+1}(j);
                parents=tpl{i}(j);
                switch options.pType
                    case 0
                        prior=1/len*ones(parents,len);
                    case 1
                        prior=ones(parents,len);
                    case 2
                        prior=2/len*ones(parents,len);
                    otherwise
                        error('Error. \nNo prior type selected');
                end
                si=[sum(tpl{i}(1:j)),sum(tpl{i+1}(1:j))];
                cts=accumarray(paths(:,[sum(L(1:(i-1)))+j,...
                    sum(L(1:i))+j]),1,si);
                cts=cts((size(cts,1)-parents+1):size(cts,1),...
                    (size(cts,2)-len+1):size(cts,2));
                [prob{i,j},p]=drchrnd(prior+cts,parents,options);
                prob{i,j}=prob{i,j}./sum(prob{i,j},2);
                LL=LL+sum(log(p));
                ent=ent+entropy(p);
            end
        end

        %i=modes;
        for j=1:(L(1)-1)
            len=tpl{1}(j+1);
            parents=tpl{2}(j);
            switch options.pType
                case 0
                    prior=1/len*ones(len,parents);
                case 1
                    prior=ones(len,parents);
                case 2
                    prior=2/len*ones(len,parents);
                otherwise
                    error('Error. \nNo prior type selected');
            end
            si=[sum(tpl{modes}(1:j)),sum(tpl{1}(1:(j+1)))];
            cts=accumarray(paths(:,[sum(L(1:(modes-1)))+j,j+1]),1,si);
            cts=cts((size(cts,1)-parents+1):size(cts,1),...
                (size(cts,2)-len+1):size(cts,2));
            [prob{modes,j},p]=drchrnd(prior+cts,parents,options);
            prob{modes,j}=prob{modes,j}./sum(prob{modes,j},2);
            LL=LL+sum(log(p));
            ent=ent+entropy(p);
        end

    end

    cts=cell(modes,1); ctsA=cell(modes,1);
    gcts=cell(modes,1); subs=cell(modes,1);
    vals=cell(modes,1); start=cell(modes,1); prior=cell(modes,1);
    for j=1:modes
        %get counts
        cts{j}=ctree{j};
        if options.sparse==1
           subs{j}=cts{j}.subs;
           vals{j}=cts{j}.vals;
           [~,start{j},~]=unique(subs{j}(:,1));
           start{j}=[start{j}; nnz(cts{j})+1];
        else
           cts{j}=permute(cts{j},[2:(modes+1),1]);
        end
        ctsA{j}=cpsi{j};

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

        %modes 2:modes, level 1
        j=1;
        for i=2:modes
            
            %get restaurant list
            rStart=sum(tpl{i}(1:(j-1)))+1;
            rList=rStart:sum(tpl{i}(1:j));
            pdf=prob{mod(i-2,modes)+1,j-(i==1)}(res,:);

            %get counts
            if options.sparse==1
                cts1=ctsA{i}(:,rList);
                tsubs=subs{i}(start{i}(i):(start{i}(i+1)-1),:);
                tvals=vals{i}(start{i}(i):(start{i}(i+1)-1));
                [incl,tsubs(:,3)]=ismember(tsubs(:,3),rList);
                if sum(incl)>0
                    tsubs=tsubs(incl,:);
                    tvals=tvals(incl);
                    tsubs=sub2ind(size(cts1), tsubs(:,2), tsubs(:,3));
                    cts1(tsubs)=cts1(tsubs)-tvals;
                end
            else
                cts1=ctsA{i}(:,rList)-cts{i}(:,rList,p);
            end
            cts2=ctsA{i}(:,rList);
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
            for i=1:modes
                
                %get restaurant list
                rStart=sum(tpl{i}(1:(j-1)))+1;
                rList=rStart:sum(tpl{i}(1:j));
                pdf=prob{mod(i-2,modes)+1,j-(i==1)}(res,:);

                %get counts
                if options.sparse==1
                    cts1=ctsA{i}(:,rList);
                    tsubs=subs{i}(start{i}(i):(start{i}(i+1)-1),:);
                    tvals=vals{i}(start{i}(i):(start{i}(i+1)-1));
                    [incl,tsubs(:,3)]=ismember(tsubs(:,3),rList);
                    if sum(incl)>0
                        tsubs=tsubs(incl,:);
                        tvals=tvals(incl);
                        tsubs=sub2ind(size(cts1), tsubs(:,2), tsubs(:,3));
                        cts1(tsubs)=cts1(tsubs)-tvals;
                    end
                else
                    cts1=ctsA{i}(:,rList)-cts{i}(:,rList,p);
                end
                cts2=ctsA{i}(:,rList);
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

    if nargout==4
        varargout{1}=LL;
        varargout{2}=ent;
    elseif nargout==5
        varargout{1}=prob;
        varargout{2}=LL;
        varargout{3}=ent;
    else
        error("Error. \nIncorrect number of outputs.");
    end

end
