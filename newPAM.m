function [paths,LL,ent] = newPAM(dims,ocpsi,ctree,paths,tpl,prob,L)
    %dims = dimensions of tensor
    %oSampless = x, y, z values
    %paths = tree paths
    %oPaths = old tree paths
    %tpl = number of topics per level of PAM
    %prob = probability tree
    %L = levels of hierarchical tree

    LL=0; %initialize log-likelihood
    ent=0; %initialize entropy
    
    cts=cell(2,1);
    ctsA=cell(2,1);
    for j=1:2
       %get counts
       ctsA{j}=ocpsi{j};
       cts{j}=permute(ctree{j},[2,3,1]);
    end
    
    for p=1:dims(1)
        res=1;
        
        %mode 2, level 1
        j=1;
        i=2;
        
        %get restaurant list
        rStart=sum(tpl{i}(1:(j-1)))+1;
        rList=rStart:sum(tpl{i}(1:j));
        pdf=prob{mod(i,2)+1,j-(i==1)}(res,:);

        %get counts
        cts1=ctsA{i}(:,rList);
        cts2=ctsA{i}(:,rList)+cts{i}(:,rList,p);

        %compute contribution to pdf
        pdf=log(pdf); %take log to prevent overflow
        pdf=pdf+gammaln(sum(cts1,1)+1);
        pdf=pdf-sum(gammaln(cts1+1/dims(1+i)),1);
        pdf=pdf+sum(gammaln(cts2+1/dims(1+i)),1);
        pdf=pdf-gammaln(sum(cts2,1)+1);
        pdf=exp(pdf);
        pdf=pdf/sum(pdf); %normalize

        %pick new table
        res=multi(pdf);
        top=res+rStart-1;
        paths(p,j+(i==2)*L(1))=top;
        LL=LL+log(pdf(res));
        ent=ent+entropy(pdf(res));
        
        for j=2:L(1)
            for i=1:2
                %get restaurant list
                rStart=sum(tpl{i}(1:(j-1)))+1;
                rList=rStart:sum(tpl{i}(1:j));
                pdf=prob{mod(i,2)+1,j-(i==1)}(res,:);
                
                %get counts
                cts1=ctsA{i}(:,rList);
                cts2=ctsA{i}(:,rList)+cts{i}(:,rList,p);
                
                %compute contribution to pdf
                pdf=log(pdf); %take log to prevent overflow
                pdf=pdf+gammaln(sum(cts1,1)+1);
                pdf=pdf-sum(gammaln(cts1+1/dims(1+i)),1);
                pdf=pdf+sum(gammaln(cts2+1/dims(1+i)),1);
                pdf=pdf-gammaln(sum(cts2,1)+1);
                pdf=exp(pdf);
                pdf=pdf/sum(pdf); %normalize
                
                %pick new table
                res=multi(pdf);
                top=res+rStart-1;
                paths(p,j+(i==2)*L(1))=top;
                LL=LL+log(pdf(res));
                ent=ent+entropy(pdf(res));
            end
        end
    end

end