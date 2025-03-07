function [xMat, xtMat] = asdGeneSelectCV2(xSparse, y, t, ind, b, test)
    %performs logistic regression on genes to reduce tensor size
    %xSparse = sparse matrix to be turned into tensor
    %t = p-value threshold
    
    xM=sptensor(xSparse(:,1:2),xSparse(:,3));
    xM=full(tenmat(xM,1));
    xM=xM(find(ind),:);
    xt=xM(find(b),:);
    xM=xM(find(~b),:);
    nGenes = size(xM,2);

    pval=ones(1,nGenes);

    switch test
        case 'logistic regression'
            for i=1:nGenes
                %logistic regression
                [~,~,stats]=glmfit(xM(:,i),y,'binomial');
                pval(i)=stats.p(2);
            end
        case 'exact'
            for i=1:nGenes
                %exact test
                ct = crosstab(xM(:,i)~=0,y);
                if size(ct,1)~=2
                    pval(i)=1;
                else
                    [~,pval(i),~] = fishertest(ct);
                end
            end
        case 'backward'
            maxdev = chi2inv(1-t,1);     
            opt = statset('display','iter',...
              'TolFun',maxdev,...
              'TolTypeFun','abs');
            inmodel = sequentialfs(@critfun,xM,y,...
               'cv','none',...
               'nullmodel',true,...
               'options',opt,...
               'direction','backward');
            pval(inmodel)=0;
        otherwise
            error('Error. \nNo text selected');
    end

    ind = pval<t;
    fprintf('Threashold= %5.4f\n',t);
    fprintf('Number of Genes in Threshold= %5d\n',sum(ind));
    
    %subset x by genes in threashold
    xM = xM(:,find(ind));
    xt = xt(:,find(ind));
    
    xMat=tenmat(xM,1); %flatten tensor to matrix
    xMat=xMat(:,:); %convert to matrix
    i=sum(xMat,1)>0;
    xMat=xMat(:,i); %remove columns of all zeros
    
    xtMat=tenmat(xt,1); %flatten tensor to matrix
    xtMat=xtMat(:,:); %convert to matrix
    xtMat=xtMat(:,i); %remove columns of all zeros
    
end

function dev = critfun(X,Y)
    model = fitglm(X,Y,'Distribution','binomial');
    dev = model.Deviance;
end
