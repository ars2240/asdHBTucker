function [xMat, xtMat] = asdGeneSelectCV(xSparse, y, t, ind, b, test)
    %performs logistic regression on genes to reduce tensor size
    %xSparse = sparse matrix to be turned into tensor
    %t = p-value threshold
    
    x=sptensor(xSparse(:,1:3),xSparse(:,4));
    x=x(find(ind),:,:);
    x=x(find(b),:,:);
    xt=x(find(~b),:,:);
    
    xM=sptensor(xSparse(:,1:2),xSparse(:,4),max(xSparse(:,1:2)),@max);
    xM=full(tenmat(xM,1));
    xM=xM(find(ind),:);
    xM=xM(find(b),:);
    nGenes = size(xM,2);

    pval=ones(1,nGenes);

    for i=1:nGenes
        switch test
            case 'logistic regression'
                %logistic regression
                [~,~,stats]=glmfit(xM(:,i),y,'binomial');
                pval(i)=stats.p(2);
            case 'exact'
                %exact test
                %source: https://www.mathworks.com/matlabcentral/fileexchange/24379-fisher-s-exact-test-with-n-x-m-contingency-table
                [~,pval(i),~] = FisherExactTest(xM(:,i),y);
            otherwise
                error('Error. \nNo text selected');
        end
    end

    ind = pval<t;
    fprintf('Threashold= %5.4f\n',t);
    fprintf('Number of Genes in Threshold= %5d\n',sum(ind));
    
    %subset x by genes in threashold
    x = x(:,find(ind),:);
    xt = xt(:,find(ind),:);
    
    %remove pathways with 0 count
    xM = tenmat(x, 3);
    xM = xM(:,:);
    pCount=sum(xM,2);
    ind = pCount>0;
    fprintf('Number of Pathways= %5d\n',sum(ind));
    x = x(:,:,find(ind));
    xt = xt(:,:,find(ind));
    
    xMat=tenmat(x,1); %flatten tensor to matrix
    xMat=xMat(:,:); %convert to matrix
    i=sum(xMat,1)>0;
    xMat=xMat(:,i); %remove columns of all zeros
    
    xtMat=tenmat(xt,1); %flatten tensor to matrix
    xtMat=xtMat(:,:); %convert to matrix
    xtMat=xtMat(:,i); %remove columns of all zeros
    
end
