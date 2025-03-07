function nPhi = newTopics(asdTens,psi,nPaths,ind,options)

    L=options.L;
    
    %adjustment if using constant L across dims
    if length(L)==1
        L=repelem(L,2);
    end


    ind=find(ind);
    nPhi=zeros(length(ind),size(psi{1},2),size(psi{2},2));
    
    r=cell(2,1);
    r{1}=unique(nPaths(:,1:L(1)));
    r{2}=unique(nPaths(:,(L(1)+1):sum(L)));
    
    for i=1:length(ind)
        r1=nPaths(i,1:L(1));
        [~,res1]=ismember(r1,r{1});
        r2=nPaths(i,(L(1)+1):sum(L));
        [~,res2]=ismember(r2,r{2});
        psi1=psi{1}(:,res1);
        psi1=log(psi1);
        psi2=psi{2}(:,res2);
        psi2=log(psi2);
        t=tenmat(full(asdTens(ind(i),:,:)),1,2);
        t=t(:,:);
        x1=max(t,[],2);
        x2=max(t,[],1);
        x2=x2';
        m1=sum(x1.*psi1,1);
        m1=m1-mean(m1);
        m1=exp(m1);
        m1=m1/sum(m1);
        m2=sum(x2.*psi2,1);
        m2=m2-mean(m2);
        m2=exp(m2);
        m2=m2/sum(m2);
        nPhi(i,res1,res2)=m1'*m2;
    end
end
