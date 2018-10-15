function LL = logLikelihood(x, xTest, npats, prior, epsilon, psi, r, opaths, tree, vargin)

    % generate artificial patients
    sparse=generatePatients(x, npats, prior, psi, r, opaths, tree, vargin);
    
    %convert from sparse to dense
    tens=sptensor(sparse(:,1:3),sparse(:,4));
    
    dims=size(tens);
    
    %normalize tens by dividing by counts
    cts=sptenmat(tens, 1);
    cts=double(cts);
    cts=sum(cts, 2);
    cts=full(cts);
    ctsS=cts(sparse(:,1))+epsilon*prod(dims(2:end));
    sparse(:,4)=sparse(:,4)./ctsS;
    tens=sptensor(sparse(:,1:3),sparse(:,4));
    
    LL=0;
    n=size(xTest,1);
    for i=1:n
        switch ndims(xTest)
            case 3
                xTemp=xTest(i,:,:);
                xTemp=double(sptenmat(xTemp, []));
            case 2
                xTemp=xTest(i,:);
                xTemp=double(xTemp);
                xTemp=xTemp';
        end
        y=find(xTemp);
        xTemp=xTemp(y);
        tTens=double(sptenmat(tens, 1));
        tTens=tTens(:,y);
        lTens=log(full(tTens)+epsilon./cts);
        xP=xTemp.*lTens;
        lP=sum(xP, 2);
        p=exp(lP);
        m=0;
        k=1;
        while sum(p)==0 || isinf(sum(p))
            % handling of underflow error
            if sum(p)==0
                k=k*1.5;
            else
                k=k/2;
            end
            m=k*mean(lP);
            p=exp(lP-m);
        end
        w=p./sum(p);
        ll=log(sum(w.*p))+m;
        if isnan(ll)
            display(sum(p));
        end
        LL=LL+ll;
    end
    
    LL=LL/n;

end