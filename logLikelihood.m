function LL = logLikelihood(x, xTest, npats, prior, epsilon, psi, r, opaths, tree, varargin)

    % generate artificial patients
    sparPat=generatePatients(x, npats, prior, psi, r, opaths, tree, varargin);
    
    %convert from sparse to dense
    si = size(x);
    si(1) = npats;
    modes = length(si)-1;  %number of dependent modes
    tens=sptensor(sparPat(:,1:(modes+1)),sparPat(:,modes+2), si);
    
    dims=size(tens);
    
    %normalize tens by dividing by counts
    cts=sptenmat(tens, 1);
    cts=double(cts);
    cts=sum(cts, 2);
    cts=full(cts);
    ctsS=cts(sparPat(:,1))+epsilon*prod(dims(2:end));
    sparPat(:,modes+2)=sparPat(:,modes+2)./ctsS;
    tens=sptensor(sparPat(:,1:(modes+1)),sparPat(:,modes+2), si);
    
    LL=0;
    n=size(xTest,1);
    if ndims(xTest)>=3
        xTest=sptenmat(xTest, 1);
        xSubs=xTest.subs;
        xVals=xTest.vals;
        xTest=sparse(xSubs(:,1),xSubs(:,2),xVals);
    end
    for i=1:n
        xTemp=xTest(i,:);
        xTemp=double(xTemp);
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
        if ~isfinite(ll)
            display(sum(w.*p));
        end
        LL=LL+ll;
    end
    
    LL=LL/n;

end