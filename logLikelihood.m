function LL = logLikelihood(x, xTest, prior, epsilon, psi, opaths, tree, varargin)

    % generate artificial patients
    sparPat=generatePatients(x, prior, psi, opaths, tree, varargin);
    
    %convert from sparse to dense
    si = size(x);
    si(1) = max(sparPat(:,1));
    modes = length(si)-1;  %number of dependent modes
    tens=sptensor(sparPat(:,1:(modes+1)),sparPat(:,modes+2), si);
    
    dims=size(tens);
    
    %normalize tens by dividing by counts
    cts=sptenmat(tens, 1);
    cts=double(cts);
    cts=sum(cts, 2);
    cts=full(cts);
    s=prod(dims(2:end));
    ctsS=cts(sparPat(:,1))+epsilon*s;
    sparPat(:,modes+2)=sparPat(:,modes+2)./ctsS;
    tens=sptensor(sparPat(:,1:(modes+1)),sparPat(:,modes+2), si);
    
    LL=0;
    n=size(xTest,1);
    ll=zeros(n,1);
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
        lTens=log(full(tTens)+epsilon./(cts+s*epsilon));
        xP=xTemp.*lTens;
        lP=sum(xP, 2);
        if sum(lP)~=0
            k=1;
            m=k*mean(lP);
            p=exp(lP-m);
            w=p./sum(p);
            while sum(p)==0 || sum(w)==0 || isinf(sum(p)) || isinf(sum(w))
                % handling of underflow error
                if sum(p)==0 || sum(w)==0
                    k=k*1.5;
                else
                    k=k/2;
                end
                m=k*mean(lP);
                p=exp(lP-m);
                w=p./sum(p);
            end
            ll(i)=log(sum(w.*p))+m;
            if isnan(ll)
                display(sum(p));
            elseif ~isfinite(ll)
                display(sum(w.*p));
            else
                LL=LL+ll(i);
            end
        end
    end
    
    if sum(ll==0)/length(ll)>0.5
        disp(ll);
    else
        LL=LL/n;
    end

end