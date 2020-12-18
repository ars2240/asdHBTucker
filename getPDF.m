function pdf = getPDF(c, rList, cts1, cts2, gcts, prior)
    pdf=log(c); %take log to prevent overflow
    w=prior*length(rList);
    pdf=pdf+gammaln(sum(cts1,1)+w);
    pdf=pdf-sum(gammaln(cts1+prior),1); %test
    pdf=pdf+sum(gcts,1);
    pdf=pdf-gammaln(sum(cts2,1)+w);
    lP=pdf; t=1;
    m=t*mean(lP);
    p=exp(lP-m);
    while sum(p)==0 || isinf(sum(p))
        % handling of underflow error
        if sum(p)==0
            t=t*1.5;
        else
            t=t/2;
        end
        m=t*mean(lP);
        p=exp(lP-m);
    end
    pdf=p/sum(p); %normalize
end