function r = drchrnd(a,n)
    %take n samples from a dirichlet distribution with prior a
    p = length(a);
    r = gamrnd(repmat(a,n,1),1,n,p);
    r = r ./ repmat(sum(r,2),1,p);
end