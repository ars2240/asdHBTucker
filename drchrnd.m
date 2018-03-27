function [r, p] = drchrnd(a,n)
    %take n samples from a dirichlet distribution with prior a
    
    l = length(a);
    A = repmat(a,n,1);
    r = gamrnd(A,1,n,l);
    d = repmat(sum(r,2),1,l);
    
    %calculate pdf
    p = gampdf(r,A,1);
    p = sum(p,2);
    
    %normalize
    r = r ./ d;
end