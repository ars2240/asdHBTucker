function [r, p] = drchrnd(a,n,options)
    %take n samples from a dirichlet distribution with prior a
    %options
    % tol = tolerance for zeros of dirichlet distribution
    
    l = length(a);
    A = repmat(a,n,1);
    r = gamrnd(A,1,n,l);
    r(r<options.tol) = options.tol;
    d = repmat(sum(r,2),1,l);
    
    %calculate pdf
    p = log(r)-log(d);
    p = p.*repmat(a-1,n,1);
    p = sum(p,2);
    p = p-sum(gammaln(a))+gammaln(sum(a));
    
    %normalize
    r = r./d;
end