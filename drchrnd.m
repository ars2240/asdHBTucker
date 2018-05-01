function [r, p] = drchrnd(a,n,options)
    %take n samples from a dirichlet distribution with prior a
    %options
    % prior = value to add to prior
    % tol = tolerance for zeros of dirichlet distribution
    
    a = a+options.prior;
    l = length(a);
    A = repmat(a,n,1);
    if min(a)<options.minA
        r = rgamma(A,1,n,l,options);
        d = repmat(sum(exp(r),2),1,l);
        r = r-log(d);
    else
        r = gamrnd(A,1,n,l);
        d = repmat(sum(r,2),1,l);
        r = log(r)-log(d);
    end
    
    %calculate pdf
    b = sum(gammaln(a))-gammaln(sum(a));
    p = r.*repmat(a-1,n,1);
    p = sum(p,2);
    p = p-b;
    
    %normalize
    r = exp(r);
    if options.tol~=0
       r(r<options.tol) = options.tol; 
    end
end