function x = rgamma(A,B,r,c,options)
    % samples small-shape gamma RVs via accept-reject
    % based on paper - arXiv:1302.1884
    % A,B = parameters of gamma distribution (shape, scale)
    % r,c = rows and columns of output array
    % x = log of RNs drawn from gamma dist
    % options = 
    %   minA = all A's above this value will be drawn normally
    
    %re-size A and B
    if size(A,1)==1
        A = repmat(A,r,1);
    end
    if size(A,2)==1
        A = repmat(A,1,c);
    end
    if size(B,1)==1
        B = repmat(B,r,1);
    end
    if size(B,2)==1
        B = repmat(B,1,c);
    end
    
    %initialize x,Z
    x=zeros(r,c);
    Z=zeros(r,c);
    
    for i=1:r
        for j=1:c
            if A(i,j)>options.minA
                x(i,j)=log(gamrnd(A(i,j),B(i,j),1,1));
            else
                Z(i,j)=rh(A(i,j));
                x(i,j)=log(B(i,j))-Z(i,j)/A(i,j);
            end
        end
    end
end

function x = eta(z,w,L)
    if z>=0
        x=exp(-z);
    else
        x=w * L * exp(L * z);
    end
end

function x = h(z,a)
    x=exp(-z - exp(-z / a));
end

function z = rh(a)
    con=1;
    L=1/a-1;
    w=a/exp(1)/(1-a);
    ww=1/(1+w);
    while con==1
        U=rand;
        if U<=ww
            z=-log(U/ww);
        else
            z=log(rand)/L;
        end
        if (h(z,a)/eta(z,w,L))>rand
            con=0;
        end
    end
end