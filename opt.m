x0 = [1,0.5];     % Make a starting guess at the solution
options = optimoptions(@fmincon,'Algorithm','sqp');
[x,fval] = ... 
fmincon(@objFun,x0,[],[],[],[],[],[],@conFun,options);
x

function f = objFun(x)
    n=3408; t=8; c=3;
    d=floor(n/t);
    r=rem(n,t);
    f1=gammaln(n-t+1-x(2))+(t-1)*gammaln(1-x(2));
    f2=(t-r)*gammaln(d-x(2))+r*gammaln(d+1-x(2));
    f=sum(gammaln(x(1)+(1:t)*x(2)));
    f=f+gammaln(x(1)+1)-gammaln(x(1)+n)-gammaln(1-x(2))^t;
    f1=f1+f;
    f2=f2+f;
    f=log(exp(f1)-exp(f2)+1-c/(1+exp(x(1)+x(2))));
end

function [c, ceq] = conFun(x)
    % Nonlinear inequality constraints
    c = [-x(1);     
         -x(2);
         x(2)-1];
    % Nonlinear equality constraints
    ceq = [];
end