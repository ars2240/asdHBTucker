function l=logml2(z,Y,M,pM,p)
[z0,m]=unique(sortrows([Y z]),'rows','legacy');
C=tensor(zeros([3 M]),[3 M]);
C(z0)=C(z0)+m-[0;m(1:(end-1))];
Cdata=tenmat(C,1);
l=sum(gammaln(Cdata(1,:)+1/3)+gammaln(Cdata(2,:)+1/3)+gammaln(Cdata(3,:)+1/3)...
    -gammaln(Cdata(1,:)+Cdata(2,:)+Cdata(3,:)+1)-3*gammaln(1/3)+gammaln(1))...
    +(p-sum(M>1))*log(pM(1));
d=length(pM);
for k=2:d
    l=l+sum(M==k)*log(pM(k)/stirling(d,k));
end





