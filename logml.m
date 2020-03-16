function l=logml(z,Y,M,pM,p)
pM=double(pM);
[z0,m]=unique(sortrows([Y+1 z]),'rows','legacy');
t=max(Y+1);
C=tensor(zeros([t M]),[t M]);
C(z0)=C(z0)+m-[0;m(1:(end-1))];
Cdata=double(tenmat(C,1));
al=0.5*ones(1,t);
l=sum(sum(gammaln(Cdata'+al),2)-gammaln(sum(Cdata'+al,2))...
    -sum(gammaln(al)))+(p-sum(M>1))*log(pM(1));
d=length(pM);
for k=2:d
    l=l+sum(M==k)*log(pM(k)/stirling(d,k));
end





