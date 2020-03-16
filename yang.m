function [PPE,Ypred] = yang(x,y,bi)
% -- tensor factorization method -- %

% -- Yun Yang -- %
% last modified: Oct, 10, 2012 -- %

    rng(123);

    %constants
    x=double(x);
    p=size(x,1); %number of features
    N=length(bi); %total size
    n=sum(~bi); %training size
    ep=max(6,floor(log(n)/log(4))); %expected maximum number of predictors
    d=max(max(x)); %number of categories for each features
    if d==1
        error('Not enough classes');
    end
    d0=1; %prior for Dirichlet Distribution
    c=0.3;
    pM=[1-(d-1)*c*ep/p,repelem(c*ep/p, d-1)]; %prior probability for kj 
    np=0; %number of predictors included in the model
    t=max(y+1); %number of topics
    
    X0=x;
    Y0=y;
    x=x(~bi,:);
    Y=y(~bi);

    MS=30; %number of iterations for first stage
    K=1000; %number of iterations for second stage
    
    cM=(2.^(1:d)-2)/2;
    M=ones(MS+1,p);G=ones(p,d); z=ones(n,p);log0=zeros(MS,1);
    for k=1:MS     
    M00=M(k,:);
    for j=1:p
        M0=M00(j);
        if M0==1
          if np<ep
            new=binornd(1,0.5*ones(1,d-1));
            while sum(new)==0
                new=binornd(1,0.5*ones(1,d-1));
            end
            GG=G(j,:)+[0 new];
            MM=M00;
            MM0=MM;
            MM(j)=2;
            zz=z;
            zz(:,j)=GG(x(:,j));
            ind1=find(MM>1);
            ind2=find(MM0>1);
            if isempty(ind2)
                ind2=1;
            end
            logR=logml(zz(:,ind1),Y,MM(ind1),pM,p)-logml(z(:,ind2),Y,MM0(ind2),pM,p);
            logR=logR+log(0.5)+log(cM(d));
            if log(rand)<logR&&np<ep
                G(j,:)=GG;
                M00=MM;
                z=zz;
                np=np+1;
            else
                M00=MM0;
            end    
          end
          if M00(j)==1&&np>0
              ind1=find(M00>1); %switch-random select a feature to be replaced
              tempind=randsample(length(ind1),1);  
              temp=ind1(tempind);
              zz=z;       
              zz(:,temp)=ones(n,1);
              MM=M00;
              MM0=MM;
              MM(temp)=1;
              GG=G(temp,:);   
              per=randsample(d,d);
              GG0=GG;
              for s=1:d
                  GG0(s)=GG(per(s));
              end
              GG=GG0;
              MM(j)=MM0(temp);
              zz(:,j)=GG(x(:,j));
              ind1=find(MM>1);
              ind2=find(MM0>1);
              logR=logml(zz(:,ind1),Y,MM(ind1),pM,p)-logml(z(:,ind2),Y,MM0(ind2),pM,p);
              if log(rand)<logR
                G(j,:)=GG;
                G(temp,:)=ones(1,d);
                M00=MM;
                z=zz;
              end
          end
        end
        if M0>1&&M0<d
            if rand<0.5
                cnew=randsample(M00(j),2);
                lnew=max(cnew);
                snew=min(cnew);
                GG=G(j,:);
                GG(GG==lnew)=snew;
                if lnew<M00(j)
                    GG(GG==M00(j))=lnew;
                end
                zz=z;
                zz(:,j)=GG(x(:,j));
                MM=M00;
                MM0=MM;
                MM(j)=M00(j)-1;
                ind1=find(MM>1);
                ind2=find(MM0>1);
                if isempty(ind1)
                    ind1=1;
                end
                if isempty(ind2)
                    ind2=1;
                end
                logR=logml(zz(:,ind1),Y,MM(ind1),pM,p)-logml(z(:,ind2),Y,MM0(ind2),pM,p);
                if M00(j)>2
                    [z0,mm]=unique(sort(GG),'legacy');
                    gn=mm-[0 mm(1:(end-1))];
                    logR=logR-log(sum(cM(gn)))+log(M00(j)*(M00(j)-1)/2);
                else
                    logR=logR-log(cM(d))-log(0.5); 
                end
                if log(rand)<logR
                    G(j,:)=GG;
                    M00=MM;
                    z=zz;
                    if M00(j)==1
                        np=np-1;
                    end
                else
                    M00=MM0;
                end
            else
                [z0,mm]=unique(sort(G(j,:)),'legacy');
                gn=mm-[0 mm(1:(end-1))];
                pgn=cM(gn)/sum(cM(gn));
                l=sum(mnrnd(1,pgn).*(1:M00(j)));
                new=binornd(1,0.5*ones(1,gn(l)-1));
                while sum(new)==0
                    new=binornd(1,0.5*ones(1,gn(l)-1));
                end
                GG=G(j,:);
                GG(GG==l)=l+(M00(j)+1-l)*[0 new];
                zz=z;
                zz(:,j)=GG(x(:,j));
                MM=M00;
                MM0=MM;
                MM(j)=M00(j)+1;
                ind1=find(MM>1);
                ind2=find(MM0>1);
                if isempty(ind2)
                    ind2=1;
                end                
                logR=logml(zz(:,ind1),Y,MM(ind1),pM,p)-logml(z(:,ind2),Y,MM0(ind2),pM,p);
                if M00(j)<d-1
                    logR=logR-log(M00(j)*(M00(j)+1)/2)+log(sum(cM(gn)));
                else
                    logR=logR-log(d*(d-1)/2)-log(0.5);
                end
                if log(rand)<logR
                    G(j,:)=GG;
                    M00=MM;
                    z=zz;
                else
                    M00=MM0; 
                end     
            end
        end
        if M0==d
            cnew=randsample(d,2);
            lnew=max(cnew);
            snew=min(cnew);
            GG=G(j,:);
            GG(GG==lnew)=snew;
            if lnew<d
                GG(GG==M00(j))=lnew;
            end
            zz=z;
            zz(:,j)=GG(x(:,j));
            MM=M00;
            MM0=MM;
            MM(j)=d-1;
            ind1=find(MM>1);
            ind2=find(MM0>1);
            if isempty(ind2)
               ind2=1;
            end            
            logR=logml(zz(:,ind1),Y,MM(ind1),pM,p)-logml(z(:,ind2),Y,MM0(ind2),pM,p);
            logR=logR+log(0.5)+log(d*(d-1)/2);
            if log(rand)<logR
                G(j,:)=GG;
                M00=MM;
                z=zz;
            else
                M00=MM0;
            end   
        end 
        if M00(j)>1  %resplitting-change the splitting scheme
            zz=z;                  
            GG=G(j,:);   
            per=randsample(d,d);
            GG0=GG;
            for s=1:d
                GG0(s)=GG(per(s));
            end
            GG=GG0;
            zz(:,j)=GG(x(:,j));
            ind1=find(M00>1);
            ind2=find(M00>1);
            logR=logml(zz(:,ind1),Y,M00(ind1),pM,p)-logml(z(:,ind2),Y,M00(ind2),pM,p);
            if log(rand)<logR
              G(j,:)=GG;
              z=zz;
            end
        end
    end
    M(k+1,:)=M00;
    % print informations in each iteration
    ind1=find(M(k+1,:)>1);
    if isempty(ind1)
        ind1=1;
    end   
    log0(k)=logml(z(:,ind1),Y,M(k+1,ind1),pM,p);
    [aa,b]=find(M(k+1,:)-1);
    fprintf('k = %i, %i important predictors = {',k,np);
    for i=1:length(b)
        fprintf(' %i(%i)',b(i),M(k+1,b(i)));
    end
    fprintf(' }. %f \n',log0(k));        
    end

    aveM=mean(M(floor(MS/2):MS,:),1);
    % [a,b]=max(log0);
    MM0=round(aveM);
    % MM0=M(b,:);
    if sum(MM0)==p
    [m,I]=max(mean(M,1));
    MM0(I)=2;
    end
    ind=find(MM0>1);
    M0=MM0(ind);
    p0=length(ind);
    z=ones(n,p0);
    M=ones(K+1,1)*M0;
    for j=1:p0
    z(:,j)=randsample(M(1,j),n,true);
    end
    x0=x(:,ind);

    pi=zeros(p0,d,d,K);
    PP=zeros(K,t*d^6);
    %Gibbs sampler
    for k=1:K
    %pi 
    cp=zeros(p0,d,d);%counts for pi,first=j,second=value of x,third=value of z
    for i=1:n
        for j=1:p0
            cp(j,x0(i,j),z(i,j))=cp(j,x0(i,j),z(i,j))+1;
        end
    end
    for j=1:p0
        for s=1:d
            r = gamrnd(cp(j,s,1:M(k,j))+1/M(k,j)/d0,1);
            pi(j,s,1:M(k,j),k)=r/sum(r);
        end
        %switch label
        [qq1,qq2]=sort(sum(reshape(pi(j,:,1:M(k,j),k),d,M(k,j)),1),'descend');
        for s=1:d
            pi(j,s,1:M(k,j),k)=pi(j,s,qq2,k);
        end
        for i=1:n
            z(i,j)=find(qq2==z(i,j));
        end
    end

    %lambda
    clT=tensor(zeros([t,M(k,:)]),[t,M(k,:)]);
    [z0,m]=unique(sortrows([Y+1 z]),'rows','legacy');
    clT(z0)=clT(z0)+m-[0;m(1:(end-1))];
    clTdata=tenmat(clT,1);
    tot=prod(M(k,:));
    al=0.5*ones(tot,t);
    for j=1:t
        cl0data=clTdata(j,:);
        cl0rdim=[];
        cl0cdim=1:p0;
        cl0tsize=M(k,:);
        cl02=tenmat(cl0data,cl0rdim,cl0cdim,cl0tsize);
        cl0=tensor(cl02);
        a=tenmat(cl0,[],'t');
        al(:,j)=al(:,j)+a(1:end);
    end
    options.prior=1;
    options.minA=0;
    ra=gamrnd(al,1,tot,t);
    ra=ra./sum(ra,2);
    lambda=tensor(ra,[t,M(k,:)]);

    %z
    for j=1:p0
        q=zeros(n,M(k+1,j));%for compute p(z=h|-)
        for h=1:M(k+1,j)
            q(:,h)=pi(j,x0(:,j),h,k).*(reshape(double(lambda([Y+1,z(:,1:(j-1)),h*ones(n,1),z(:,(j+1):p0)])),n,1))';
        end
        q=bsxfun(@rdivide,q,(sum(q,2)));
        z(:,j)=sum(bsxfun(@times,mnrnd(1,q),1:M(k+1,j)),2);
    end

    % calculate conditional probability tensor PP
    ep0=6;
    id1=ind;
    id3=(p0+1)*ones(1,ep0);
    id3(1:length(id1))=1:p0;
    MM=M(k+1,:); MM(p0+1)=1; MM=MM(id3);
    pp=pi(:,:,:,k); pp(p0+1,:,:)=[ones(d,1) zeros(d,d-1)]; pp=pp(id3,:,:);
    aug=cell(ep0+1,1);
    aug{1}=eye(t);
    for j=1:ep0
        aug{j+1}=reshape(pp(j,:,1:MM(j)),d,MM(j));
    end
    P=tensor(ttensor(tensor(double(lambda),[t,MM]),aug));
    S=tenmat(P,[]);
    PP(k,:)=S.data;
    % print informations in each iteration
    fprintf('k = %i.\n',k);
    end

    PP0=mean(PP(floor(K/2):K,:),1);
    PPE=tensor(reshape(PP0,t,d,d,d,d,d,d));
    Ypred=zeros(N,1);
    indpred=find(bi);
    indr=setdiff(1:p,ind);
    if length(ind)<6
    ind=[ind indr(1:(6-length(ind)))];
    end
    for i=1:N
    [~,Ypred(i)]=max(PPE([(1:t)',repelem(X0(i,ind),t,1)]));
    end
    Ypred=Ypred-1;
    mean((Y-Ypred(~bi))==0) %training accuracy
    mean((Y0(indpred)-Ypred(indpred))==0) %testing accuracy
end