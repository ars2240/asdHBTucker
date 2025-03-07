T=readtable('splice.data.csv','ReadVariableNames',false);
len = length(T.Var3{1});
npats = size(T,1);
asdSparse=zeros(npats,len+1);
[~,~,asdSparse(:,1)]=unique(T(:,2));
[~,~,asd]=unique(T(:,1));
T2=zeros(npats,len);
map('A')=1;
map('C')=2;
map('G')=3;
map('T')=4;
map('D')=5;
map('N')=6;
map('S')=7;
map('R')=8;
DNA = [];

for i=1:npats
    T2(i,:)=map(reshape(T.Var3{i},len,[]));
    DNA=unique([DNA,reshape(T.Var3{i},len,[])']);
end

[a,~]=hist(T2',unique(T2'));
b=sum(a(5:8,:),1);
bad=find(b>5);

for j=1:len
    for i=1:size(T2,1)
        if ~ismember(i,bad)
            if T2(i,j)==5
                T2(i,j)=1;
                T2T=T2(i,:);
                T2T(j)=3;
                T2=[T2;T2T];
                asdSparse=[asdSparse;asdSparse(i,:)];
                T2T(j)=4;
                T2=[T2;T2T];
                asdSparse=[asdSparse;asdSparse(i,:)];
            elseif T2(i,j)==6
                T2(i,j)=1;
                T2T=T2(i,:);
                T2T(j)=2;
                T2=[T2;T2T];
                asdSparse=[asdSparse;asdSparse(i,:)];
                T2T(j)=3;
                T2=[T2;T2T];
                asdSparse=[asdSparse;asdSparse(i,:)];
                T2T(j)=4;
                T2=[T2;T2T];
                asdSparse=[asdSparse;asdSparse(i,:)];
            elseif T2(i,j)==7
                T2(i,j)=2;
                T2T=T2(i,:);
                T2T(j)=3;
                T2=[T2;T2T];
                asdSparse=[asdSparse;asdSparse(i,:)];
            elseif T2(i,j)==8
                T2(i,j)=1;
                T2T=T2(i,:);
                T2T(j)=3;
                T2=[T2;T2T];
                asdSparse=[asdSparse;asdSparse(i,:)];
            end
        end
    end
end

ind=~ismember(asdSparse(:,1),asdSparse(bad,1));
asdSparse=asdSparse(ind,:);
T2=T2(ind,:);
[~,~,asdSparse(:,1)]=unique(asdSparse(:,1));
asd=asd(~ismember(1:npats,bad));

for i=1:len
    [~,~,asdSparse(:,i+1)]=unique(T2(:,i));
end

save('splice.mat','asdSparse','asd');
