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
        if T2(i,j)==5
            pc = [1,3,4];
        elseif T2(i,j)==6
            pc = [1,2,3,4];
        elseif T2(i,j)==7
            pc = [2,3];
        elseif T2(i,j)==8
            pc = [1,3];
        else
            pc = T2(i,j);
        end
        r = randi([1,length(pc)],1);
        T2(i,j)=pc(r);
    end
end

for i=1:len
    [~,~,asdSparse(:,i+1)]=unique(T2(:,i));
end

asdSparse=asdSparse(~ismember(asdSparse(:,1),bad),:);
[~,~,asdSparse(:,1)]=unique(asdSparse(:,1));
asd=asd(~ismember(1:npats,bad));

save('splice.mat','asdSparse','asd');