function [rN, tN, pN] = sortTopics(r, t, p, L)

    rN=cell(2,1);
    tN=cell(2,1);
    pN=zeros(size(p));
    for i=1:2
        l=length(r{i});
        rN{i}=1:l;
        tN{i}=cell(l,1);
        for j=1:l
            tN{i}{j}=arrayfun(@(x)find(r{i}==x,1),t{i}{r{i}(j)});
        end
        col=(i-1)*L(1); %starting column
        pN(:,(col+1):(col+L(i)))=arrayfun(@(x)find(r{i}==x,1),p(:,(col+1):(col+L(i))));
    end
end