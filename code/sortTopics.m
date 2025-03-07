function [rN, tN, pN] = sortTopics(r, t, p, L)

    modes=size(r,1);
    rN=cell(modes,1);
    tN=cell(modes,1);
    pN=zeros(size(p));
    for i=1:modes
        r{i}=unique(p(:,(1+sum(L(1:(i-1)))):sum(L(1:i))))';
        l=length(r{i});
        rN{i}=1:l;
        tN{i}=cell(l,1);
        for j=1:l
            tN{i}{j}=arrayfun(@(x)find(r{i}==x,1),t{i}{r{i}(j)});
        end
        col=sum(L(1:(i-1))); %starting column
        pN(:,(col+1):(col+L(i)))=arrayfun(@(x)find(r{i}==x,1),p(:,(col+1):(col+L(i))));
    end
end
