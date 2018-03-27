function ent = entropy(p)
    %p=probability vector
    %returns entropy of probability vector
    ent=-p.*log2(p);
    ent(p==0)=0;
    ent=sum(ent);
end