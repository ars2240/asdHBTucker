%draws core p(z|x) with uniform prior
function p = drawCoreUni(samps,coreDims,L,r)
    %samps = row with x, y, z, tree path values for specific x
    %coreDims = dimensions of core tensor
    %L = levels of hierarchical tree
    %r = restaurant lists
    
    %initialize probability vectors
    prob{1}=zeros(1,coreDims(2));
    prob{2}=zeros(1,coreDims(3));

    %get restaurants for patient
    res{1}=samps(6:(5+L(1)));
    res{1}=find(ismember(r{1},res{1}));
    res{2}=samps((6+L(1)):(5+L(1)+L(2)));
    res{2}=find(ismember(r{2},res{2}));

    %draw values from dirichlet distribution with uniform prior
    vals{1}=drchrnd(repelem(1/L(1),L(1)),1);
    vals{2}=drchrnd(repelem(1/L(2),L(2)),1);

    %set values
    prob{1}(res{1})=vals{1};
    prob{2}(res{2})=vals{2};
    p=prob{1}'*prob{2};
end