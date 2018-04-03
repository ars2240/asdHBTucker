%draws core p(z|x) with uniform prior
function [phi,p] = drawCoreUni(paths,coreDims,L,r,options)
    %path = row with tree path values for specific x
    %coreDims = dimensions of core tensor
    %L = levels of hierarchical tree
    %r = restaurant lists
    %options = 
    % prior = value to add to prior
    
    %initialize probability vectors
    prob{1}=zeros(1,coreDims(2));
    prob{2}=zeros(1,coreDims(3));

    %get restaurants for patient
    res{1}=paths(1:L(1));
    res{1}=find(ismember(r{1},res{1}));
    res{2}=paths((1+L(1)):(L(1)+L(2)));
    res{2}=find(ismember(r{2},res{2}));

    %draw values from dirichlet distribution with uniform prior
    [vals{1},p1]=drchrnd(repelem(1/L(1)+options.prior,L(1)),1,options);
    [vals{2},p2]=drchrnd(repelem(1/L(2)+options.prior,L(2)),1,options);

    %set values
    prob{1}(res{1})=vals{1};
    prob{2}(res{2})=vals{2};
    phi=prob{1}'*prob{2};
    p=[p1;p2];
end