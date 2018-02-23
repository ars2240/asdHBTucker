%draws core p(z|x) with conditional prior
function p = drawCoreCon(samps,coreDims,L,r)
    %samps = rows with x, y, z, tree path values for specific x
    %coreDims = dimensions of core tensor
    %L = levels of hierarchical tree
    %r = restaurant lists

    %initialize probability vectors
    prob{1}=zeros(1,coreDims(2));
    prob{2}=zeros(1,coreDims(3));

    %get restaurants for patient
    res{1}=samps(1,6:(5+L(1)));
    res{1}=find(ismember(r{1},res{1}));
    res{2}=samps(1,(6+L(1)):(5+L(1)+L(2)));
    res{2}=find(ismember(r{2},res{2}));

    %get topics and locations in restaurant
    prior{1}=histc(samps(:,4)',res{1}); %calculate counts

    %get topics and locations in restaurant
    prior{2}=histc(samps(:,5)',res{2}); %calculate counts

    %add prior to uniform prior
    prior{1}=prior{1}+repelem(1/L(1),L(1));
    prior{2}=prior{2}+repelem(1/L(2),L(2)); 

    %draw values from dirichlet distribution with prior
    vals{1}=drchrnd(prior{1},1);
    vals{2}=drchrnd(prior{2},1);

    %set values
    prob{1}(res{1})=vals{1};
    prob{2}(res{2})=vals{2};
    p=prob{1}'*prob{2};
end