%draws core p(z|x) with conditional prior
function [phi,p] = drawCoreCon(samples,paths,coreDims,L,r,options)
    %sampless = rows with x, y, z values
    %path = row with tree path values
    %coreDims = dimensions of core tensor
    %L = levels of hierarchical tree
    %r = restaurant lists
    %options = passed to drchrnd
    
    %initialize tucker decomposition
    %core tensor
    phi=zeros(coreDims(1),coreDims(2),coreDims(3));

    %get counts
    cts=accumarray(samples(:,[4 5 1]),1);
    while max(r{1})>size(cts,1)
        cts=padarray(cts,[1 0 0],'post');
    end
    while max(r{2})>size(cts,2)
        cts=padarray(cts,[0 1 0],'post');
    end
    cts=cts(r{1},r{2},:);
    
    len = L(1)*L(2); %size of z-space
    
    p = zeros(1,coreDims(1)); %initialize probability matrix
    
    for i=1:coreDims(1)
        %get restaurants for patient
        res{1}=paths(i,1:L(1));
        res{1}=ismember(r{1},res{1});
        res{2}=paths(i,(1+L(1)):(L(1)+L(2)));
        res{2}=ismember(r{2},res{2});

        %add prior to uniform prior
        switch options.pType
            case 0
                prior=repelem(1/len,len);
            case 1
                prior=repelem(1,len);
            otherwise
                error('Error. \nNo prior type selected');
        end
        prior=prior+reshape(cts(res{1},res{2},i),[1,len]);

        %draw values from dirichlet distribution with prior
        [vals,p(i)]=drchrnd(prior,1,options);

        %set values
        phi(i,res{1},res{2})=reshape(vals,[L(1),L(2)]);
    end
    
end