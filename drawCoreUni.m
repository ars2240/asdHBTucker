%draws core p(z|x) with uniform prior
function [phi,p] = drawCoreUni(paths,coreDims,L,r,options)
    %path = row with tree path values
    %coreDims = dimensions of core tensor
    %L = levels of hierarchical tree
    %r = restaurant lists
    %options = passed to drchrnd
    
    %initialize tucker decomposition
    %core tensor
    phi=zeros(coreDims(1),coreDims(2),coreDims(3));
    
    %draw values from dirichlet distribution with uniform prior
    len = L(1)*L(2);
    switch options.pType
        case 0
            prior=repelem(1/len,len);
        case 1
            prior=repelem(1,len);
        otherwise
            error('Error. \nNo prior type selected');
    end
    [vals,p]=drchrnd(prior,coreDims(1),options);
    
    for i=1:coreDims(1)

        %get restaurants for patient
        res{1}=paths(i,1:L(1));
        res{1}=ismember(r{1},res{1});
        res{2}=paths(i,(1+L(1)):(L(1)+L(2)));
        res{2}=ismember(r{2},res{2});
        
        %set values
        phi(i,res{1},res{2})=reshape(vals(i,:),[L(1),L(2)]);
    end
    
end