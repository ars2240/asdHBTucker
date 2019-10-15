%draws core p(z|x) with uniform prior
function [phi,p] = drawCoreUni(paths,coreDims,varargin)
    %path = row with tree path values
    %coreDims = dimensions of core tensor
    %L = levels of hierarchical tree
    %r = restaurant lists
    %options = passed to drchrnd
    
    if length(varargin)==1
        options=varargin{1};
    elseif length(varargin)==2
        r=varargin{1};
        options=varargin{2};
    else
        error("Error. \nIncorrect number of inputs.");
    end
    
    modes=length(coreDims)-1; %number of dependent modes
    L=options.L;
    %adjustment if using constant L across dims
    if length(L)==1
        L=repelem(L,modes);
    end
    
    %initialize tucker decomposition
    %core tensor
    if options.sparse==0
        phi=zeros(coreDims);
    else
        phi=zeros([coreDims(1),L]);
    end
    
    % size of topic space
    switch options.topicType
        case 'Cartesian'
            len = prod(L);
        case 'Level'
            len = L(1);
        otherwise
            error('Error. \nNo topic type selected');
    end
    
    %draw values from dirichlet distribution with uniform prior
    switch options.pType
        case 0
            prior=repelem(1/len,len);
        case 1
            prior=repelem(1,len);
        otherwise
            error('Error. \nNo prior type selected');
    end
    [vals,p]=drchrnd(prior,coreDims(1),options);
    
    res=cell(modes,1);
    for i=1:coreDims(1)
        
        %set values
        if options.sparse==0
            %get restaurants for patient
            for j=1:modes
                res{j}=paths(i,(1+sum(L(1:(j-1)))):sum(L(1:j)));
                %res{2}=ismember(r{2},res{2});
            end
        else
            %get restaurants for patient
            for j=1:modes
                res{j}=1:L(j);
            end
        end
        switch options.topicType
            case 'Cartesian'
                ind=tensIndex(res);
                len=size(ind,1);
                ind2=tensIndex2([repmat(i,[len,1]),ind],size(phi));
                phi(ind2)=reshape(vals(i,:),L);
            case 'Level'
                ind = zeros(L(1),modes);
                for j=1:modes
                    ind(:,j)=res{j};
                end
                len=L(1);
                ind2=tensIndex2([repmat(i,[len,1]),ind],size(phi));
                phi(ind2)=vals(i,:);
            otherwise
                error('Error. \nNo topic type selected');
        end
    end
    
end