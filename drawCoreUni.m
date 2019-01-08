%draws core p(z|x) with uniform prior
function [phi,p] = drawCoreUni(paths,coreDims,L,varargin)
    %path = row with tree path values
    %coreDims = dimensions of core tensor
    %L = levels of hierarchical tree
    %r = restaurant lists
    %options = passed to drchrnd
    
    if length(varargin)==1
        options=varargin;
    elseif length(varargin)==2
        r=varargin{1};
        options=varargin{2};
    else
        error("Error. \nIncorrect number of inputs.");
    end
    
    L=options.L;
    %adjustment if using constant L across dims
    if length(L)==1
        L=repelem(L,2);
    end
    
    %initialize tucker decomposition
    %core tensor
    if options.sparse==0
        phi=zeros(coreDims(1),coreDims(2),coreDims(3));
    else
        phi=zeros(coreDims(1),L(1),L(2));
    end
    
    % size of topic space
    switch options.topicType
        case 'Cartesian'
            len = L(1)*L(2);
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
    
    for i=1:coreDims(1)

        %get restaurants for patient
        res{1}=paths(i,1:L(1));
        %res{1}=ismember(r{1},res{1});
        res{2}=paths(i,(1+L(1)):(L(1)+L(2)));
        %res{2}=ismember(r{2},res{2});
        
        %set values
        if options.sparse==0
            switch options.topicType
                case 'Cartesian'
                    phi(i,res{1},res{2})=reshape(vals(i,:),[L(1),L(2)]);
                case 'Level'
                    phi(i,res{1},res{2})=diag(vals(i,:));
                otherwise
                    error('Error. \nNo topic type selected');
            end
        else
            switch options.topicType
                case 'Cartesian'
                    phi(i,:,:)=reshape(vals(i,:),[L(1),L(2)]);
                case 'Level'
                    phi(i,:,:)=diag(vals(i,:));
                otherwise
                    error('Error. \nNo topic type selected');
            end
        end
    end
    
end