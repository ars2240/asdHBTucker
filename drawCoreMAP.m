%draws core p(z|x) with conditional prior
function phi = drawCoreMAP(samples,paths,coreDims,r,options)
    %sampless = rows with x, y, z values
    %path = row with tree path values
    %coreDims = dimensions of core tensor
    %r = restaurant lists
    %options = passed to drchrnd

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
    
    %get counts
    cts=accumarray(samples(:,[(2+modes):(1+2*modes),1]),1);
    for i=1:modes
    	pad=max(r{i})-size(cts,i);
        if pad>0
            padding=zeros(1,modes+1);
            padding(i)=pad;
            cts=padarray(cts,padding,'post');
        end
    end
    ind=cell(modes+1,1);
    for i=1:modes
        ind{i}=r{i};
    end
    ind{modes+1}=1:size(cts,modes+1);
    cs=[coreDims(2:end),coreDims(1)];
    cts=cts(tensIndex2(ind,cs));
    
    % size of topic space
    switch options.topicType
        case 'Cartesian'
            len = prod(L);
        case 'Level'
            len = L(1);
        otherwise
            error('Error. \nNo topic type selected');
    end
    
    p = zeros(1,coreDims(1)); %initialize probability matrix
    
    for i=1:coreDims(1)

        %add prior to uniform prior
        switch options.pType
            case 0
                prior=repelem(1/len,len);
            case 1
                prior=repelem(1,len);
            case 2
                prior=repelem(2/len,len);
            otherwise
                error('Error. \nNo prior type selected');
        end
        
        switch options.topicType
            case 'Cartesian'
                ind=cell(modes+1,1);
                %get restaurants for patient
                for j=1:modes
                    ind{j}=paths(i,(1+sum(L(1:(j-1)))):sum(L(1:j)));
                end
                ind{modes+1}=i;
                prior=prior+reshape(cts(tensIndex2(ind,cs)),[1,len]);
            case 'Level'
                ind = zeros(L(1),modes+1);
                %get restaurants for patient
                for j=1:modes
                    ind(:,j)=paths(i,(1+sum(L(1:(j-1)))):sum(L(1:j)));
                end
                ind(:,modes+1)=i;
                prior=prior+reshape(cts(tensIndex2(ind,cs)),[1,len]);
            otherwise
                error('Error. \nNo topic type selected');
        end

        %set values using MAP estimate
        prior(prior<=options.cutoff)=0;
        vals=prior./sum(prior);
        
        %set values
        switch options.topicType
            case 'Cartesian'
                ind=cell(modes+1,1);
                %get restaurants for patient
                ind{1}=i;
                for j=1:modes
                    ind{j+1}=1:L(j);
                end
                phi(tensIndex2(ind,size(phi)))=reshape(vals,L);
            case 'Level'
                ind = zeros(L(1),modes+1);
                ind(:,1)=i;
                %get restaurants for patient
                for j=1:modes
                    ind(:,j+1)=1:L(1);
                end
                phi(tensIndex2(ind,size(phi)))=vals;
            otherwise
                error('Error. \nNo topic type selected');
        end
    end
    
    if ndims(phi) < 3
        phi(end, end, 2) = 0; 
    end
    
end