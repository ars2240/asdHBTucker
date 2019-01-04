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
    if options.sparse==0
        phi=zeros(coreDims(1),coreDims(2),coreDims(3));
    else
        phi=sptensor([],[],coreDims);
    end

    %get counts
    cts=accumarray(samples(:,[4 5 1]),1);
    while max(r{1})>size(cts,1)
        cts=padarray(cts,[1 0 0],'post');
    end
    while max(r{2})>size(cts,2)
        cts=padarray(cts,[0 1 0],'post');
    end
    cts=cts(r{1},r{2},:);
    
    % size of topic space
    switch options.topicType
        case 'Cartesian'
            len = L(1)*L(2);
        case 'Level'
            len = L(1);
        otherwise
            error('Error. \nNo topic type selected');
    end
    
    p = zeros(1,coreDims(1)); %initialize probability matrix
    
    for i=1:coreDims(1)
        %get restaurants for patient
        res{1}=paths(i,1:L(1));
        %res{1}=ismember(r{1},res{1});
        res{2}=paths(i,(1+L(1)):(L(1)+L(2)));
        %res{2}=ismember(r{2},res{2});
        
%         if sum(ismember(r{1},res{1}))~=L(1)
%             display(res{1});
%             display(r{1});
%             error('Bad restaurant list.');
%         end
%         if sum(ismember(r{2},res{2}))~=L(2)
%             display(res{2});
%             display(r{2});
%             error('Bad restaurant list.');
%         end

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
        if options.sparse==0
            switch options.topicType
                case 'Cartesian'
                    phi(i,res{1},res{2})=reshape(vals,[L(1),L(2)]);
                case 'Level'
                    phi(i,res{1},res{2})=diag(vals);
                otherwise
                    error('Error. \nNo topic type selected');
            end
        else
            switch options.topicType
                case 'Cartesian'
                    subs=zeros(prod(L),3);
                    subs(:,1)=i;
                    subs(:,2)=repmat(res{1},[1,L(2)]);
                    subs(:,3)=repelem(res{2},L(1));
                case 'Level'
                    subs=zeros(L(1),3);
                    subs(:,1)=i;
                    subs(:,2)=res{1};
                    subs(:,3)=res{2};
                otherwise
                    error('Error. \nNo topic type selected');
            end
            phi=phi+sptensor(subs,vals',coreDims);
        end
    end
    
end