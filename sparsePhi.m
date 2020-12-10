function phi = sparsePhi(phi, coreDims, paths, options)
    modes=length(coreDims)-1; L=options.L;
    phiT=sptensor([],[],coreDims);
    for i=1:coreDims(1)
        res=cell(modes,1);
        for j=1:modes
            res{j}=paths(i,(1+sum(L(1:(j-1)))):sum(L(1:j)));
        end
        switch options.topicType
            case 'Cartesian'
                len = prod(L);
                subs=[repmat(i,[len,1]),tensIndex(res)];
                vals=reshape(phi(i,:),[len,1]);
            case 'Level'
                subs=zeros(L(1),1+modes);
                subs(:,1)=i;
                for j=1:modes
                    subs(:,j+1)=res{j};
                end
                vals=squeeze(phi(i,:));
                vals=vals(vals>0)';
            otherwise
                error('Error. \nNo topic type selected');
        end   
        phiT=phiT+sptensor(subs,vals,coreDims);
    end
    phi=phiT;
end