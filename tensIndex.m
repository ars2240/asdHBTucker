function indices = tensIndex(r)
    modes=length(r);
    if ~iscell(r)
        dims2=r;
        r=cell(modes,1);
        for i=1:modes
            r{i}=1:dims2(i);
        end
    else
        dims2=zeros(1,modes);
        for i=1:modes
            dims2(i)=length(r{i});
        end
    end
    
    indices=zeros(prod(dims2),modes);
    for i=1:modes
        indices(:,i)=repmat(repelem(r{i},prod(dims2(1:(i-1)))),...
            [1,prod(dims2((i+1):modes))]);
    end
end