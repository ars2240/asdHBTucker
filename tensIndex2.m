function tindices = tensIndex2(indices, dims)
    %outputs tensor of indices
    if iscell(indices)
        indices=tensIndex(indices);
    end
    modes=size(indices,2);
    tindices=indices(:,1);
    for i=2:modes
        tindices=tindices+prod(dims(1:(i-1)))*(indices(:,i)-1);
    end
end