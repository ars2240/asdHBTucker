function y = elems(ir,irEnd)
    %returns all values between two vectors
    y=[];
    for i=1:lendth(ir)
        y=[y, ir(i):irEnd(i)];
    end
end