function y = elems(ir,irEnd)
    %returns all values between two vectors
    y=[];
    for i=1:length(ir)
        y=[y, ir(i):irEnd(i)];
    end
end
