function [coreDims] = coreSize(modes,dims,r)
    %calculate dimensions of core
    coreDims=zeros(1,modes+1);
    coreDims(1)=dims(1);
    for i=1:modes
       %set core dimensions to the number of topics in each mode
       coreDims(i+1)=length(r{i});
    end
end