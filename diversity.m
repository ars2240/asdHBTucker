function div = diversity(psi)
    div = zeros(size(psi));
    for i=1:length(div)
        n=max(round(0.5*size(psi{i},1)),1);
        [~,n]=maxk(psi{i},n,1);
        div(i)=length(unique(n))/numel(n);
    end
end