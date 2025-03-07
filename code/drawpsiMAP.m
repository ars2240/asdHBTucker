function psi = drawpsiMAP(samples, dims, r, paths, options)
    %redraw matrices p(y|z)
    
    [~,cpsi,~] = counts(samples, dims, r, paths, [0,1,0], options);
	
    %compute psi
    modes = length(dims)-1;
    psi=cell(modes,1); %initialize
    for i=1:modes
        dim=dims(i+1);
        switch options.pType
            case 0
                prior=repelem(1/dim,dim);
            case 1
                prior=repelem(1,dim);
            case 2
                prior=repelem(2/dim,dim);
            otherwise
                error('Error. \nNo prior type selected');
        end
        psiT=cpsi{i}+prior';
        if psiT<=options.cutoff
            psiT=0;
        end
        psi{i}=psiT./sum(psiT);
    end
    
end
