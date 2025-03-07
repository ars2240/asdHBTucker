function [psi,LL,ent] = drawpsi(dims, modes, samples, r, options)
    %redraw matrices p(y|z)
    
    coreDims=coreSize(modes, dims, r);
    
    psi=cell(modes,1); %initialize
    
    for i=1:modes
        psi{i}=zeros(dims(i+1),coreDims(i+1));
    end
    
    LL=0; ent=0;
    
    for i=1:modes
        [u,~,ir]=unique(samples(:,1+modes+i));
        samps=accumarray(ir,1:size(samples,1),[],@(w){samples(w,:)});
        dim=dims(i+1);
        [~,loc]=ismember(r{i},u);
        psiT=zeros(dim,coreDims(i+1));
        for j=1:coreDims(i+1)
            %draw values from dirichlet distribution with uniform prior
            %plus counts of occurances of both y & z
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
            if loc(j)~=0
                prior=prior+histc(samps{loc(j)}(:,i+1)',1:dim);
            end
            [psiT(:,j),p]=drchrnd(prior,1,options);
            LL=LL+sum(p);
            ent=ent+entropy(exp(p));
        end
        psi{i}=psiT;
    end
