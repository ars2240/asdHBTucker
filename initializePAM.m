function [paths,tpl,prob,r,LL,ent]=initializePAM(dims,options)
    %Initializes PAM
    %Inputs
    % L = levels of hierarchical tree
    % dims = dimensions of tensor
    % options =
    %   topicsPerLevel = number of topics per level of PAM
    %   dominantMode = which mode is dominant
    %Outputs
    % path = tree paths
    % tpl = number of topics per level of PAM
    % prob = probability tree
    % r = restaurant lists
    % LL = log-likelihood
    % ent = entropy
    
    [tpl, r]=initPAM(dims,options);
    modes=length(dims)-1;  %number of dependent modes
    
    LL=0; %initialize log-likelihood
    ent=0; %initialize entropy
    
    L=options.L;
    %adjustment if using constant L across dims
    if length(L)==1
        L=repelem(L,modes);
    end
    
    %initialize probability tree
    prob=cell(modes,L(1));
    
    if options.collapsed==1
        %i=1:(modes-1);
        for i=1:(modes-1)
            for j=1:L(1)
                len=tpl{i+1}(j);
                parents=tpl{i}(j);
                prob{i,j}=repelem(1/len,len,parents);
            end
        end

        %i=modes;
        for j=1:(L(1)-1)
            len=tpl{1}(j+1);
            parents=tpl{modes}(j);
            prob{modes,j}=repelem(1/len,len,parents);
        end
    else
        %i=1:(modes-1);
        for i=1:(modes-1)
            for j=1:L(1)
                len=tpl{i+1}(j);
                switch options.pType
                    case 0
                        prior=repelem(1/len,len);
                    case 1
                        prior=repelem(1,len);
                    otherwise
                        error('Error. \nNo prior type selected');
                end
                parents=tpl{i}(j);
                [prob{i,j},p]=drchrnd(prior,parents,options);
                LL=LL+sum(log(p));
                ent=ent+entropy(p);
            end
        end

        %i=modes;
        for j=1:(L(1)-1)
            len=tpl{1}(j+1);
            switch options.pType
                case 0
                    prior=repelem(1/len,len);
                case 1
                    prior=repelem(1,len);
                otherwise
                    error('Error. \nNo prior type selected');
            end
            parents=tpl{modes}(j);
            [prob{modes,j},p]=drchrnd(prior,parents,options);
            LL=LL+sum(log(p));
            ent=ent+entropy(p);
        end
    end
    
    %initialize paths matrix
    paths=zeros(dims(1),sum(L));
    paths(:,1)=1; %sit at root table
    
    for p=1:dims(1)
        res=1;
        
        %modes 2:modes, level 1
        j=1;
        for i=2:modes
            pdf=prob{mod(i,modes)+1,j-(i==1)}(res,:);
            res=multi(pdf);
            top=res+sum(tpl{i}(1:(j-1)));
            paths(p,j+(i-1)*L(1))=top;
            LL=LL+log(pdf(res));
            ent=ent+entropy(pdf(res));
        end
        
        %other modes
        for j=2:L(1)
            for i=1:modes
                pdf=prob{mod(i,modes)+1,j-(i==1)}(res,:);
                res=multi(pdf);
                top=res+sum(tpl{i}(1:(j-1)));
                paths(p,j+(i-1)*L(1))=top;
                LL=LL+log(pdf(res));
                ent=ent+entropy(pdf(res));
            end
        end
    end
    
end