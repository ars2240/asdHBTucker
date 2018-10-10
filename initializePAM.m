function [paths,tpl,prob,r,LL,ent]=initializePAM(L,dims,options)
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
    
    if L(1)~=L(2)
        error("Error. \nLevels do not match");
    end
    
    %reformat topicsPerLevel as cell of vectors of correct length
    if iscell(options.topicsPerLevel)
        tpl=options.topicsPerLevel;
        if length(tpl)~=2
            error("Error. \nNumber of cells !=2");
        end
        if length(tpl{1})==1
            tpl{1}=[1,repelem(tpl{1}(1),L(1)-1)];
        elseif length(tpl{1})~=(L(1)-1)
            error("Error. \nInvalid length of topics per level");
        end
        if length(tpl{2})==1
            tpl{2}=repelem(tpl{2}(1),L(2));
        elseif length(tpl{2})~=(L(2)-1)
            error("Error. \nInvalid length of topics per level");
        end
    else
        tplV=options.topicsPerLevel;
        tpl=cell(2,1);
        if length(tplV)==1
            tpl{1}=[1,repelem(tplV(1),L(1)-1)];
            tpl{2}=repelem(tplV(1),L(2));
        elseif length(tplV)==2
            tpl{1}=[1,repelem(tplV(1),L(1)-1)];
            tpl{2}=repelem(tplV(2),L(2));
        elseif length(tplV)==L(1)
            tpl{1}=tplV;
            tpl{2}=tplV;
        elseif length(tplV)==2*(L(1))
            tpl{1}=tplV(1:L(1));
            tpl{2}=tplV((L(1)+1):2*L(1));
        else
            error("Error. \nInvalid length of topics per level");
        end
    end
    
    %initialize restaurant list
    r=cell(2,1);
    r{1}=1:(sum(tpl{1}));
    r{2}=1:(sum(tpl{2}));
    
    LL=0; %initialize log-likelihood
    ent=0; %initialize entropy
    
    %initialize probability tree
    prob=cell(2,L(1));
    
    if options.collapsed==1
        %i=1;
        for j=1:L(1)
            len=tpl{2}(j);
            parents=tpl{1}(j);
            prob{1,j}=repelem(1/len,len,parents);
        end

        %i=2;
        for j=1:(L(1)-1)
            len=tpl{1}(j+1);
            parents=tpl{2}(j);
            prob{2,j}=repelem(1/len,len,parents);
        end
    else
        %i=1;
        for j=1:L(1)
            len=tpl{2}(j);
            switch options.pType
                case 0
                    prior=repelem(1/len,len);
                case 1
                    prior=repelem(1,len);
                otherwise
                    error('Error. \nNo prior type selected');
            end
            parents=tpl{1}(j);
            [prob{1,j},p]=drchrnd(prior,parents,options);
            LL=LL+sum(log(p));
            ent=ent+entropy(p);
        end

        %i=2;
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
            parents=tpl{2}(j);
            [prob{2,j},p]=drchrnd(prior,parents,options);
            LL=LL+sum(log(p));
            ent=ent+entropy(p);
        end
    end
    
    %initialize paths matrix
    paths=zeros(dims(1),sum(L));
    paths(:,1)=1; %sit at root table
    
    for p=1:dims(1)
        res=1;
        
        %mode 2, level 1
        j=1;
        i=2;
        pdf=prob{mod(i,2)+1,j-(i==1)}(res,:);
        res=multi(pdf);
        top=res+sum(tpl{i}(1:(j-1)));
        paths(p,j+(i==2)*L(1))=top;
        LL=LL+log(pdf(res));
        ent=ent+entropy(pdf(res));
        
        %other modes
        for j=2:L(1)
            for i=1:2
                pdf=prob{mod(i,2)+1,j-(i==1)}(res,:);
                res=multi(pdf);
                top=res+sum(tpl{i}(1:(j-1)));
                paths(p,j+(i==2)*L(1))=top;
                LL=LL+log(pdf(res));
                ent=ent+entropy(pdf(res));
            end
        end
    end
    
end