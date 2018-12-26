function [cphi,cpsi,ctree] = counts(samples, dims, r, varargin)
    %sampless = x, y, z values
    %dims = dimensions of tensor
    %r = restaurant lists
    
    switch length(varargin)
        case 1
            options=varargin;
            argsout=ones(3,1);
        case 2
            options=varargin{2};
            if options.sparse~=0
                paths=varargin{1};
                argsout=ones(3,1);
            else
                argsout=varargin{1};
            end
        case 3
            paths=varargin{1};
            argsout=varargin{2};
            options=varargin{3};
        otherwise
            error("Error. \nIncorrect number of inputs.");
    end
    
    si=[dims,max(max(r{1}),max(samples(:,4))),...
        max(max(r{2}),max(samples(:,5)))];
    % count of entire sample tensor
    cts=sptensor(samples,1,si);
    
    cpsi=cell(2,1);
    ctree=cell(2,1);
    
    % count of topics, patient
    if argsout(1)~=0
        cphi=collapse(cts,[2,3],@sum);
        cphi=cphi(:,r{1},r{2});
        
        % convert to normal MatLab tensor
        if options.sparse==0
            cphi=double(cphi);
        else
            L=options.L;
            %adjustment if using constant L across dims
            if length(L)==1
                L=repelem(L,2);
            end

            cphiS=zeros(dims(1),L(1),L(2));
            for i=1:dims(1)
                r1=paths(i,1:L(1));
                r2=paths(i,(L(1)+1):(sum(L)));
                cphiS(i,:,:)=cphi(i,r1,r2);
            end
            cphi=cphiS;
        end
    else
        cphi=[];
    end
    
    if argsout(3)~=0 || argsout(2)~=0
        % count of patient, GV, GV topic
        ctree{1}=collapse(cts,[3,5],@sum);
        
        % count of patient, p'way, p'way topic
        ctree{2}=collapse(cts,[2,4],@sum);
        
        % convert to normal MatLab tensor
        if options.sparse==0
            ctree{1}=double(ctree{1});
            ctree{2}=double(ctree{2});
        end
    end
    
    if argsout(2)~=0
        % count of GV, GV topic
        cpsi{1}=collapse(ctree{1},1,@sum);

        % count of p'way, p'way topic
        cpsi{2}=collapse(ctree{2},1,@sum);
        
        % convert to normal MatLab tensor
        cpsi{1}=double(cpsi{1});
        cpsi{2}=double(cpsi{2});
    end

end
