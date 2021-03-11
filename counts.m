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
    
    modes=length(dims)-1;
    si=[dims,ones(1,modes)];
    for i=1:modes
        if iscell(r)
            si(1+modes+i)=max(max(r{i}),max(samples(:,1+modes+i)));
        else
            si(1+modes+i)=max(max(r),max(samples(:,1+modes+i)));
        end
    end
    % count of entire sample tensor
    cts=sptensor(samples,1,si);
    
    cpsi=cell(modes,1);
    ctree=cell(modes,1);
    
    % count of topics, patient
    if argsout(1)~=0
        cphi=collapse(cts,2:(modes+1),@sum);
        ind=cell(modes+1,1);
        ind{1}=1:size(cphi,1);
        cs=size(cphi);
        for i=1:modes
            if iscell(r)
                ind{i+1}=r{i};
                cs(i+1)=length(r{i});
            else
                ind{i+1}=r;
                cs(i+1)=length(r);
            end
        end
        cphi=cphi(tensIndex2(ind,size(cphi)));
        
        % convert to normal MatLab tensor
        if options.sparse==0
            cphi=double(cphi);
        else
            L=options.L;
            %adjustment if using constant L across dims
            if length(L)==1
                L=repelem(L,modes);
            end

            cphiS=zeros([dims(1),L]);
            for i=1:dims(1)
                ind=cell(modes+1,1);
                ind{1}=i;
                ind2=ind;
                %get restaurants for patient
                for j=1:modes
                    if iscell(r)
                        [~,ind{j+1}]=ismember(paths(i,...
                            (1+sum(L(1:(j-1)))):sum(L(1:j))),r{j});
                    else
                        [~,ind{j+1}]=ismember(paths(i,...
                            (1+sum(L(1:(j-1)))):sum(L(1:j))),r);
                    end
                    ind2{j+1}=1:L(j);
                end
                cphiS(tensIndex2(ind2,[dims(1),L]))=cphi(tensIndex2(ind,cs));
            end
            cphi=cphiS;
        end
    else
        cphi=[];
    end
    
    if argsout(3)~=0 || argsout(2)~=0
        mlist=1:modes; %vector of all modes
        
        for i=1:modes
            % convert to normal MatLab tensor
            ml2=mlist(mlist~=i);
            ctree{i}=collapse(cts,[ml2+1,ml2+modes+1],@sum);
            if options.sparse==0
                ctree{i}=full(ctree{i});
            end
        end
    end
    
    if argsout(2)~=0
        
        for i=1:modes
            % count of mode, topic
            cpsi{i}=collapse(ctree{i},1,@sum);
            % convert to normal MatLab tensor
            cpsi{i}=double(cpsi{i});
        end
    end

end
