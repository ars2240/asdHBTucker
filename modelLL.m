function [LL, zLL, treeLL] = modelLL(phi, psi, samples, paths, r, varargin)

    if length(varargin)==1
        varargin=varargin{1};
    end
    
    if length(varargin)==1
        if iscell(varargin)
            options=varargin{1};
        else
            options=varargin;
        end
    elseif length(varargin)==2
        prob=varargin{1};
        options=varargin{2};
    else
        error("Error. \nIncorrect number of inputs.");
    end
    
    gam=options.gam; L=options.L;
    
    treeLL=0;
    switch options.topicModel
        case 'IndepTrees'
            for i=1:size(paths,2)
                if length(gam)>1
                    g=gam(find(i<=cumsum(L),1));
                else
                    g=gam;
                end
                t=tabulate(paths(:,i));
                t=t(:,2); t=t(t>0);
                treeLL=treeLL+(length(t)-1)*log(g)+sum(gammaln(t))...
                    +gammaln(1+g)-gammaln(sum(t)+g);
            end
        case 'PAM'
            tpl = options.topicsPerLevel;
            for i=1:size(prob,1)
                for j=1:size(prob,2)
                    if i~=size(prob,1) && j~=size(prob,2)
                        in = paths(:,j+(i-1)*L(1))-sum(tpl{i}(1:(j-1)));
                        out = paths(:,j+(i==2)+mod(i,2)*L(1))...
                            -sum(tpl{mod(i,2)+1}(1:(j-(i~=2))));
                        treeLL=treeLL+sum(log(prob{i,j}(sub2ind(...
                            size(prob{i,j}), in, out))));
                    end
                end
            end
        case 'None'
            treeLL=0;
        otherwise
            error('Error. \nNo topic model type selected');
    end
    
    if iscell(r)
        [~,samples(:,4)]=ismember(samples(:,4),r{1});
        [~,samples(:,5)]=ismember(samples(:,5),r{2});
    else
        [~,samples(:,4)]=ismember(samples(:,4),r);
        [~,samples(:,5)]=ismember(samples(:,5),r);
    end
    zLL=sum(log(phi(sub2ind(size(phi), samples(:,1), samples(:,4), samples(:,5)))))...
        +sum(log(psi{1}(sub2ind(size(psi{1}), samples(:,2), samples(:,4)))))...
        +sum(log(psi{2}(sub2ind(size(psi{2}), samples(:,3), samples(:,5)))));
    
    LL=zLL+treeLL;
    
    if ~isfinite(LL)
        error('Error. \nUndifined LL');
    end
end