function sparse = generatePatients(x, prior, psi, opaths, tree, varargin)
    %generates new patients given trained parameters and information about
    %   the hierarchical structure of the model
    
    varargin=varargin{1};
    if length(varargin)==1
        if iscell(varargin)
            options=varargin{1};
        else
            options=varargin;
        end
    elseif length(varargin)==3
        prob=varargin{1};
        samples=varargin{2};
        options=varargin{3};
    else
        error("Error. \nIncorrect number of inputs.");
    end
    
    rng('shuffle'); %seed RNG
    
    L=options.L;
    npats=options.npats;
    
    dims=size(x); %dimensions of tensor
    modes=length(dims)-1;   %number of dependent modes
    
    %adjustment if using constant L across dims
    if length(L)==1
        L=repelem(L,modes);
    end
    
    %compute variant counts
    x=sptenmat(x, 1);
    x=double(x);
    cts=sum(x, 2);
    cts=full(cts);
    
    % fit poisson distribution
    lambda=poissfit(cts);
    
    %draw number of words
    n=poissrnd(lambda,npats,1);
    
    %initialize tree
    switch options.topicModel
        case 'IndepTrees'
            paths=ones(npats,sum(L));
            for i=1:npats
                for j=1:modes
                    col=(j-1)*L(1); %starting column
                    curRes=1; %set current restaurant as root
                    for k=2:L(j)
                        %get restaurant list
                        rList=tree{j}{curRes};
                        rList=sort(rList);

                        %compute CRP part of pdf
                        pdf=histc(opaths(:,col+k)',rList);
                        pdf=pdf/sum(pdf); %normalize
                        
                        next=multi(pdf);
                        nextRes=rList(next);

                        paths(i,col+k)=nextRes; %sit at table
                        
                        curRes=nextRes;
                    end
                end
            end
        case 'PAM'
            paths=ones(npats,sum(L));
            [tpl, r]=initPAM(dims,options);

            %old counts
            [~,ocpsi,~] = counts(samples, ...
                [max(samples(:,1)), dims(2:end)], r, paths, [0,1,0], options);
            
            ctree=cell(modes,1);
            for i=1:modes
                ctree{i}=zeros(npats,dims(i+1),length(r{i}));
            end
            
            [paths,~,~] = newPAM(dims,ocpsi,ctree,paths,tpl,prob,options);
        case 'None'
            r=cell(modes,1); %initialize
            path=zeros(1,sum(L));
            for i=1:modes
                r{i}=1:L(i);
                path(1+sum(L(1:(i-1))):sum(L(1:i)))=1:L(i);
            end         
            paths=repmat(path,npats,1);
        otherwise
            error('Error. \nNo topic model type selected');
    end
    
    sparse=[];
    
    if length(prior)==1
        prior=repmat(prior,1,L(1));
    end
    if length(prior)==L(1) && strcmp(options.topicType,'Cartesian')
        prior=repmat(prior,1,prod(L(2:modes)));
    end
        
    res=cell(modes,1);
    zr=cell(modes,1);
    y=zeros(1,modes);
    
    for i=1:npats
        
        %draw core tensor
        for j=1:modes
            res{j}=paths(i,1+sum(L(1:(j-1))):sum(L(1:j)));
            % res{1}=ismember(r{1},res{1});
        end
        [vals,~]=drchrnd(prior,1,options);
        
        gvs=zeros(dims(2:end));
        
        %for each word
        for j=1:n(i)
            % draw z
            z=multi(vals);
            
            switch options.topicType
                case 'Cartesian'
                    for k=1:modes
                        zt=mod(z-1,prod(L(1:k)));
                        zt=floor(zt/prod(L(1:(k-1))))+1;
                        zr{k}=res{k}(zt);
                    end
                case 'Level'
                    for k=1:modes
                        zr{k}=res{k}(z);
                    end
                otherwise
                    error('Error. \nNo topic type selected');
            end
            
            %draw y
            for k=1:modes
                y(k)=multi(psi{k}(:,zr{k}));
            end
            
            %add count
            gvs(tensIndex2(y, dims(2:end)))=gvs(tensIndex2(y, dims(2:end)))+1;
        end
        
        gvs=sptensor(gvs);
        y2=gvs.subs;
        v=gvs.vals;
        
        t=[i*ones(size(y2,1),1),y2,v];
        
        sparse=[sparse; t];
        
    end
end