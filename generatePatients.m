function sparse = generatePatients(x, prior, psi, opaths, tree, varargin)
    %generates new patients given trained parameters and information about
    %   the hierarchical structure of the model
    
    if length(varargin)==1
        varargin=varargin{1};
    end
    
    if length(varargin)==1 && iscell(varargin)
        options=varargin{1};
    elseif length(varargin)==2
        samples=varargin{1};
        options=varargin{2};
    elseif length(varargin)==3
        prob=varargin{1};
        samples=varargin{2};
        options=varargin{3};
    else
        error("Error. \nIncorrect number of inputs.");
    end
    
    %rng('shuffle'); %seed RNG
    rng(123);
    
    L=options.L;
    npats=options.npats;
    
    % store original tensor
    odims=size(x); %dimensions of tensor
    modes=length(size(x))-1;  %number of dependent modes
    
    cts=collapse(x,[2,3]);
    zind=cts==0;
    if sum(zind)>0
        ind=cell(modes+1,1);
        for i=1:modes
            ind{1+i}=1:odims(1+i);
        end
        ind{1}=find(cts>0)';
        x=x(tensIndex2(ind,odims));
        x=reshape(x,[length(ind{1}),odims(2:end)]);
        x=sptensor(x);
    end
    
    dims=size(x); %dimensions of tensor
    
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
            
            if ~exist('prob','var')
                prob=tree;
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
    
    if exist('samples','var') == 1
        r=cell(2,1);
        switch options.topicModel
            case 'IndepTrees'
                r{1}=unique(opaths(:,1:L(1)))';
                r{2}=unique(opaths(:,(L(1)+1):(sum(L))))';
            case 'PAM'
                for i=1:length(L)
                    r{i}=1:(sum(tpl{i}));
                end
            case 'None'
                for i=1:length(L)
                    r{i}=1:L(i);
                end
            otherwise
                error('Error. \nNo topic model type selected');
        end
        si=[dims,ones(1,modes)];
        for i=1:modes
            si(1+modes+i)=max(max(r{i}),max(samples(:,1+modes+i)));
        end
        cts=sptensor(samples,1,si);
        cphi=collapse(cts,1:(modes+1),@sum);
    end
        
    res=cell(modes,1);
    zr=cell(modes,1);
    y=zeros(1,modes);
    %zv = [];
    
    for i=1:npats
        
        %draw core tensor
        ind = zeros(L(1),modes);
        for j=1:modes
            res{j}=paths(i,1+sum(L(1:(j-1))):sum(L(1:j)));
            ind(:,j)=res{j};
            % res{1}=ismember(r{1},res{1});
        end
        if exist('samples','var') == 1
            if strcmp(options.topicType,'Level')
                alpha = prior + cphi(tensIndex2(ind,size(cphi)))';
            else
                alpha = prior + cphi(tensIndex2(res,size(cphi)))';
            end
        else
            alpha = prior;
        end
        [vals,~]=drchrnd(alpha,1,options);
        
        gvs=zeros(dims(2:end));
        
        %for each word
        for j=1:n(i)
            % draw z
            z=multi(vals);
            %zv=[zv z];
            
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
        if sum(gvs.vals) ~= n(i)
            disp(i);
        end
        
        t=[i*ones(size(y2,1),1),y2,v];
        
        sparse=[sparse; t];
        
    end
    
    %tabulate(zv);
end