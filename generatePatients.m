function sparse = generatePatients(x, npats, prior, psi, r, opaths, tree, varargin)
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
    
    %adjustment if using constant L across dims
    if length(L)==1
        L=repelem(L,2);
    end
    
    dims=size(x); %dimensions of tensor
    
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
                for j=1:2
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
            paths=ones(dims(1),sum(L));
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

            ttpl=[sum(tpl{1}),sum(tpl{2})];
            %initialize restaurant list
            r=cell(2,1);
            r{1}=1:(ttpl(1));
            r{2}=1:(ttpl(2));
            
            %old counts
            [~,ocpsi,~] = counts(samples, ...
                [max(samples(:,1)), dims(2:3)], r, paths, [0,1,0], options);
            
            ctree=cell(2,1);
            ctree{1}=zeros(dims(1),dims(2),ttpl(1));
            ctree{2}=zeros(dims(1),dims(3),ttpl(2));
            
            [paths,~,~] = newPAM(dims,ocpsi,ctree,paths,tpl,prob,options);
        case 'None'
            paths=repmat([1:L(1),1:L(2)],npats,1);
            r=cell(2,1); %initialize
            r{1}=1:L(1);
            r{2}=1:L(2);
        otherwise
            error('Error. \nNo topic model type selected');
    end
    
    sparse=[];
    
    if length(prior)==1
        prior=repmat(prior,1,L(1));
    end
    if length(prior)==L(1) && strcmp(options.topicType,'Cartesian')
        prior=repmat(prior,1,L(2));
    end
        
    
    for i=1:npats
        
        %draw core tensor
        res{1}=paths(i,1:L(1));
        % res{1}=ismember(r{1},res{1});
        res{2}=paths(i,(1+L(1)):(L(1)+L(2)));
        % res{2}=ismember(r{2},res{2});
        [vals,~]=drchrnd(prior,1,options);
        
        gvs=zeros(size(psi{1},1),size(psi{2},1));
        
        %for each word
        for j=1:n(i)
            % draw z
            z=multi(vals);
            
            switch options.topicType
                case 'Cartesian'
                    zt=mod(z-1,L(1))+1;
                    z1=res{1}(zt);
                    zt=floor((z-1)/L(1))+1;
                    z2=res{2}(zt);
                case 'Level'
                    z1=res{1}(z);
                    z2=res{2}(z);
                otherwise
                    error('Error. \nNo topic type selected');
            end
            
            %draw y
            y1=multi(psi{1}(:,z1));
            y2=multi(psi{2}(:,z2));
            
            %add count
            gvs(y1,y2)=gvs(y1,y2)+1;
        end
        
        [y1,y2,v]=find(gvs);
        
        t=[i*ones(size(y1,1),1),y1,y2,v];
        
        sparse=[sparse; t];
        
    end
end