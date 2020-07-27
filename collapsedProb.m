function p = collapsedProb(samples, oSamples, dims, r, paths, options)
    modes=length(dims)-1;
    
    if min(min(oSamples))>0
        [cphi,cpsi,~] = counts(oSamples, dims, r, paths, [1,1,0], options);
    else
        L=options.L;
        %adjustment if using constant L across dims
        if length(L)==1
            L=repelem(L,modes);
        end
        
        %initialize zero counts
        dimsM=zeros(modes,1);
        for i=1:modes
            dimsM(i)=length(r{i});
        end
        
        if options.sparse==0
            cphi=zeros([dims(1),dimsM]);
        else
            cphi=zeros([dims(1),L]);
        end
        
        cpsi=cell(modes,1);
        for i=1:modes
            cpsi{i}=zeros(dims(i+1),dimsM(i));
        end
    end
    
    s=size(cphi);
    indO=tensIndex2(oSamples(:,(2+modes):end),s(2:end));
    ind=tensIndex2(samples(:,(2+modes):end),s(2:end));
    t=prod(s(2:end));
    cphiT=tenmat(cphi,1);
    switch options.pType
        case 0
            prior=1/t;
        case 1
            prior=1;
        otherwise
            error('Error. \nNo prior type selected');
    end
    lp=log(double(cphiT(samples(:,1),:))+prior-(ind==indO));
    sh=[size(lp,1),s(2:end)];
    lp=reshape(lp,sh);
    for i=1:modes
        switch options.pType
            case 0
                prior=1/dims(1+i);
            case 1
                prior=1;
            otherwise
                error('Error. \nNo prior type selected');
        end
        lpT=log(cpsi{i}(samples(:,1+i),:)+prior-...
            (samples(:,1+i+modes)==oSamples(:,1+i+modes)));
        sh(2:end)=1;
        sh(1+modes)=size(lpT,2);
        lp=lp+reshape(lpT,sh);
    end
    p=exp(reshape(lp,[sh(1),t]));
    p=p(tensIndex2([(1:sh(1))',ind],[sh(1),t]))./sum(p,2);
end