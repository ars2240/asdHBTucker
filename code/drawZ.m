function samp = drawZ(samp,phi,psi,r)
    %draws single latent topic z
    %samp = sample row
    %phi = tucker decomposition tensor core tensor
    %psi = tucker decomposition matrices
    %r = restaurant list
    
    x=samp(1); %get evidence variable
    y=samp(2); %get response variable
    z=samp(5); %get other topic

    z=find(r{2}==z,1);
    if isempty(z)
        z=1;
    end
    vec=phi(x,:,z); %vector with set z
    if sum(vec)==0
        vec=phi(x,:,1); %vector with set z
    end

    pdf=psi{1}(y,:).*vec; %get probabilities
    if sum(pdf)==0
        z=1;
    else
        z=multi(pdf); %draw new z
    end

    samp(4)=r{1}(z); %set topic

    %sample other z
    y=samp(3); %get response variable
    vec=phi(x,z,:); %vector with set z
    vec=vec(:)';

    pdf=psi{2}(y,:).*vec; %get probabilities
    if sum(pdf)==0
        z=1;
    else
        z=multi(pdf); %draw new z
    end
    samp(5)=r{2}(z); %set topic
end
