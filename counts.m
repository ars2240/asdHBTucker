function [cphi,cpsi,ctree] = counts(samples, dims, r)
    %sampless = x, y, z values
    %dims = dimensions of tensor
    %r = restaurant lists
    
    si=[dims,max(max(r{1}),max(samples(:,4))),...
        max(max(r{2}),max(samples(:,5)))];
    % count of entire sample tensor
    cts=sptensor(samples,1,si);
    
    cpsi=cell(2,1);
    ctree=cell(2,1);
    
    % count of topics, patient
    cphi=collapse(cts,[2,3],@sum);
    cphi=full(cphi);
    cphi=cphi(:,:,:);
    cphi=cphi(:,r{1},r{2});
    
    % count of patient, GV, GV topic
    ctree{1}=collapse(cts,[3,5],@sum);
    ctree{1}=full(ctree{1});
    ctree{1}=ctree{1}(:,:,:);
    
    % count of patient, p'way, p'way topic
    ctree{2}=collapse(cts,[2,4],@sum);
    ctree{2}=full(ctree{2});
    ctree{2}=ctree{2}(:,:,:);
    
    % count of GV, GV topic
    cpsi{1}=collapse(ctree{1},1,@sum);
    cpsi{1}=cpsi{1}(:,:);
    
    % count of p'way, p'way topic
    cpsi{2}=collapse(ctree{2},1,@sum);
    cpsi{2}=cpsi{2}(:,:);
    
    % convert to normal MatLab tensor
    cphi=double(cphi);
    ctree{1}=double(ctree{1});
    ctree{2}=double(ctree{2});
    cpsi{1}=double(cpsi{1});
    cpsi{2}=double(cpsi{2});

end
