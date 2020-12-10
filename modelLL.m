function [LL, zLL, treeLL] = modelLL(phi, psi, samples, paths, r, options)

    gam=options.gam;
    
    treeLL=0;
    switch options.topicModel
        case 'IndepTrees'
            for i=1:size(paths,2)
                g=gam(1);
                t=tabulate(paths(:,i));
                t=t(:,2); t=t(t>0);
                treeLL=treeLL+(length(t)-1)*log(g)+sum(gammaln(t))...
                    +gammaln(1+g)-gammaln(sum(t)+g);
            end
        case 'PAM'
            error('Error. \nNot implemented yet');
        case 'None'
            treeLL=0;
        otherwise
            error('Error. \nNo topic model type selected');
    end
    
    [~,samples(:,4)]=ismember(samples(:,4),r{1});
    [~,samples(:,5)]=ismember(samples(:,5),r{2});
    zLL=sum(log(phi(sub2ind(size(phi), samples(:,1), samples(:,4), samples(:,5)))))...
        +sum(log(psi{1}(sub2ind(size(psi{1}), samples(:,2), samples(:,4)))))...
        +sum(log(psi{2}(sub2ind(size(psi{2}), samples(:,3), samples(:,5)))));
    
    LL=zLL+treeLL;
    
end