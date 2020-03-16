try
    asdSparse=csvread('cancerSparse.csv',1,1);
    asd=sptensor(asdSparse(:,1:2),asdSparse(:,4),[max(sparse(:,1)), ...
        max(sparse(:,2))],@max);

    %save data
    load('data/cancerHBTuckerGenData.mat');
    
    npats=size(asd,1); %number of articificial patients
    
    disp(options); %print options

    L=options.L;

    %adjustment if using constant L across dims
    if length(L)==1
        L=[L,1];
    else
        L(2)=1;
    end
    
    options.L=L;
    
    tree{2}=cell(1,1);
    tree{2}{1}=[];
    psi{2}=1;
        
    r=cell(2,1);
    r{1}=unique(paths(:,1:L(1)));
    r{2}=1;
        
    sparse=generatePatients(asd, npats, 1, psi, r, paths, tree, options);

    save('cancerHBTuckerGenDataGV.mat','sparse');
catch e
    display(e.identifier);
    display(e.message);
    for i=1:size(e.stack,1)
        display(e.stack(i,1));
    end
end