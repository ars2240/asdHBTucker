function [tpl, r]=initPAM(dims, options)

    L=options.L;
    modes=length(dims)-1;  %number of dependent modes
    
    %adjustment if using constant L across dims
    if length(L)==1
        L=repelem(L,modes);
    end
    
    if any(diff(sort(L)))
        error("Error. \nLevels do not match");
    end

    %reformat topicsPerLevel as cell of vectors of correct length
    if iscell(options.topicsPerLevel)
        tpl=options.topicsPerLevel;
        if length(tpl)~=modes
            error("Error. \nNumber of cells != number of modes");
        end
        if length(tpl{1})==1
            tpl{1}=[1,repelem(tpl{1}(1),L(1)-1)];
        elseif length(tpl{1})~=L(1)
            error("Error. \nInvalid length of topics per level");
        end
        for i=2:modes
            if length(tpl{i})==1
                tpl{i}=repelem(tpl{i}(1),L(i));
            elseif length(tpl{i})~=L(i)
                error("Error. \nInvalid length of topics per level");
            end
        end
    else
        tplV=options.topicsPerLevel;
        tpl=cell(modes,1);
        if length(tplV)==1
            tpl{1}=[1,repelem(tplV(1),L(1)-1)];
            for i=2:modes
                tpl{i}=repelem(tplV(1),L(i));
            end
        elseif length(tplV)==modes
            tpl{1}=[1,repelem(tplV(1),L(1)-1)];
            for i=2:modes
                tpl{i}=repelem(tplV(i),L(i));
            end
        elseif length(tplV)==L(1)
            for i=1:modes
                tpl{i}=tplV;
            end
        elseif length(tplV)==sum(L)
            for i=1:modes
                tpl{i}=tplV((sum(L(1:(i-1)))+1):sum(L(1:i)));
            end
        else
            error("Error. \nInvalid length of topics per level");
        end
    end

    %initialize restaurant list
    r=cell(modes,1);
    for i=1:modes
        r{i}=1:(sum(tpl{i}));
    end
end
