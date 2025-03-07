function pdf = adjPDF(lP)
    t=1; m=t*mean(lP);
    pdf=exp(lP-m);
    while sum(pdf)==0 || isinf(sum(pdf)) || isnan(sum(pdf))
        % handling of underflow error
        if sum(pdf)==0
            t=t*1.5;
        elseif isnan(sum(pdf))
            display(lP);
        else
            t=t/2;
        end
        m=t*mean(lP);
        pdf=exp(lP-m);
    end
end
