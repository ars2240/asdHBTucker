function pdf = getPDF(c, rList, cts1, cts2, gcts, prior)
    eps=1e-12;
    pdf=log(c+eps); %take log to prevent overflow
    lP=logPDF(pdf, rList, cts1, cts2, gcts, prior);
    pdf=adjPDF(lp);
end