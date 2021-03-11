function pdf = logPDF(pdf, rList, cts1, cts2, gcts, prior)
    w=prior*length(rList);
    pdf=pdf+gammaln(sum(cts1,1)+w);
    pdf=pdf-sum(gammaln(cts1+prior),1); %test
    pdf=pdf+sum(gcts,1);
    pdf=pdf-gammaln(sum(cts2,1)+w);
end