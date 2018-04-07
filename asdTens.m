asdSparse=csvread('asdSparse.csv',1,1);
asd=sptensor(asdSparse(:,1:3),asdSparse(:,4));
options=init_options();
% mex drawZscPar.c CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp";
% mex drawZsc.c;
% asd=poissrnd(2,20,20,20);
% asd=sptensor(asd);
options.par=0;
options.maxIter = 10;
options.freq = 1;
% options.prior = 10;
[phi, psi, tree]=asdHBTucker3(asd,options);
save('asdHBTucker.mat','phi','psi','tree');