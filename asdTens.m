% asdSparse=csvread('asdSparse.csv',1,1);
% asd=sptensor(asdSparse(:,1:3),asdSparse(:,4));
% mex drawZscPar.c CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp";
% [phi, psi, tree]=asdHBTuckerPar2(asd,2,0.5);
asd=poissrnd(2,20,20,20);
asd=sptensor(asd);
mex drawZsc.c;
[phi, psi, tree]=asdHBTucker2(asd,2,0.5);
save('asdHBTucker.mat','phi','psi','tree');