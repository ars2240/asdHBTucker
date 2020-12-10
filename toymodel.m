asd=sptensor([2 2 2]);
asd(1,1,1)=2; asd(2,2,2)=1;

options=init_options();
% mex drawZscPar.c CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp";
options.gam = .5;
options.L = 2;
%options.topicType = 'Level';
%options.topicModel = 'PAM';
options.par = 0;
options.maxIter = 100;
options.pType = 0;
% options.treeReps = 5;
% options.btReps = 5;
options.topicsPerLevel{1}=tpl;
options.topicsPerLevel{2}=tpl;
% options.collapsed = 0;
options.keepBest = 1;
options.time = 0;
options.print = 1;
% options.cutoff = 0.1;
% options.sparse = 0;
    
[phi, psi, tree, samples, paths, ll,~] = asdHBTucker3(asd,options);
x= ttm(tensor(phi), psi{1}', 2);
x= ttm(x, psi{2}', 3);