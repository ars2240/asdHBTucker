asd=sptensor([5 5 5]);
asd(1,1,1)=2; asd(2,2,2)=1; asd(3,3,3)=1; asd(4,4,4)=1; asd(5,5,5)=1;
sparse=[asd.subs, asd.vals];
save('toy.mat', 'sparse');

n=12; % different iterates

options=init_options();
% mex drawZscPar.c CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp";
options.gam = 1;
options.L = 2;
%options.topicType = 'Level';
%options.topicModel = 'PAM';
options.par = 0;
options.maxIter = 100;
options.pType = 0;
% options.treeReps = 5;
% options.btReps = 5;
tpl=2;
options.topicsPerLevel{1}=tpl;
options.topicsPerLevel{2}=tpl;
% options.collapsed = 0;
options.keepBest = 1;
options.time = 0;
options.print = 1;
% options.cutoff = 0.1;
% options.sparse = 0;

asdC=collapse(asd,[2,3]);

for i=1:n
    [~, ~, ~, ~, ~, o, ~, ~] = asdHBTucker3(asd,options);
    KB = o.best; display(KB.iter);
    phi=KB.phi; psi=KB.psi; samples=KB.samples;
    paths=KB.paths;% options.gam=KB.gamma;
    phi=phi.*asdC;
    x= ttm(tensor(phi), psi, [2,3]);
    disp(norm(asd-x));
    display(samples);
end