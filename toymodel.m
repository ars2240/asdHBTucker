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
%options.topicModel = 'None';
options.par = 0;
options.maxIter = 100;
options.pType = 1;
% options.treeReps = 5;
% options.btReps = 5;
tpl=4;
options.topicsPerLevel{1}=tpl;
options.topicsPerLevel{2}=tpl;
% options.collapsed = 0;
options.keepBest = 2;
options.time = 0;
options.print = 1;
% options.cutoff = 0.1;
% options.sparse = 0;

asdC=collapse(asd,[2,3]);

correct = 0;
for i=1:n
    if strcmp(options.topicModel,'PAM')
        [~, ~, ~, ~, ~, ~, o, ~, ~] = asdHBTucker3(asd,options);
    else
        [~, ~, ~, ~, ~, o, ~, ~] = asdHBTucker3(asd,options);
    end
    KB = o.best; %display(KB.iter);
    phi=KB.phi; psi=KB.psi; samples=KB.samples;
    paths=KB.paths;% options.gam=KB.gamma;
    phi=phi.*asdC;
    x= ttm(tensor(phi), psi, [2,3]);
    disp(norm(asd-x));
    correct=correct+int8(optimal(samples));
end
disp(correct);

function b=optimal(samples)
    n = size(samples,1)-1;
    b = samples(1,4) == samples(2,4) && samples(1,5) == samples(2,5) ...
        && length(unique(samples(2:end,4))) == n ...
        && length(unique(samples(2:end,5))) == n;
end