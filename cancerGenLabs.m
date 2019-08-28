labs = csvread('cancerLabel.csv',1,1);
cts = countcats(categorical(labs));
cts = cts/sum(cts);

npats = length(labs);
r = mnrnd(npats, cts);
rlabs = [zeros(r(1),1);ones(r(2),1);2*ones(r(3),1);3*ones(r(4),1)];
rlabs = rlabs(randperm(length(rlabs)));


csvwrite('cancerGenLabel.csv',rlabs);