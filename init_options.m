function [ options ] = init_options( )
% Option initialization
% par = whether or not z's are computed in parallel
% time = whether or not time is printed
% print = whether or not loglikelihood & perplexity are printed
% freq = how frequent loglikelihood & perplexity are printed
% maxIter = number of Gibbs sample iterations
% gam = hyper parameter(s) of CRP
% L = levels of hierarchical trees
% prior = value to add to prior
% pType = prior type
%   0 = 1/N
%   1 = 1
% tol = tolerance for zeros of dirichlet distribution
% minA = all A's above this value will be drawn normally
% topicModel = type of hierarchical tree
% topicsPerLevel = number of topics in each level, used in PAM

options.par = 1;
options.time = 1;
options.print = 1;
options.maxIter = 1000;
options.freq = 100;
options.gam = 0.1;
options.L = 2;
options.prior = 0;
options.pType = 1;
options.tol = 0;
options.minA = 0.1;
options.topicModel = 'IndepTrees';
options.topicsPerLevel = cell(2,1);
options.topicsPerLevel{1} = 10;
options.topicsPerLevel{2} = 10;

end