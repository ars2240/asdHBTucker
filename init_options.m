function [ options ] = init_options( )
% Option initialization
% par = whether or not z's are computed in parallel
% time = whether or not time is printed
% print = whether or not loglikelihood & perplexity are printed
% freq = how frequent loglikelihood & perplexity are printed
% maxIter = number of Gibbs sample iterations
% gam = hyper parameter(s) of CRP
% L = levels of hierarchical trees
% pType = prior type
%   0 = 1/N
%   1 = 1
%   2 = 2/N
% tol = tolerance for zeros of dirichlet distribution
% minA = all A's above this value will be drawn normally
% topicModel = type of hierarchical tree
% topicsPerLevel = number of topics in each level, used in PAM
% topicType = topic/tree correspondence
%   Cartesian = full
%   Level = diagonal
% treeReps = number of iterations of tree per large iteration
% btReps = number of iterations of Bayesian Tucker per large iteration
% collapsed = whether or not collapsed Gibbs sampler is used
% sparse = whether or not sparse matrices/tensors are used
% map = whether or not MAP estimate is used
% keepBest = whether or not best model is kept (as opposed to last)
% npats = number of articificial patients (for computing LL)
% cutoff = zeros out small values

options.par = 1;
options.time = 1;
options.print = 1;
options.maxIter = 1000;
options.freq = 10;
options.gam = 0.1;
options.L = 2;
options.pType = 0;
options.tol = 0;
options.minA = 0.1;
options.topicModel = 'IndepTrees';
options.topicsPerLevel = cell(2,1);
options.topicsPerLevel{1} = 10;
options.topicsPerLevel{2} = 10;
options.topicType = 'Cartesian';
options.treeReps = 1;
options.btReps = 1;
options.collapsed = 1;
options.sparse = 1;
options.map = 1;
options.keepBest = 0;
options.npats = 1000;
options.cutoff = 0;

end