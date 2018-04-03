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
% tol = tolerance for zeros of dirichlet distribution

options.par = 1;
options.time = 1;
options.print = 1;
options.maxIter = 1000;
options.freq = 100;
options.gam = 0.5;
options.L = 2;
options.prior = 0;
options.tol = eps();

end