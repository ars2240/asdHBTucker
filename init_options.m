function [ options ] = init_options( )
% Option initialization
% par = whether or not z's are computed in parallel
% time = whether or not time is printed
% maxIter = number of Gibbs sample iterations
% gam = hyper parameter(s) of CRP
% L = levels of hierarchical trees

options.par = 0;
options.time = 0;
options.maxIter = 10;
options.gam = 0.5;
options.L = 2;

end