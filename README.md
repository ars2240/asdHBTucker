Simulink Project: asdHBTucker

Author: Adam Sandler
Date: 3/15/18

Files:
- AsdHBTucker.prj- Simulink Project file
- asdHBTucker.mat- data file with phi, psi, and tree from newest iteration
- asdHBTucker3.m- hierarchical Bayesian Tucker decomposition function
- asdSparse.csv- tensor of ASD data
- asdTens.m- main run file
- crp.m- draws new restaurant from Chinese Restaurant Process (CRP)
- drawCoreCon.m- draws the core tensor for the conditional Dirichlet distribution
- drawCoreUni.m- draws the core tensor for the uniform Dirichlet distribution
- drawZ.m- draws topics for a specific sample
- drawZc.c- C version of drawZ function
- drawZsc.c- C version of drawZs function
- drchrnd.m- generates probabilities from the Dirichlet distribution
- init_options.m- option initialization
- multi.m- draws a single sample from the multinomial distribution
- opt.m- separate file that computes tests for our optimization problem
- redrawTree.c- draws the tree from the CRP