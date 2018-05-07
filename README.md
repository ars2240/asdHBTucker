Simulink Project: asdHBTucker

Author: Adam Sandler
Date: 5/7/18

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
- elems.m- returns all values between two vectors
- entropy.m- calculates entropy of probability vector
- init_options.m- option initialization
- initializeTree.m- initializes hierarchical tree from the CRP
- ldaTests.R- computes LDA decomposition baseline tests
- logistic_reg.py- predict using logistic regression with regularization
- logisticReg.m- uses a logistic regression model to learn & predict ASD
- logisticRegPCA.m- predict using logistic regression, using first X PCs
- multi.m- draws a single sample from the multinomial distribution
- opt.m- separate file that computes tests for our optimization problem
- redrawTree.c- draws the tree from the CRP
- rgamma.c- samples small-shape gamma RVs via accept-reject