Simulink Project: asdHBTucker

Author: Adam Sandler
Date: 5/31/18

Instructions:

Tensor decompsition:
    Required data files: asdSparse.csv
    Required packages: Sandia NL Tensor Toolbox
    
    1. Compile appropriate MEX file (sample code provided but commented out in lines 5-6)
    Note: parallel verion requires OpenMP
    2. Adjust any settings in asdTens.m (see init_options.m for more info)
    Note: for pre-processing of genes, uncomment-out (line 2):
        asd=asdGeneSelect(asdSparse, .1);
    and comment out (line 3):
        asd=sptensor(asdSparse(:,1:3),asdSparse(:,4));
    3. Run asdTens.m
    4. Output will be in asdHBTucker.mat

Classification (MatLab):
    Required data files: asdHBTucker*.mat (output of Tensor decomposition)
    
    1. Ensure logisticReg.m or logisticRegPCA.m loads the right .mat file
    2. If, running ogisticRegPCA.m, adjust nPCs (number of principal components) in line 23
    2. Run logisticReg.m or logisticRegPCA.m
    3. Output will be in command line

Classification (Python):
    Required data files: asdHBTucker*.mat (output of Tensor decomposition)
    Required packages: matplotlib, numpy, scipy, sklearn
    
    1. Ensure gbm.py, logistic_reg.py, logistic_feature_select.py, or svm.py loads the right .mat file
    2. Adjust any settings (number of features, regression factors, depth, and/or # of estimators
    2. Run gbm.py, logistic_reg.py, logistic_feature_select.py, or svm.py
    3. Output will be in command line


Files:
- asdGeneSelect.m- method for pre-selecting specific genes based on logistic regression
- AsdHBTucker.prj- Simulink Project file
- asdHBTucker3.m- hierarchical Bayesian Tucker decomposition function
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
- gbm.py- uses a gradient boosting model to learn & predict ASD
- init_options.m- option initialization
- initializeTree.m- initializes hierarchical tree from the CRP
- initializePAM.m- initializes hierarchical DAG from the PAM
- ldaTests.R- computes LDA decomposition baseline tests
- logistic_reg.py- predict using logistic regression with regularization
- logistic_feature_select.py- predict using logistic regression with MI feature selection
- logisticReg.m- uses a logistic regression model to learn & predict ASD
- logisticRegPCA.m- predict using logistic regression, using first X PCs
- multi.m- draws a single sample from the multinomial distribution
- opt.m- separate file that computes tests for our optimization problem
- redrawTree.c- draws the tree from the CRP
- redrawPAM.c- draws the DAG from the PAM
- rgamma.c- samples small-shape gamma RVs via accept-reject
- svm.py- uses SVM to learn & predict ASD