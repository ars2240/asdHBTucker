Simulink Project: asdHBTucker

Author: Adam Sandler
Date: 10/10/18

Instructions:

Tensor decompsition:
    Required data files: asdSparse.csv
    Required packages: Sandia NL Tensor Toolbox
    
    1. Compile appropriate MEX file (sample code provided but commented out in lines 5-6)
    Note: parallel verion requires OpenMP
    2. Adjust any settings in asdTens.m (see init_options.m for more info)
    Note: for pre-processing of genes, uncomment out (line 2):
        asd=asdGeneSelect(asdSparse, .1);
    and comment out (line 3):
        asd=sptensor(asdSparse(:,1:3),asdSparse(:,4));
    3. Run asdTens.m
    4. Output will be in asdHBTucker.mat

Classification (MatLab):
    Required data files: asdHBTucker*.mat (output of Tensor decomposition)
    
    1. Ensure logisticReg.m or logisticRegPCA.m loads the right .mat file
    2. If running ogisticRegPCA.m, adjust nPCs (number of principal components) in line 23
    3. Run logisticReg.m or logisticRegPCA.m
    4. Output will be in command line

Classification - no decomposition (MatLab):
    Required data files: asdSparseGenes.csv (sparse representation of patient and genetic variants counting tensor)
    
    1. Run logisticReg_noDecomp_genes.m
    2. Output will be in command line

Classification (Python):
    Required data files: asdHBTucker*.mat (output of Tensor decomposition)
    Required packages: matplotlib, numpy, scipy, sklearn, xgboost (for gbm.py only)
    
    1. Make create a /plot/ folder (if one does not exist)
    2. Ensure gbm.py, logistic_reg.py, logistic_feature_select.py, ran_forest.py, or svm.py loads the right .mat file
    3. Adjust any settings (number of features, regression factors, depth, and/or # of estimators)
    4. Run gbm.py, logistic_reg.py, logistic_feature_select.py, ran_forest.py, or svm.py
    5. Output will be in command line

New ASD Classification Method (Python):
    Required data files: asdHBTucker*.mat (output of Tensor decomposition)
    Required packages: matplotlib, numpy, scipy, sklearn

    1. Make create a /plot/ folder (if one does not exist)
    2. Compile data using proper *gatherCVData*.m script
    3. Ensure logistic_reg2.py or ran_forest2.py loads the right .mat file
    4. Adjust any settings (number of features, regression factors, depth, and/or # of estimators)
    5. Run logistic_reg2.py or ran_forest2.py
    6. Output will be in command line

New Cancer Classification Method (Python):
    Required data files: cancerHBTucker*.mat (output of Tensor decomposition)
    Required packages: matplotlib, numpy, scipy, sklearn

    1. Make create a /plot/ folder (if one does not exist)
    2. Compile data using proper *gatherCVData*.m script
    3. Ensure logistic_reg3.py loads the right .mat file
    4. Adjust any settings (number of features, regression factors, depth, and/or # of estimators)
    5. Run logistic_reg3.py
    6. Output will be in command line


Files:
- acc_cv.py- python script for CV accuracy (used for cancer dataset classification)
- asdGeneSelect.m- method for pre-selecting specific genes based on logistic regression
- asdGeneSelectCV.m- method for pre-selecting specific genes based on logistic regression, while cross-validating
- asdGeneSelectCV2.m- method for pre-selecting specific genes based on logistic regression, while cross-validating, using genetic variants only
- asdHBTucker3.m- hierarchical Bayesian Tucker decomposition function
- AsdHBTucker.prj- Simulink Project file
- asdTens.m- main run file
- asdTensCV.m- main run file for ASD dataset, separates decomposition into CV folds
- asdTensCVTest.m- main run file, computes groups for CV test folds
- cancerGatherCVData.m- gathers proper CV data from Bayesian tensor decomposition into a single .mat file
- cancerGatherCVData_noDecomp.m- gathers proper CV data for no decompositon into a single .mat file
- cancerGatherCVDataLDA.m- gathers proper CV data from LDA decomposition into a single .mat file
- cancerTensCV.m- main run file for cancer dataset, separates decomposition into CV folds
- cancerTensHLDA.m- main run file for cancer dataset for hLDA decompositons, separates decomposition into CV folds
- crp.m- draws new restaurant from Chinese Restaurant Process (CRP)
- createMRMRcsv.m- creates data csv for use in mRMR
- drawCoreCon.m- draws the core tensor for the conditional Dirichlet distribution
- drawCoreUni.m- draws the core tensor for the uniform Dirichlet distribution
- drawZ.m- draws topics for a specific sample
- drawZc.c- C version of drawZ function
- drawZsc.c- C version of drawZs function
- drawZsCollapsed.c- C version of drawZs function, collapsed sampling
- drawZsCollapsedPar.c- C version of drawZs function, collapsed sampling, OpenMP parallelization
- drawZscPar.c- C version of drawZs function, OpenMP parallelization
- drchrnd.m- generates probabilities from the Dirichlet distribution
- elems.m- returns all values between two vectors
- entropy.m- calculates entropy of probability vector
- gatherCVData.m- collects all data into one file for CV classification
- gbm.py- uses a gradient boosting model to learn & predict ASD
- gbm_mi.py- uses a gradient boosting model to learn & predict ASD, with MI feature selection
- init_options.m- option initialization
- initializePAM.m- initializes hierarchical DAG from the PAM
- initializeTree.m- initializes hierarchical tree from the CRP
- ldaTests.R- computes LDA decomposition baseline tests
- logistic_feature_select.py- predict using logistic regression with MI feature selection
- logistic_reg.py- predict using logistic regression with regularization
- logistic_reg2.py- predict using logistic regression with regularization, uses CV tensors
- logistic_reg3.py- predict using logistic regression with regularization, multi-class accuracy (for cancer dataset)
- logisticReg.m- uses a logistic regression model to learn & predict ASD
- logisticReg_augmentData.m- uses a logistic regression model to learn & predict ASD, augmented data set
- logisticReg_mRMR.r- uses a logistic regression model to learn & predict ASD, with mRMR feature selection
- logisticReg_noDecomp.m- uses a logistic regression model to learn & predict ASD, uses gene selection rather than a decomposition
- logisticReg_noDecomp_genes.m- uses a logistic regression model to learn & predict ASD, uses gene selection rather than a decomposition, using genetic variants only
- logisticRegDecompCV.m- uses a logistic regression model to learn & predict ASD, uses results from asdTensCV.m
- logisticRegPCA.m- predict using logistic regression, using first X PCs
- mRMR.r- selects features using mRMR method
- multi.m- draws a single sample from the multinomial distribution
- newPAM- draws PAM model for test documents
- newTreePaths.m- draws tree for test documents
- newTreePathsInit.m- draws tree for test documents
- nn.py- uses a neural network model to learn & predict ASD
- opt.m- separate file that computes tests for our optimization problem
- ran_forest.py- predict using random forest
- ran_forest2.py- predict using random forest, for CV datasets
- ran_forest_mi.py- predict using random forest, with MI feature selection
- redrawPAM.c- draws the DAG from the PAM
- redrawTree.c- draws the tree from the CRP
- rgamma.c- samples small-shape gamma RVs via accept-reject
- roc_cv.py- computes and plots ROC for each CV
- roc_cv2.py- computes and plots ROC for each CV, for CV datasets
- roc_cv_nn.py- computes and plots ROC for each CV, for nn.py
- sortTopics.m- re-orders topics
- svm.py- uses SVM to learn & predict ASD