Simulink Project: asdHBTucker

Author: Adam Sandler
Date: 10/10/18

Instructions:

Tensor decompsition:
    Required data files: asdSparse.csv or cancerSparse.csv
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
    Required data files: asdHBTucker*.mat or cancerHBTucker*.mat (output of Tensor decomposition)
    
    1. Ensure logisticReg.m or logisticRegPCA.m loads the right .mat file
    2. If running logisticRegPCA.m, adjust nPCs (number of principal components) in line 23
    3. Run logisticReg.m or logisticRegPCA.m
    4. Output will be in command line

Classification - no decomposition (MatLab):
    Required data files: asdSparseGenes.csv or cancerSparseGenes.csv (sparse representation of patient and genetic variants counting tensor)
    
    1. Run logisticReg_noDecomp_genes.m
    2. Output will be in command line

Classification (Python):
    Required data files: asdHBTucker*.mat or cancerHBTucker*.mat (output of Tensor decomposition)
    Required packages: matplotlib, numpy, scipy, sklearn, xgboost (for gbm.py only)
    
    1. Make create a /plot/ folder (if one does not exist)
    2. Ensure gbm.py, logistic_reg.py, logistic_feature_select.py, ran_forest.py, or svm.py loads the right .mat file
    3. Adjust any settings (number of features, regression factors, depth, and/or # of estimators)
    4. Run gbm.py, logistic_reg.py, logistic_feature_select.py, ran_forest.py, or svm.py
    5. Output will be in command line

New ASD Classification Method (Python):
    Required data files: asdHBTucker*.mat or cancerHBTucker*.mat (output of Tensor decomposition)
    Required packages: matplotlib, numpy, scipy, sklearn

    1. Make create a /plot/ folder (if one does not exist)
    2. Compile data using proper *gatherCVData*.m script
    3. Ensure logistic_reg2.py or ran_forest2.py loads the right .mat file
    4. Adjust any settings (number of features, regression factors, depth, and/or # of estimators)
    5. Run logistic_reg2.py or ran_forest2.py
    6. Output will be in command line

New Cancer Classification Method (Python):
    Required data files: cancerHBTucker*.mat or cancerHBTucker*.mat (output of Tensor decomposition)
    Required packages: matplotlib, numpy, scipy, sklearn

    1. Make create a /plot/ folder (if one does not exist)
    2. Compile data using proper *gatherCVData*.m script
    3. Ensure logistic_reg3.py loads the right .mat file
    4. Adjust any settings (number of features, regression factors, depth, and/or # of estimators)
    5. Run logistic_reg3.py
    6. Output will be in command line


Files:
- acc_cv.py- python script for CV accuracy (used for cancer dataset classification)
- acc_cv2.py- 
- acc_cv_sep.py- 
- acc_cv_w_gen.py-
- acc_yang.py-
- asdGeneSelect.m- method for pre-selecting specific genes based on logistic regression
- asdGeneSelectCV.m- method for pre-selecting specific genes based on logistic regression, while cross-validating
- asdGeneSelectCV2.m- method for pre-selecting specific genes based on logistic regression, while cross-validating, using genetic variants only
- asdHBTucker3.m- hierarchical Bayesian Tucker decomposition function
- asdHBTuckerNew.m- 
- AsdHBTucker.prj- Simulink Project file
- asdTens.m- main run file
- asdTensCV.m- main run file for ASD dataset, separates decomposition into CV folds
- asdTensCVTest.m- main run file, computes groups for CV test folds
- cancerCVLL.m- 
- cancerGatherCVData.m- gathers proper CV data from Bayesian tensor decomposition into a single .mat file
- cancerGatherCVDataAugm_LDANoDecomp.m- 
- cancerGatherCVDataAugm_LDANoDecompRmTopGVs.m- 
- cancerGatherCVDataGen.m- 
- cancerGatherCVDataLDA.m- gathers proper CV data from LDA decomposition into a single .mat file
- cancerGatherCVDataLDAGen.m- 
- cancerGatherCVDataLDA_cheating.m- 
- cancerGatherCVDataLDA_rmDomTop.m- 
- cancerGatherCVData_noDecomp.m- gathers proper CV data for no decompositon into a single .mat file
- cancerGatherGenData_noDecomp.m- 
- cancerGenData.m- 
- cancerGenDataLoad.m- 
- cancerGenLabs.m- 
- cancerInd.m- 
- cancerTensCV.m- main run file for cancer dataset, separates decomposition into CV folds
- cancerTensCVGen.m- 
- cancerTensCVGen2.m- 
- cancerTensHLDA.m- main run file for cancer dataset for hLDA decompositons, separates decomposition into CV folds
- computeLL.m- 
- counts.m- 
- createMRMRcsv.m- creates data csv for use in mRMR
- crp.m- draws new restaurant from Chinese Restaurant Process (CRP)
- drawCoreCon.m- draws the core tensor for the conditional Dirichlet distribution
- drawCoreUni.m- draws the core tensor for the uniform Dirichlet distribution
- drawZ.m- draws topics for a specific sample
- drawZc.c- C version of drawZ function
- drawZsCollapsed.c- C version of drawZs function, collapsed sampling
- drawZsCollapsedPar.c- C version of drawZs function, collapsed sampling, OpenMP parallelization
- drawZsc.c- C version of drawZs function
- drawZscPar.c- C version of drawZs function, OpenMP parallelization
- drawZscSparse.c- 
- drawZscSparsePar.c- 
- drchrnd.m- generates probabilities from the Dirichlet distribution
- elems.m- returns all values between two vectors
- entropy.m- calculates entropy of probability vector
- gatherCVData.m- collects all data into one file for CV classification
- gbm.py- uses a gradient boosting model to learn & predict ASD
- gbm_mi.py- uses a gradient boosting model to learn & predict ASD, with MI feature selection
- genLDA.py- 
- generatePatients.m- 
- generatedDataClass.py- 
- generatedDataLDA.py- 
- initPAM.m- 
- init_options.m- option initialization
- initializePAM.m- initializes hierarchical DAG from the PAM
- initializeTree.m- initializes hierarchical tree from the CRP
- lda.py- 
- ldaCoherence.py- 
- ldaCust.py- 
- ldaGV.py- 
- ldaGVcancer.py- 
- ldaParseLog.py- 
- ldaTests.R- computes LDA decomposition baseline tests
- logLikelihood.m- 
- logisticReg.m- uses a logistic regression model to learn & predict ASD
- logisticRegDecompCV.m- 
- logisticRegPCA.m- 
- logisticReg_augmentData.m- uses a logistic regression model to learn & predict ASD, augmented data set
- logisticReg_mRMR.r- uses a logistic regression model to learn & predict ASD, with mRMR feature selection
- logisticReg_noDecomp.m- uses a logistic regression model to learn & predict ASD, uses gene selection rather than a decomposition
- logisticReg_noDecomp_genes.m- uses a logistic regression model to learn & predict ASD, uses gene selection rather than a decomposition, using genetic variants only
- logisticRegDecompCV.m- uses a logistic regression model to learn & predict ASD, uses results from asdTensCV.m
- logisticRegPCA.m- predict using logistic regression, using first X PCs
- logistic_feature_select.py- predict using logistic regression with MI feature selection
- logistic_reg.py- predict using logistic regression with regularization
- logistic_reg2.py- predict using logistic regression with regularization, uses CV tensors
- logistic_reg3.py- predict using logistic regression with regularization, multi-class accuracy (for cancer dataset)
- logistic_reg4.py- 
- logistic_reg_yang.py- 
- mRMR.r- selects features using mRMR method
- multi.m- draws a single sample from the multinomial distribution
- newPAM- draws PAM model for test documents
- newTreePaths.m- draws tree for test documents
- newTreePathsInit.m- draws tree for test documents
- nn.py- uses a neural network model to learn & predict ASD
- opt.m- separate file that computes tests for our optimization problem
- psiCompMH.m- 
- psiMH.m- 
- ran_forest.py- predict using random forest
- ran_forest2.py- predict using random forest, for CV datasets
- ran_forest_mi.py- predict using random forest, with MI feature selection
- ran_forest_yang.py- 
- redrawPAM.c- draws the DAG from the PAM
- redrawTree.c- draws the tree from the CRP
- rgamma.c- samples small-shape gamma RVs via accept-reject
- roc_cv.py- computes and plots ROC for each CV
- roc_cv2.py- computes and plots ROC for each CV, for CV datasets
- roc_cv_nn.py- computes and plots ROC for each CV, for nn.py
- sortTopics.m- re-orders topics
- spliceFormatData.m- 
- spliceFormatDataRand.m- 
- spliceTensCV.m- 
- svm.py- uses SVM to learn & predict ASD
- svm2.py- 
- svm_yang.py- 
- tenDec.py- 
- tensIndex.m- 
- tensIndex2.m- 
- tuckerCompMH.m- 
- tuckerCompPCA.m- 
- tuckerMH.m- 
- yangAcc.m- 
- yangTest.m- 
