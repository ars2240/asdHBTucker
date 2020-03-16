Simulink Project: asdHBTucker

Author: Adam Sandler
Date: 3/16/20

Instructions:

Tensor decompsition (without CV):

    Required data files: asdSparse.csv or cancerSparse.csv
    Required packages: Sandia NL Tensor Toolbox
    
    1. Compile appropriate MEX file (sample code provided but commented out in lines 5-6)
    Note: parallel verion requires OpenMP
    2. Adjust any settings in asdTensCV.m (see init_options.m for more info)
    Note: for pre-processing of genes, uncomment out (line 2):
        asd=asdGeneSelect(asdSparse, .1);
    and comment out (line 3):
        asd=sptensor(asdSparse(:,1:3),asdSparse(:,4));
    3. Run asdTens.m
    4. Output will be in asdHBTucker.mat

Tensor decompsition (with CV):

    Required data files: asdSparse.csv or cancerSparse.csv
    Required packages: Sandia NL Tensor Toolbox
    
    Note: codeDiagram.jpg may be a helpful reference
    
    1. Create a /data/ folder (if one does not exist)
    2. Compile appropriate MEX files
    Note: parallel verion requires OpenMP
    3. Adjust any settings in cancerTensCV.m (see init_options.m for more info)
    4. Adjust saved data file name(s) as appropriate
    5. Run cancerTensCV.m
    6. Output will be in /data/ folder
    
Tensor decompsition (Yang model):

    Required data files: asdSparse.csv or cancerSparse.csv
    
    1. Create a /data/ folder (if one does not exist)
    2. Some settings can be adjusted in the yang.m function
    3. Adjust saved data file name(s) as appropriate
    4. Run cancerYangCV.m
    5. Output will be in /data/ folder

Old Classification (Python):

    Required data files: asdHBTucker*.mat or cancerHBTucker*.mat (output of Tensor decomposition)
    Required packages: matplotlib, numpy, scipy, sklearn, xgboost (for gbm.py only)
    
    1. Create a /plot/ folder (if one does not exist)
    2. Ensure gbm.py, logistic_reg.py, logistic_feature_select.py, ran_forest.py, or svm.py loads the right .mat file
    3. Adjust any settings (number of features, regression factors, depth, and/or # of estimators)
    4. Run gbm.py, logistic_reg.py, logistic_feature_select.py, ran_forest.py, or svm.py
    5. Output will be in command line

ASD Classification Method (Python):

    Required data files: asdHBTucker*.mat or cancerHBTucker*.mat (output of Tensor decomposition)
    Required packages: matplotlib, numpy, scipy, sklearn

    1. Create a /plot/ folder (if one does not exist)
    2. Compile data using proper *gatherCVData*.m script
    3. Ensure logistic_reg2.py or ran_forest2.py loads the right .mat file
    4. Adjust any settings (number of features, regression factors, depth, and/or # of estimators)
    5. Run logistic_reg2.py or ran_forest2.py
    6. Output will be in command line

Cancer Classification Method (Python):

    Required data files: cancerHBTucker*.mat or cancerHBTucker*.mat (output of Tensor decomposition)
    Required packages: matplotlib, numpy, scipy, sklearn

    1. Create a /plot/ folder (if one does not exist)
    2. Compile data using proper *gatherCVData*.m script
    3. Ensure logistic_reg*.py (* other than above) loads the right .mat file
    4. Adjust any settings (number of features, regression factors, depth, and/or # of estimators)
    5. Run logistic_reg*.py
    6. Output will be in command line

Yang Model:
    From: "Bayesian Conditional Tensor Factorizations for High-Dimensional Classification" by Yun Yang & David Dunson

Files:
- acc_cv.py- CV classification accuracy (for cancer dataset)
- acc_cv2.py- CV classification accuracy - uses csv from no-decomposition model
- acc_cv_sep.py- CV classification accuracy - useful for if each CV fold is in a different file
- acc_cv_w_gen.py- CV classification accuracy with generated classes
- acc_yang.py- CV classification accuracy for Yang generated dataset
- asdGeneSelect.m- method for pre-selecting specific genes based on logistic regression
- asdGeneSelectCV.m- method for pre-selecting specific genes based on logistic regression, while cross-validating
- asdGeneSelectCV2.m- method for pre-selecting specific genes based on logistic regression, while cross-validating, using genetic variants only
- asdHBTucker3.m- hierarchical Bayesian Tucker (HBT) decomposition function
- asdHBTuckerNew.m- HBT decomposition function (for new/test/validation data)
- AsdHBTucker.prj- Simulink Project file
- asdTens.m- main run file
- asdTensCV.m- main run file for ASD dataset, separates decomposition into CV folds
- asdTensCVTest.m- main run file, computes groups for CV test folds
- cancerCP.m- computes CP decomposition for cancer data (using Tensor Toolbox ALS method)
- cancerCVLL.m- computes log-likelihood (LL) for existing cancer HBT decomposition data
- cancerGatherCVData.m- gathers proper CV data from HBT decomposition into a single .mat file
- cancerGatherCVDataAugm_LDANoDecomp.m- gathers proper augmented CV data from LDA & no decomposition into a single .mat file
- cancerGatherCVDataAugm_LDANoDecompRmTopGVs.m- gathers proper augmented CV data from LDA & no decomposition into a single .mat file (and removes top genetic variants in each topic)
- cancerGatherCVDataGen.m- gathers proper augmented CV data from HBT decomposition into a single .mat file (for generated data)
- cancerGatherCVDataLDA.m- gathers proper CV data from LDA decomposition into a single .mat file
- cancerGatherCVDataLDAGen.m- gathers proper CV data from LDA decomposition into a single .mat file (for generated data)
- cancerGatherCVDataLDA_rmDomTop.m- gathers proper CV data from LDA decomposition into a single .mat file (and removes top genetic variants in each topic)
- cancerGatherCVData_noDecomp.m- gathers proper CV data for no decompositon into a single .mat file
- cancerGatherGenData_noDecomp.m- gathers proper CV data for no decompositon into a single .mat file (for generated data)
- cancerGenData.m- generates fictitious patients using the HBT generative process (from original ASD/cancer data)
- cancerGenDataLoad.m- generates fictitious patients using the HBT generative process (from an existing decomposition)
- cancerGenDataLoadSumPwy.m- generates fictitious patients using the HBT generative process (from an existing decomposition)
- cancerGenNumInd.m- creates CV indices for generated cancer dataset
- cancerInd.m- creates CV indices for cancer dataset
- cancerLLGraph.R- generates LL graph for cancer models
- cancerTensCV.m- main run file for cancer dataset, separates decomposition into CV folds
- cancerTensCVGen.m- HBT decomposition function (for generated cancer data)
- cancerTensGen.m- HBT decomposition function (for generated cancer data, without CV)
- cancerTensHLDA.m- main run file for cancer dataset for hLDA decompositons, separates decomposition into CV folds
- cancerYangCV.m- computes Yang model for cancer dataset
- computeLL.m- computes LL for existing cancer HBT decomposition data (from single .mat file)
- counts.m- computes counts of samples in tree and for the decomposition tensor (phi & psi)
- createMRMRcsv.m- creates data csv for use in mRMR
- crp.m- draws new restaurant from Chinese Restaurant Process (CRP)
- drawCoreCon.m- draws the core tensor for the conditional Dirichlet distribution
- drawCoreUni.m- draws the core tensor for the uniform Dirichlet distribution
- drawZ.m- draws topics for a specific sample
- drawZc.c- C version of drawZ function
- drawZsCollapsed.c- C version of drawZs function, collapsed sampling
- drawZsCollapsedPar.c- C version of drawZs function, collapsed sampling, with OpenMP parallelization
- drawZsc.c- C version of drawZs function
- drawZscPar.c- C version of drawZs function, with OpenMP parallelization
- drawZscSparse.c- draws topics for a specific sample (using sparse tensor representation)
- drawZscSparsePar.c- draws topics for a specific sample, with OpenMP parallelization (using sparse tensor representation)
- drchrnd.m- generates probabilities from the Dirichlet distribution
- elems.m- returns all values between two vectors
- entropy.m- calculates entropy of probability vector
- gatherCVData.m- collects all data into one file for CV classification
- gbm.py- uses a gradient boosting model to learn & predict ASD
- gbm_mi.py- uses a gradient boosting model to learn & predict ASD, with MI feature selection
- genLDA.py- generates fictitious patients using LDA
- generatePatients.m- generates new patients from trained HBT model
- generatedDataClass.py- creates classes for generated data
- generatedDataLDA.py- computes LDA decomposition (on generated data)
- initPAM.m- initializes variales for initializePAM.m
- init_options.m- option initialization
- initializePAM.m- initializes hierarchical DAG from the PAM
- initializeTree.m- initializes hierarchical tree from the CRP
- lda.py- class for LDA model (modified from Gensim)
- ldaCoherence.py- computes LDA model coherence
- ldaCust.py- trains LDA models, using CV (uses modified LDA code)
- ldaGV.py- trains LDA models, using CV
- ldaGVcancer.py- main file for doing LDA decomposition on cancer dataset
- ldaMI.py- computes mutual information for LDA topics
- ldaParseLog.py- parses LDA log files to get LL and perplexity values, plot LL over iterations
- ldaTests.R- computes LDA decomposition baseline tests
- logLikelihood.m- computes the LL of a HBT model
- logisticReg_mRMR.r- uses a logistic regression model to learn & predict ASD, with mRMR feature selection
- logistic_feature_select.py- predict using logistic regression with MI feature selection
- logistic_reg.py- predict using logistic regression with regularization
- logistic_reg2.py- predict using logistic regression with regularization, uses CV tensors
- logistic_reg3.py- predict using logistic regression with regularization, multi-class accuracy (for cancer dataset)
- logistic_reg4.py- predict using logistic regression with regularization, multi-class accuracy (for raw CSV data)
- logistic_reg_yang.py- predict using logistic regression with regularization, multi-class accuracy (for Yang model)
- logml.m- modified function, used in Yang model
- logml2.m- function, used in Yang model
- mRMR.r- selects features using mRMR method
- multi.m- draws a single sample from the multinomial distribution
- newPAM- draws PAM model for test documents
- newTreePaths.m- draws tree for test documents
- newTreePathsInit.m- draws tree for test documents
- nn.py- uses a neural network model to learn & predict ASD
- opt.m- separate file that computes tests for our optimization problem
- psiCompMH.m- compares psi from original HBT decomposition and that on generated data, using Metropolis-Hastings (MH) algorithm
- psiMH.m- compare 2 psi matrices, using MH
- ran_forest.py- predict using random forest
- ran_forest2.py- predict using random forest (for CV datasets)
- ran_forest_mi.py- predict using random forest, with MI feature selection
- ran_forest_yang.py- predict using random forest (for Yang model)
- redrawPAM.c- draws the DAG from the PAM
- redrawTree.c- draws the tree from the CRP
- rgamma.c- samples small-shape gamma RVs via accept-reject
- roc_cv.py- computes and plots ROC for each CV
- roc_cv2.py- computes and plots ROC for each CV, for CV datasets
- roc_cv_nn.py- computes and plots ROC for each CV, for nn.py
- sortTopics.m- re-orders topics
- spliceFormatData.m- reformating gene splice data to include both genes for those unknown
- spliceFormatDataRand.m- reformating gene splice data to include random gene, from options, for those unknown
- spliceTensCV.m- HBT decomposition function (for splice data)
- stirling.m- unction, used in Yang model
- svm.py- computes SVM tests with regularization parameter
- svm2.py- computes SVM tests with regularization parameter (cleaner version)
- svm_yang.py- computes SVM tests with regularization parameter (for Yang model)
- tenDec.py- computes CP decomposition for cancer data (using tensorly package)
- tensIndex.m- transform cell with lists of indices for each dimension into a matrix of every combination of indices
- tensIndex2.m- transforms indices from multivariate matrix to single linear index
- topicCount.m- computes number of topics in each mode for HBT decomposition
- tuckerCompMH.m- compares two Tucker decompositions using MH
- tuckerCompPCA.m- compares two Tucker decompositions, using PCA to equate the number of topics
- tuckerMH.m- compares phi from two Tucker decompositions, using MH
- yang.m- modified function, computes Yang model decomposition
- yangAcc.m- computes accuracy of trained model on Yang generated data
- yangTest.m- computes HBT decomposition on Yang generated data
