# logistic_reg3.py
#
# Author: Adam Sandler
# Date: 1/17/19
#
# Computes logistic regression tests with regularization parameter
#
# Uses CV data
#
# Dependencies:
#   Packages: matplotlib, numpy, scipy, sklearn
#   Files: acc_cv_sep
#   Data: asdHBTucker

# load packages
from acc_cv_sep import acc
from sklearn.linear_model import LogisticRegression

fname = 'cancerHBTuckerCVND_L2_tpl0.1_{i}_Cartesian_Trees'
yfname = 'cancerHBTuckerCVDataLDA'
#yfname = None

# #############################################################################
# Classification and ROC analysis

print('%6s\t %6s\t %6s\t %6s\t %6s' % ('dset', 'lam', 'mean', 'stdev', 'pval'))

lam_v = [1e-6, 1e-3, 1, 1e3, 1e6]

for lam in lam_v:
    # Run classifier with cross-validation and plot ROC curves
    classifier = LogisticRegression(C=1 / lam, solver='liblinear', multi_class='ovr')

    mean_acc, std_acc, p_val, mean_acc_tr, std_acc_tr, p_val_tr = acc(classifier, fname, yfname)

    print('%6s\t %6.3e\t %0.4f\t %0.4f\t %0.4f' % ('valid', lam, mean_acc, std_acc, p_val))
    print('%6s\t %6.3e\t %0.4f\t %0.4f\t %0.4f' % ('train', lam, mean_acc_tr, std_acc_tr, p_val_tr))
