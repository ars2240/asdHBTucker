# logistic_reg4.py
#
# Author: Adam Sandler
# Date: 10/18/18
#
# Computes logistic regression tests with regularization parameter
#   - uses csv from no-decomposition
#
# Uses CV data
#
# Dependencies:
#   Packages: matplotlib, numpy, pandas, scipy, sklearn
#   Files: acc_cv2
#   Data: asdHBTucker

# load packages
from acc_cv2 import acc
from sklearn.linear_model import LogisticRegression

fname = 'cancerNumber'
indF = 'cancerCVInd'
labelF = 'cancerLabel'


# #############################################################################
# Classification and ROC analysis

print('%6s\t %6s\t %6s\t %6s\t %6s' % ('dset', 'lam', 'mean', 'stdev', 'pval'))

lam_v = [1E-15, 1E-9, 1E-6, 1E-3, 1, 1E3, 1E6, 1E9]

for lam in lam_v:
    # Run classifier with cross-validation and plot ROC curves
    classifier = LogisticRegression(C=1 / lam)

    mean_acc, std_acc, p_val, mean_acc_tr, std_acc_tr, p_val_tr = acc(classifier, fname, labelF, indF, fselect='min')

    print('%6s\t %6.3e\t %0.4f\t %0.4f\t %0.4f' % ('valid', lam, mean_acc, std_acc, p_val))
    print('%6s\t %6.3e\t %0.4f\t %0.4f\t %0.4f' % ('train', lam, mean_acc_tr, std_acc_tr, p_val_tr))
