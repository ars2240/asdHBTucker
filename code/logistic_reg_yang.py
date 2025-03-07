# logistic_reg_yang.py
#
# Author: Adam Sandler
# Date: 1/13/20
#
# Computes logistic regression tests with regularization parameter
#
# Uses CV data
#
# Dependencies:
#   Packages: matplotlib, numpy, pandas, scipy, sklearn
#   Files: acc_cv2
#   Data: asdHBTucker

# load packages
from acc_yang import acc
from sklearn.linear_model import LogisticRegression

fname = 'yangHBTuckerCV_L2_gam2_Cartesian_PAM'
yfname = 'yangTucker'


# #############################################################################
# Classification and ROC analysis

print('%6s\t %6s\t %6s' % ('dset', 'lam', 'acc'))

lam_v = [1E-9, 1E-6, 1E-3, 1, 1E3, 1E6, 1E9]

for lam in lam_v:
    # Run classifier with cross-validation and plot ROC curves
    classifier = LogisticRegression(C=1 / lam)

    acc_te, acc_tr = acc(classifier, fname, yfname, fselect='min')

    print('%6s\t %6.3e\t %0.4f' % ('valid', lam, acc_te))
    print('%6s\t %6.3e\t %0.4f' % ('train', lam, acc_tr))
