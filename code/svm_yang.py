# svm_yang.py
#
# Author: Adam Sandler
# Date: 1/14/20
#
# Computes svm tests with regularization parameter
#
# Uses CV data
#
# Dependencies:
#   Packages: matplotlib, numpy, pandas, scipy, sklearn
#   Files: acc_cv2
#   Data: asdHBTucker

# load packages
from acc_yang import acc
from sklearn.svm import SVC

fname = 'yangHBTuckerCV_L2_gam2_Level_PAM'
yfname = 'yangTucker'


# #############################################################################
# Classification and ROC analysis

print('%6s\t %6s\t %6s' % ('dset', 'C', 'acc'))

C_v = [1E-9, 1E-6, 1E-3, 1, 1E3, 1E6, 1E9]

for C in C_v:
    # Run classifier with cross-validation and plot ROC curves
    classifier = SVC(C=C, probability=True)

    acc_te, acc_tr = acc(classifier, fname, yfname, fselect='min')

    print('%6s\t %6.3e\t %0.4f' % ('valid', C, acc_te))
    print('%6s\t %6.3e\t %0.4f' % ('train', C, acc_tr))
