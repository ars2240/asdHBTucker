# ran_forest_yang.py
#
# Author: Adam Sandler
# Date: 1/14/20
#
# Computes random forest tests with regularization parameter
#
# Uses CV data
#
# Dependencies:
#   Packages: matplotlib, numpy, pandas, scipy, sklearn
#   Files: acc_cv2
#   Data: asdHBTucker

# load packages
from acc_yang import acc
from sklearn.ensemble import RandomForestClassifier

fname = 'yangHBTuckerCV_L2_gam2_Cartesian_PAM'
yfname = 'yangTucker'


# #############################################################################
# Classification and ROC analysis

print('%6s\t %6s\t %6s\t %6s' % ('dset', 'd', 'nEst', 'acc'))

d_v = [1, 2, 3, 4, 5]
N_v = [1, 10, 100, 1000]

for d in d_v:
    for N in N_v:
        # Run classifier with cross-validation and plot ROC curves
        classifier = RandomForestClassifier(max_depth=d, n_estimators=N)

        acc_te, acc_tr = acc(classifier, fname, yfname, fselect='min')

        print('%6s\t %6d\t %6d\t %0.4f' % ('valid', d, N, acc_te))
        print('%6s\t %6d\t %6d\t %0.4f' % ('train', d, N, acc_tr))
