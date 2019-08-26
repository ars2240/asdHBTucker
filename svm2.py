# svm.py
#
# Author: Adam Sandler
# Date: 8/5/19
#
# Computes SVM tests with regularization parameter
#
# Uses CV data
#
# Dependencies:
#   Packages: matplotlib, numpy, scipy, sklearn
#   Files: acc_cv
#   Data: asdHBTucker

# load packages
from acc_cv import acc
import scipy.io
from sklearn.svm import SVC

fname = 'cancerHBTuckerGenData_noDecomp'
mdict = scipy.io.loadmat(fname)  # import dataset from matlab


# #############################################################################
# Classification and ROC analysis

print('%6s\t %6s\t %6s\t %6s\t %6s' % ('dset', 'lam', 'mean', 'stdev', 'pval'))

C_v = [1e-6, 1e-3, 1, 1e3, 1e6]

for C in C_v:
    # Run classifier with cross-validation and plot ROC curves
    classifier = SVC(C=C, kernel='linear', gamma='auto')

    mean_acc, std_acc, p_val, mean_acc_tr, std_acc_tr, p_val_tr = acc(classifier, mdict)

    print('%6s\t %6.3e\t %0.4f\t %0.4f\t %0.4f' % ('valid', C, mean_acc, std_acc, p_val))
    print('%6s\t %6.3e\t %0.4f\t %0.4f\t %0.4f' % ('train', C, mean_acc_tr, std_acc_tr, p_val_tr))
