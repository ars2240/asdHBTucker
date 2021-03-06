# svm.py
#
# Author: Adam Sandler
# Date: 6/7/18
#
# Computes svm
#
# Dependencies:
#   Packages: matplotlib, numpy, scipy, sklearn
#   Files: roc_cv
#   Data: asdHBTucker

# load packages
import numpy as np
from roc_cv import roc
import scipy.io
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

fname = 'asdHBTucker_gam0.1'
mdict = scipy.io.loadmat(fname)  # import dataset from matlab

# reformat data
phi = mdict.get('phi')
s = phi.shape
X = np.reshape(phi, [s[0], s[1]*s[2]])
y = np.empty((s[0],))
y[::2] = 1
y[1::2] = 0

# split set into training & test sets
X, X_test, y, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

# #############################################################################
# Classification and ROC analysis

print('%6s\t %9s\t %6s\t %6s\t %6s' % ('dset', 'C', 'mean', 'stdev', 'pval'))

C_v = [10]

for C in C_v:
    # Run classifier with cross-validation and plot ROC curves
    classifier = SVC(C=C, probability=True)

    pname = 'svm_AUC_' + str(C) + '_' + str(fname)
    mean_auc, std_auc, p_val, mean_auc_tr, std_auc_tr, p_val_tr = roc(classifier, X, y, pname)

    print('%6s\t %6.3e\t %0.4f\t %0.4f\t %0.4f' % ('valid', C, mean_auc, std_auc, p_val))
    print('%6s\t %6.3e\t %0.4f\t %0.4f\t %0.4f' % ('train', C, mean_auc_tr, std_auc_tr, p_val_tr))
