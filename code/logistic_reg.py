# logistic_reg.py
#
# Author: Adam Sandler
# Date: 6/7/18
#
# Computes logistic regression tests with regularization parameter
#
# Dependencies:
#   Packages: matplotlib, numpy, scipy, sklearn
#   Files: roc_cv
#   Data: asdHBTucker

# load packages
import numpy as np
from roc_cv import roc
import scipy.io
from sklearn.linear_model import LogisticRegression
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

lam_v = [1E-15, 1E-9, 1E-6, 1E-3, 1, 1E3, 1E6, 1E9]

# #############################################################################
# Classification and ROC analysis

print('%6s\t %9s\t %6s\t %6s\t %6s' % ('dset', 'lambda', 'mean', 'stdev', 'pval'))

# Run classifier with cross-validation and plot ROC curves
for lam in lam_v:
    classifier = LogisticRegression(C=1 / lam)

    pname = 'logreg_AUC_' + str(lam) + '_' + str(fname)
    mean_auc, std_auc, p_val, mean_auc_tr, std_auc_tr, p_val_tr = roc(classifier, X, y, pname)

    print('%6s\t %6.3e\t %0.4f\t %0.4f\t %0.4f' % ('valid', lam, mean_auc, std_auc, p_val))
    print('%6s\t %6.3e\t %0.4f\t %0.4f\t %0.4f' % ('train', lam, mean_auc_tr, std_auc_tr, p_val_tr))
