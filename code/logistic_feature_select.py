# logistic_feature_select.py
#
# Author: Adam Sandler
# Date: 6/7/18
#
# Computes logistic regression tests with feature selection
#
# Dependencies:
#   Packages: matplotlib, numpy, scipy, sklearn
#   Files: roc_cv
#   Data: asdHBTucker

# load packages
import numpy as np
from roc_cv import roc
import scipy.io
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
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

nfeat_v = [10, 25, 50, 75, 100, 200]

# #############################################################################
# Classification and ROC analysis

print('%6s\t %6s\t %6s\t %6s\t %6s' % ('dset', 'nfeat', 'mean', 'stdev', 'pval'))

for nfeat in nfeat_v:
    X_new = SelectKBest(mutual_info_classif, k=nfeat).fit_transform(X, y)

    # Run classifier with cross-validation and plot ROC curves
    classifier = LogisticRegression(C=1e15)

    pname = 'logreg_AUC_feat_select_' + str(nfeat) + '_' + str(fname)
    mean_auc, std_auc, p_val, mean_auc_tr, std_auc_tr, p_val_tr = roc(classifier, X, y, pname)

    print('%6s\t %6d\t %0.4f\t %0.4f\t %0.4f' % ('valid', nfeat, mean_auc, std_auc, p_val))
    print('%6s\t %6d\t %0.4f\t %0.4f\t %0.4f' % ('train', nfeat, mean_auc_tr, std_auc_tr, p_val_tr))
