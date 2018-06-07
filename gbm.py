# gbm.py
#
# Author: Adam Sandler
# Date: 6/7/18
#
# Computes Gradient Boosting classifier
#
# Dependencies:
#   Packages: matplotlib, numpy, scipy, sklearn, xgboost
#   Files: roc_cv
#   Data: asdHBTucker

# load packages
import numpy as np
from roc_cv import roc
import scipy.io
from xgboost import XGBClassifier
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

print('%6s\t %6s\t %6s\t %6s\t %6s\t %6s' % ('dset', 'N', 'depth', 'mean', 'stdev', 'pval'))

N_v = [100]
d_v = [3]

for N in N_v:
    for d in d_v:
        # Run classifier with cross-validation and plot ROC curves
        classifier = XGBClassifier(max_depth=d, n_estimators=N)

        pname = 'gbm_AUC_' + str(N) + '_' + str(d) + '_' + str(fname)
        mean_auc, std_auc, p_val, mean_auc_tr, std_auc_tr, p_val_tr = roc(classifier, X, y, pname)

        print('%6s\t %6d\t %6d\t %0.4f\t %0.4f\t %0.4f' % ('valid', N, d, mean_auc, std_auc, p_val))
        print('%6s\t %6d\t %6d\t %0.4f\t %0.4f\t %0.4f' % ('train', N, d, mean_auc_tr, std_auc_tr, p_val_tr))