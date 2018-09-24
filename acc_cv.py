# acc_cv.py
#
# Author: Adam Sandler
# Date: 9/20/18
#
# Computes accuracy for each CV, returns plot in /plots/ folder, and
# mean, stDev, and p-value for both train & validation sets
#
# Uses CV data
#
# Modified from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
#
# Dependencies:
#   Packages: matplotlib, numpy, scipy, sklearn
#   Data: asdHBTucker

import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc


def acc(classifier, mdict, splits=10, fselect='None', nfeat=100):

    acc = []
    acc_tr = []

    # load data
    phi = mdict.get('phi')
    testPhi = mdict.get('testPhi')
    asd = mdict.get('cvTrainASD')
    testASD = mdict.get('cvTestASD')

    i = 0
    for i in range(0, splits):

        X = phi[(i, 0)]
        s = X.shape
        if len(s) == 3:
            X = np.reshape(X, [s[0], s[1] * s[2]])
        else:
            X = np.reshape(X, [s[0], s[1]])
        y = asd[(i, 0)]
        y = np.reshape(y, s[0])
        X_test = testPhi[(i, 0)]
        s = X_test.shape
        if len(s) == 3:
            X_test = np.reshape(X_test, [s[0], s[1] * s[2]])
        else:
            X_test = np.reshape(X_test, [s[0], s[1]])
        y_test = testASD[(i, 0)]
        y_test = np.reshape(y_test, s[0])

        # reformat
        if fselect == 'MI':
            model = SelectKBest(mutual_info_classif, k=nfeat).fit(X, y)
            X = model.transform(X)
            X_test = model.transform(X_test)
        elif fselect == 'PCA':
            model = PCA(n_components=nfeat).fit(X)
            X = model.transform(X)
            X_test = model.transform(X_test)

        # fit model
        model = classifier.fit(X, y)

        # Compute accuracy for validation set
        probas_ = model.predict_proba(X_test)
        y_hat = np.argmax(probas_, axis=1)
        acc.append(sum(y_hat == y_test)/len(y_test))

        # Compute accuracy for training set
        probas_ = model.predict_proba(X)
        y_hat = np.argmax(probas_, axis=1)
        acc_tr.append(sum(y_hat == y) / len(y))

        i += 1

    results = stats.ttest_1samp(acc, popmean=0.25)
    p_val = results[1]

    results = stats.ttest_1samp(acc_tr, popmean=0.25)
    p_val_tr = results[1]

    return np.mean(acc), np.std(acc), p_val, np.mean(acc_tr), np.std(acc_tr), p_val_tr
