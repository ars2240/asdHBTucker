# acc_cv_w_gen.py
#
# Author: Adam Sandler
# Date: 8/1/19
#
# Computes accuracy for each CV, returns plot in /plots/ folder, and
# mean, stDev, and p-value for both train & validation sets
#
# Uses CV data
#
# Dependencies:
#   Packages: matplotlib, numpy, scipy, sklearn
#   Data: asdHBTucker

import numpy as np
import random
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA


def acc(classifier, mdict, splits=10, fselect='', nfeat=100, fmin=0, fmax=1000, a=.05, thresh=0):

    acc = []
    acc_tr = []

    # load data
    phi = mdict.get('phi')
    testPhi = mdict.get('testPhi')

    for i in range(0, splits):

        X = phi[(i, 0)]
        s = X.shape
        if len(s) == 3:
            X = np.reshape(X, [s[0], s[1] * s[2]])
        else:
            X = np.reshape(X, [s[0], s[1]])
        # random data
        rows = random.sample(range(0, s[0]), 4)
        rcts = X[rows, ]
        rlabs = range(0, 4)
        classifier.fit(rcts, rlabs)
        y = classifier.predict(X)
        y = np.reshape(y, s[0])
        X_test = testPhi[(i, 0)]
        s = X_test.shape
        if len(s) == 3:
            X_test = np.reshape(X_test, [s[0], s[1] * s[2]])
        else:
            X_test = np.reshape(X_test, [s[0], s[1]])
        y_test = classifier.predict(X_test)
        y_test = np.reshape(y_test, s[0])

        # subset features
        if 'min' in fselect:
            cols = X.astype(bool).sum(axis=0) > fmin
            X = X[:, cols]
            X_test = X_test[:, cols]
        if 'max' in fselect:
            cols = X.astype(bool).sum(axis=0) < fmax
            X = X[:, cols]
            X_test = X_test[:, cols]
        if 'thresh' in fselect:
            X[X < thresh] = 0
            X_test[X_test < thresh] = 0

        if 'MI' in fselect:
            model = SelectKBest(mutual_info_classif, k=nfeat).fit(X, y)
            X = model.transform(X)
            X_test = model.transform(X_test)
        elif 'PCA'in fselect:
            model = PCA(n_components=nfeat).fit(X)
            X = model.transform(X)
            X_test = model.transform(X_test)
        elif 'reg' in fselect:
            model = SelectFpr(f_classif, alpha=a).fit(X, y)
            X = model.transform(X)
            X_test = model.transform(X_test)
        elif 'kbest' in fselect:
            model = SelectKBest(f_classif, k=nfeat).fit(X, y)
            X = model.transform(X)
            X_test = model.transform(X_test)

        # fit model
        model = classifier.fit(X, y)

        # Compute accuracy for validation set
        y_hat = model.predict(X_test)
        acc.append(sum(y_hat == y_test)/len(y_test))

        # Compute accuracy for training set
        y_hat = model.predict(X)
        acc_tr.append(sum(y_hat == y) / len(y))

        i += 1

    results = stats.ttest_1samp(acc, popmean=755/2126)
    p_val = results[1]

    results = stats.ttest_1samp(acc_tr, popmean=755/2126)
    p_val_tr = results[1]

    return np.mean(acc), np.std(acc), p_val, np.mean(acc_tr), np.std(acc_tr), p_val_tr
