# acc_cv2.py
#
# Author: Adam Sandler
# Date: 10/18/18
#
# Computes accuracy for each CV, returns plot in /plots/ folder, and
# mean, stDev, and p-value for both train & validation sets
#   - uses csv from no-decomposition
#
# Uses CV data
#
# Dependencies:
#   Packages: matplotlib, numpy, pandas, scipy, sklearn
#   Data: asdHBTucker

import numpy as np
from scipy import stats
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from hmap import heatmap


def acc(classifier, fname, labelF, indF, splits=10, fselect='None', nfeat=100, featmin=3, a=.05, header=True, hmap=False):

    acc = []
    acc_tr = []

    # load data
    if header:
        cts = pd.read_csv(fname + '.csv', header=0, index_col=0, dtype={0: str})
    else:
        cts = pd.read_csv(fname + '.csv', header=None, index_col=None)
    ind = pd.read_csv(indF + '.csv', header=None)
    label = pd.read_csv(labelF + '.csv', header=0, index_col=0)
    rows = np.where(ind > 0)[0]
    if cts.shape[0] == len(rows):
        phi = cts
    else:
        phi = cts.iloc[rows]
    cancer = label.iloc[rows, 0]
    ind = ind.iloc[rows, 0]

    i = 0
    for i in range(0, splits):

        rows = np.where(ind != i + 1)
        X = phi.iloc[rows]
        s = X.shape
        if len(s) == 3:
            X = np.reshape(X, [s[0], s[1] * s[2]])
        else:
            X = np.reshape(X, [s[0], s[1]])
        y = cancer.iloc[rows]
        y = np.reshape(y, s[0])
        rows = np.where(ind == i + 1)
        X_test = phi.iloc[rows]
        s = X_test.shape
        if len(s) == 3:
            X_test = np.reshape(X_test, [s[0], s[1] * s[2]])
        else:
            X_test = np.reshape(X_test, [s[0], s[1]])
        y_test = cancer.iloc[rows]
        y_test = np.reshape(y_test, s[0])

        # subset features
        if 'min' in fselect:
            cols = np.where(X.astype(bool).sum(axis=0) > featmin)[0]
            X = X.iloc[:, cols]
            X_test = X_test.iloc[:, cols]

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

        if hmap:
            heatmap(X, y, tail='_{0}_train'.format(i))
            heatmap(X_test, y_test, tail='_{0}_test'.format(i))

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

    results = stats.ttest_1samp(acc, popmean=755/2126)
    p_val = results[1]

    results = stats.ttest_1samp(acc_tr, popmean=755/2126)
    p_val_tr = results[1]

    return np.mean(acc), np.std(acc), p_val, np.mean(acc_tr), np.std(acc_tr), p_val_tr
