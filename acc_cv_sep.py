# acc_cv_sep.py
#
# Author: Adam Sandler
# Date: 1/31/20
#
# Computes accuracy for each CV, returns plot in /plots/ folder, and
# mean, stDev, and p-value for both train & validation sets
# Useful for if each CV fold is in a different file
#
# Uses CV data
#
# Dependencies:
#   Packages: matplotlib, numpy, scipy, sklearn
#   Data: asdHBTucker

from hmap import heatmap
import numpy as np
import pandas as pd
from scipy import stats
from scipy import sparse
import scipy.io
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
import torch


def acc(classifier, fname, yfname=None, splits=10, fselect='min', root='./data/', nfeat=100, fmin=0, fmax=1000, a=.05,
        thresh=0, hmap=False):

    acc = []
    acc_tr = []
    # coeffs = []

    if yfname is None:
        yfname = fname

    if '{i}' not in fname:
        # load data
        mdict = scipy.io.loadmat(root + fname)  # import dataset from matlab
        phi = mdict.get('phi')
        testPhi = mdict.get('testPhi')
    if '{i}' not in yfname:
        ymdict = scipy.io.loadmat(root + yfname)  # import dataset from matlab
        asd = ymdict.get('cvTrainASD')
        testASD = ymdict.get('cvTestASD')

    for i in range(0, splits):

        if '{i}' in fname:
            # load data
            mdict = scipy.io.loadmat(root + fname.replace('{i}', str(i+1)))  # import dataset from matlab
            phi = mdict.get('phi')
            testPhi = mdict.get('testPhi')
        if '{i}' in yfname:
            ymdict = scipy.io.loadmat(root + yfname.replace('{i}', str(i+1)))  # import dataset from matlab
            asd = ymdict.get('cvTrainASD')
            testASD = ymdict.get('cvTestASD')

        if '{i}' in fname:
            X = phi
            X_test = testPhi
            if X.shape == (1, 1):
                X = phi[(0, 0)]
            if X_test.shape == (1, 1):
                X_test = testPhi[(0, 0)]
        else:
            X = phi[(i, 0)]
            X_test = testPhi[(i, 0)]
        if isinstance(X, np.void):
            s = X[2][0]
            X = torch.sparse.FloatTensor(torch.from_numpy(X[0].astype(dtype='float32') - 1).t().type(torch.LongTensor),
                                         torch.from_numpy(X[1][:, 0].astype(dtype='float32')), torch.Size(tuple(s)))
            X = X.to_dense().reshape(s[0], -1).numpy()
        if isinstance(X_test, np.void):
            s = X_test[2][0]
            X_test = torch.sparse.FloatTensor(
                torch.from_numpy(X_test[0].astype(dtype='float32') - 1).t().type(torch.LongTensor),
                torch.from_numpy(X_test[1][:, 0].astype(dtype='float32')), torch.Size(tuple(s)))
            X_test = X_test.to_dense().reshape(s[0], -1).numpy()
        s = X.shape
        if len(s) == 3:
            X = np.reshape(X, [s[0], s[1] * s[2]])
        else:
            X = np.reshape(X, [s[0], s[1]])
        if '{i}' in yfname:
            y = asd
            y_test = testASD
        else:
            y = asd[(i, 0)]
            y_test = testASD[(i, 0)]
        y = np.reshape(y, s[0])
        s = X_test.shape
        if len(s) == 3:
            X_test = np.reshape(X_test, [s[0], s[1] * s[2]])
        else:
            X_test = np.reshape(X_test, [s[0], s[1]])
        y_test = np.reshape(y_test, s[0])

        # add zero column if dims don't match
        dif = X.shape[1] - X_test.shape[1]
        if dif > 0:
            z = np.zeros((X_test.shape[0], dif))
            X_test = np.append(X_test, z, 1)

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

        # rescale
        if sparse.issparse(X):
            scaler = StandardScaler(with_mean=False)
        else:
            scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X_test = scaler.transform(X_test)

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

        if hmap:
            heatmap(X, y, tail='_{0}_train'.format(i))
            heatmap(X_test, y_test, tail='_{0}_test'.format(i))

        # fit model
        model = classifier.fit(X, y)

        """
        if i == 0:
            coeffs = np.array(model.coef_).transpose()
        else:
            coeffs = np.c_[coeffs, np.array(model.coef_).transpose()]
        """

        # Compute accuracy for validation set
        y_hat = model.predict(X_test)
        acc.append(sum(y_hat == y_test)/len(y_test))

        # Compute accuracy for training set
        y_hat = model.predict(X)
        acc_tr.append(sum(y_hat == y) / len(y))

        # pd.DataFrame(model.coef_).to_csv('data/cancer_coef_' + str(i) + '.csv')

        i += 1

    #np.savetxt("data/cancer_coeffs.csv", coeffs, delimiter=",")

    results = stats.ttest_1samp(acc, popmean=1066/3037)
    p_val = results[1]

    results = stats.ttest_1samp(acc_tr, popmean=1066/3037)
    p_val_tr = results[1]

    return np.mean(acc), np.std(acc), p_val, np.mean(acc_tr), np.std(acc_tr), p_val_tr
