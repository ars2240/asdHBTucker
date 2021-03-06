# acc_cv_yang.py
#
# Author: Adam Sandler
# Date: 1/13/20
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


def acc(classifier, fname, yfname=None, root='./data/', fselect='min', nfeat=100, fmin=0, fmax=1000, a=.05, thresh=0):

    # load data
    mdict = scipy.io.loadmat(root + fname)  # import dataset from matlab
    if yfname is None:
        ymdict = mdict
    else:
        ymdict = scipy.io.loadmat(root + yfname)  # import dataset from matlab
    phi = mdict.get('phi')
    testPhi = mdict.get('testPhi')
    asd = ymdict.get('asd')
    testASD = ymdict.get('asdTe')

    X = phi[(0, 0)]
    if isinstance(X, np.void):
        s = X[2][0]
        X = torch.sparse.FloatTensor(torch.from_numpy(X[0].astype(dtype='float32')-1).t().type(torch.LongTensor),
                                     torch.from_numpy(X[1][:, 0].astype(dtype='float32')), torch.Size(tuple(s)))
        X = X.to_dense().reshape(s[0], -1).numpy()
    else:
        X = phi
        s = X.shape
    y = asd
    y = np.reshape(y, s[0])
    Xt = testPhi[(0, 0)]
    if isinstance(Xt, np.void):
        s = Xt[2][0]
        Xt = torch.sparse.FloatTensor(torch.from_numpy(Xt[0].astype(dtype='float32') - 1).t().type(torch.LongTensor),
                                     torch.from_numpy(Xt[1][:, 0].astype(dtype='float32')), torch.Size(tuple(s)))
        Xt = Xt.to_dense().reshape(s[0], -1).numpy()
    else:
        Xt = testPhi
        s = Xt.shape
    X_test = Xt
    y_test = testASD
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
    acc = sum(y_hat == y_test)/len(y_test)

    # Compute accuracy for training set
    y_hat = model.predict(X)
    acc_tr = sum(y_hat == y) / len(y)

    # pd.DataFrame(model.coef_).to_csv('data/cancer_coef_' + str(i) + '.csv')

    #np.savetxt("data/cancer_coeffs.csv", coeffs, delimiter=",")

    return acc, acc_tr
