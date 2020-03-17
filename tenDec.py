# tenDec.py
#
# Author: Adam Sandler
# Date: 1/27/20
#
# Computes tensor decompositions
#
#
# Dependencies:
#   Packages: numpy, scipy, tensorly

import numpy as np
import pandas as pd
import sparse
from tensorly.contrib.sparse import tensor
from tensorly.contrib.sparse.decomposition import parafac
import time


def ten_dec(fname='cancerSparse', indF='cancerCVInd', rank=5, fselect='min dupe', fmin=0, fmax=1000, thresh=0,
            head='cancer_py_tenDec_'):
    cts = np.array(pd.read_csv(fname + '.csv', header=0, index_col=0, dtype={0: str}))
    # cts = pd.read_csv(fname + '.csv', header=None, index_col=None)
    ind = pd.read_csv(indF + '.csv', header=None)
    splits = np.max(np.array(ind))
    s = np.max(cts[:, :-1], axis=0)
    p = sparse.COO(np.transpose(cts[:, :-1] - 1), cts[:, -1], shape=tuple(s))

    for i in range(1, splits+1):

        # training set
        indT, _ = np.where(np.logical_and(ind != i, ind > 0))
        X = p[indT]

        # valid set
        indV, _ = np.where(ind == i)
        Xv = p[indV]

        if 'min' in fselect:
            # remove slices with min or fewer occurances
            cols = (X.astype(bool).max(axis=2).sum(axis=0) > fmin).todense()
            #print(np.where((X.astype(bool).sum(axis=(0, 2)) <= fmin).todense())[0])
            X = X[:, cols, :]
            Xv = Xv[:, cols, :]
            cols = (X.astype(bool).max(axis=1).sum(axis=0) > 0).todense()
            X = X[:, :, cols]
            Xv = Xv[:, :, cols]
        if 'max' in fselect:
            # remove slices with max or more occurances
            cols = (X.astype(bool).max(axis=2).sum(axis=0) < fmax).todense()
            #print((X.astype(bool).sum(axis=(0, 2)) >= fmin).todense())
            X = X[:, cols, :]
            Xv = Xv[:, cols, :]
            cols = (X.astype(bool).max(axis=1).sum(axis=0) > 0).todense()
            X = X[:, :, cols]
            Xv = Xv[:, :, cols]
        if 'thresh' in fselect:
            # change elements below threashold to zero
            X[X < thresh] = 0
            Xv[Xv < thresh] = 0
        if 'dupe' in fselect:
            s = X.shape
            # check for and remove duplicate slices
            dupes = []
            for j in range(1, s[2]):
                for k in range(0, j):
                    #start = time.time()
                    if np.abs(X[:, :, j]-X[:, :, k]).sum() == 0:
                        print('%d, %d' % (j, k))
                        dupes.append(j)
                        break
                    #end = time.time()
                    #t = end - start
                    #print(t)
            print(dupes)
            #print(len(dupes))
            cols = list(set(range(0, s[2])).difference(set(dupes)))
            X = X[:, :, cols]
            Xv = Xv[:, :, cols]
            dupes = []
            for j in range(1, s[1]):
                for k in range(0, j):
                    if np.abs(X[:, j, :]-X[:, k, :]).sum() == 0:
                        print('%d, %d' % (j, k))
                        dupes.append(j)
                        break
            print(dupes)
            cols = list(set(range(0, s[1])).difference(set(dupes)))
            X = X[:, cols, :]
            Xv = Xv[:, cols, :]
        print(X.shape)

        phiT = tensor(X)
        factors = parafac(phiT, rank=rank, init='random')
        print(factors)


ten_dec(fselect='min max dupe', fmin=200, fmax=1000)
