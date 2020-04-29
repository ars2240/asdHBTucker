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
import time


def ten_dec(fname='cancerSparse', indF='cancerCVInd', rank=5, fselect='min dupe', fmin=0, fmax=1000, thresh=0,
            head='cancer_py_tenDec_', sp=True):
    if sp:
        from tensorly.contrib.sparse import tensor
        from tensorly.contrib.sparse.decomposition import parafac
    else:
        from tensorly import tensor
        from tensorly.decomposition import parafac
    if '.pik' in fname:
        import pickle
        with open(fname, "rb") as f:
            cts = pickle.load(f)[0]
    else:
        cts = np.array(pd.read_csv(fname + '.csv', header=0, index_col=0, dtype={0: str}))
    if sp:
        s = np.max(cts[:, :-1], axis=0)
        p = sparse.COO(np.transpose(cts[:, :-1] - 1), cts[:, -1], shape=tuple(s))
        if not sp:
            p = p.todense()
    else:
        p = cts
    print(p.shape)
    # cts = pd.read_csv(fname + '.csv', header=None, index_col=None)
    ind = pd.read_csv(indF + '.csv', header=None)
    splits = np.max(np.array(ind))

    for i in range(1, splits+1):

        # training set
        indT, _ = np.where(np.logical_and(ind != i, ind > 0))
        indT = indT[indT < p.shape[0]]
        X = p[indT]

        # valid set
        indV, _ = np.where(ind == i)
        indV = indV[indV < p.shape[0]]
        Xv = p[indV]
        print(X.shape)

        if 'min' in fselect:
            # remove slices with min or fewer occurances
            cols = (X.astype(bool).max(axis=2).sum(axis=0) > fmin)
            if sp:
                cols = cols.todense()
            #print(np.where((X.astype(bool).sum(axis=(0, 2)) <= fmin).todense())[0])
            X = X[:, cols, :]
            Xv = Xv[:, cols, :]
            cols = (X.astype(bool).max(axis=1).sum(axis=0) > 1)
            if sp:
                cols = cols.todense()
            X = X[:, :, cols]
            Xv = Xv[:, :, cols]
            print(X.shape)
        if 'max' in fselect:
            # remove slices with max or more occurances
            cols = (X.astype(bool).max(axis=2).sum(axis=0) < fmax)
            if sp:
                cols = cols.todense()
            #print((X.astype(bool).sum(axis=(0, 2)) >= fmin).todense())
            X = X[:, cols, :]
            Xv = Xv[:, cols, :]
            cols = (X.astype(bool).max(axis=1).sum(axis=0) > 1)
            if sp:
                cols = cols.todense()
            X = X[:, :, cols]
            Xv = Xv[:, :, cols]
            print(X.shape)
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
                    temp = X[:, :, j] - X[:, :, k]
                    abSum = np.abs(temp).sum()
                    tSum = temp.sum()
                    if abSum == 0:
                        print('%d, %d' % (j, k))
                        dupes.append(j)
                        break
                    elif tSum == abSum:
                        print('%d in %d' % (k, j))
                        dupes.append(k)
                        break
                    elif tSum == -abSum:
                        print('%d in %d' % (j, k))
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
                    temp = X[:, j, :] - X[:, k, :]
                    abSum = np.abs(temp).sum()
                    tSum = temp.sum()
                    if abSum == 0:
                        print('%d, %d' % (j, k))
                        dupes.append(j)
                        break
                    elif tSum == abSum:
                        print('%d in %d' % (k, j))
                        dupes.append(k)
                        break
                    elif tSum == -abSum:
                        print('%d in %d' % (j, k))
                        dupes.append(j)
                        break
            print(dupes)
            cols = list(set(range(0, s[1])).difference(set(dupes)))
            X = X[:, cols, :]
            Xv = Xv[:, cols, :]
        print(X.shape)
        cols = (X.astype(bool).max(axis=2).sum(axis=1) > 0)
        if sp:
            cols = cols.todense()
        X = X[cols, :, :]
        print(X.shape)
        Xsp = np.c_[X.coords.T, X.data.T]
        np.savetxt('Xsp.csv', Xsp, delimiter=',')

        phiT = tensor(X)
        factors = parafac(phiT, rank=rank, init='random')
        print(factors)


ten_dec(fname='cancerSparseND5', fselect='min dupe', fmin=200)
