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
import tensorly as tl
import time


# impose dense on variables that may be sparse
def impose(x, sp):
    if sp:
        return x.todense()
    else:
        return x


def ten_dec(fname='cancerSparse', indF='cancerCVInd', rank=5, fselect='min dupe', fmin=0, fmax=1000, thresh=0,
            head='./data/cancer_tensorlyCP', sp=True):
    if '.pik' in fname:
        import pickle
        with open(fname, "rb") as f:
            cts = pickle.load(f)[0]
    else:
        cts = np.array(pd.read_csv(fname + '.csv', header=0, index_col=0, dtype={0: str}))

    if sp:
        from tensorly.contrib.sparse import tensor
        from tensorly.contrib.sparse.decomposition import parafac

        s = np.max(cts[:, :-1], axis=0)
        p = sparse.COO(np.transpose(cts[:, :-1] - 1), cts[:, -1], shape=tuple(s))
    else:
        from tensorly import tensor
        from tensorly.decomposition import parafac

        p = cts

    print(p.shape)
    # cts = pd.read_csv(fname + '.csv', header=None, index_col=None)
    ind = pd.read_csv(indF + '.csv', header=None)

    # training set
    indT, _ = np.where(ind > 0)
    indT = indT[indT < p.shape[0]]
    X = p[indT]

    if 'min' in fselect:
        # remove slices with min or fewer occurances
        cols = (X.astype(bool).max(axis=2).sum(axis=0) > fmin)
        cols = impose(cols, sp)
        # print(np.where((X.astype(bool).sum(axis=(0, 2)) <= fmin).todense())[0])
        X = X[:, cols, :]
        cols = (X.astype(bool).max(axis=1).sum(axis=0) > 0)
        cols = impose(cols, sp)
        X = X[:, :, cols]
    if 'max' in fselect:
        # remove slices with max or more occurances
        cols = (X.astype(bool).max(axis=2).sum(axis=0) < fmax)
        cols = impose(cols, sp)
        # print((X.astype(bool).sum(axis=(0, 2)) >= fmin).todense())
        X = X[:, cols, :]
        cols = (X.astype(bool).max(axis=1).sum(axis=0) > 0)
        cols = impose(cols, sp)
        X = X[:, :, cols]
    if 'thresh' in fselect:
        # change elements below threashold to zero
        X[X < thresh] = 0
    if 'dupe' in fselect:
        s = X.shape
        # check for and remove duplicate slices
        dupes = []
        for j in range(1, s[2]):
            for k in range(0, j):
                # start = time.time()
                temp = X[:, :, j] - X[:, :, k]
                abSum = np.abs(temp).sum()
                tSum = temp.sum()
                a = X[:, :, j].reshape((-1))
                b = X[:, :, k].reshape((-1))
                if np.dot(a, b) ** 2 == np.dot(a, a) * np.dot(b, b):
                    print('%d = %d' % (j, k))
                    dupes.append(j)
                    break
                elif tSum == abSum:
                    print('%d in %d' % (k, j))
                    dupes.append(k)
                elif tSum == -abSum:
                    print('%d in %d' % (j, k))
                    dupes.append(j)
                    break
                # end = time.time()
                # t = end - start
                # print(t)
        print(dupes)
        # print(len(dupes))
        cols = list(set(range(0, s[2])).difference(set(dupes)))
        X = X[:, :, cols]

        dupes = []
        for j in range(1, s[1]):
            for k in range(0, j):
                temp = X[:, j, :] - X[:, k, :]
                abSum = np.abs(temp).sum()
                tSum = temp.sum()
                a = X[:, j, :].reshape((-1))
                b = X[:, k, :].reshape((-1))
                if np.dot(a, b) ** 2 == np.dot(a, a) * np.dot(b, b):
                    print('%d = %d' % (j, k))
                    dupes.append(j)
                    break
                elif tSum == abSum:
                    print('%d in %d' % (k, j))
                    dupes.append(k)
                elif tSum == -abSum:
                    print('%d in %d' % (j, k))
                    dupes.append(j)
                    break
        print(dupes)
        cols = list(set(range(0, s[1])).difference(set(dupes)))
        X = X[:, cols, :]

        cols = (X.astype(bool).max(axis=2).sum(axis=1) > 0)
        cols = impose(cols, sp)
        X = X[cols, :, :]
        print(X.shape)

        s = X.shape
        dupes = []
        for j in range(1, s[0]):
            for k in range(0, j):
                a = X[j, :, :].reshape((-1))
                b = X[k, :, :].reshape((-1))
                if np.dot(a, b) ** 2 == np.dot(a, a) * np.dot(b, b):
                    print('%d = %d' % (j, k))
                    dupes.append(j)
                    break
        print(dupes)
        cols = list(set(range(0, s[1])).difference(set(dupes)))
        X = X[cols, :, :]
        print(X.shape)

    cols = (X.astype(bool).max(axis=2).sum(axis=1) > 0)
    cols = impose(cols, sp)
    X = X[cols, :, :]
    print(X.shape)

    Xsp = np.c_[X.coords.T, X.data.T]
    np.savetxt('Xsp.csv', Xsp, delimiter=',')

    phiT = tensor(X, dtype=tl.float32)
    weights, factors = parafac(phiT, rank=rank, init='random')
    np.savetxt('{0}_{1}_weights.csv'.format(head, rank), impose(weights, sp), delimiter=',')
    for i, f in enumerate(factors):
        np.savetxt('{0}_{1}_{2}.csv'.format(head, rank, i), impose(f, sp), delimiter=',')


ten_dec(fname='cancerSparseND4', rank=25, fselect='min', fmin=0)
