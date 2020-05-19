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
            head='./data/cancer_tensorlyCP', sp=True, decomp=True, norm=True):
    if '.pik' in fname:
        import pickle
        with open(fname, "rb") as f:
            cts = pickle.load(f)[0]
    else:
        cts = np.array(pd.read_csv(fname + '.csv', header=0, index_col=0, dtype={0: str}))

    if sp:
        import tensorly.contrib.sparse as tl

        s = np.max(cts[:, :-1], axis=0)
        p = sparse.COO(np.transpose(cts[:, :-1] - 1), cts[:, -1], shape=tuple(s))
    else:
        import tensorly as tl

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

    phiT = tl.tensor(X, dtype=tl.float32)
    if decomp:
        weights, factors = tl.parafac(phiT, rank=rank, init='random')
        np.savetxt('{0}_{1}_weights.csv'.format(head, rank), impose(weights, sp), delimiter=',')
        for i, f in enumerate(factors):
            np.savetxt('{0}_{1}_{2}.csv'.format(head, rank, i), impose(f, sp), delimiter=',')
    else:
        weights = np.squeeze(pd.read_csv('{0}_{1}_weights.csv'.format(head, rank), header=None).to_numpy())
        if sp:
            weights = sparse.COO(weights)
        factors = []
        for i in range(3):
            factors.append(pd.read_csv('{0}_{1}_{2}.csv'.format(head, rank, i), header=None).to_numpy())
            if sp:
                factors[i] = sparse.COO(factors[i])

    if norm:
        if sp:
            import tensorly as tl
            weights = weights.todense()
            for i in range(3):
                factors[i] = factors[i].todense()
        splits = np.max(np.array(ind))
        tr_norm = np.zeros(splits)
        va_norm = np.zeros(splits)
        factors_t = list(factors)
        for i in range(1, splits+1):
            # training set
            indTr = np.where(indT != i)[0]
            psq = 0
            for j in range(len(indTr)):
                x = X[indTr[j]]
                factors_t[0] = np.expand_dims(factors[0][indTr[j], :], axis=0)
                temp = tl.kruskal_to_tensor((weights, factors_t))[0]
                psq += np.power(x - temp, 2).sum()
            tr_norm[i] = np.sqrt(psq)

            # valid set
            indV = np.where(indT == i)[0]
            psq = 0
            for j in range(len(indV)):
                x = X[indV[j]]
                factors_t[0] = np.expand_dims(factors[0][indV[j], :], axis=0)
                temp = tl.kruskal_to_tensor((weights, factors_t))[0]
                psq += np.power(x - temp, 2).sum()
            va_norm[i] = np.sqrt(psq)

        print('Train Norm: {0}'.format(np.mean(tr_norm)))
        print('Valid Norm: {0}'.format(np.mean(va_norm)))


ten_dec(fname='cancerSparseND4', rank=10, fselect='min', fmin=0, decomp=False)
