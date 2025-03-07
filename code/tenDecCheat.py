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

from file_manip import *
import numpy as np
import pandas as pd
import scipy.stats as stats
import sparse
import time


def ten_dec(fname='cancerSparseND4', indF='cancerCVInd', rank=5, fselect='min dupe', fmin=0, fmax=1000, thresh=0,
            head='r8p2_tensorlyCP_nonNeg_{fmin}_{fmax}', sp=True, decomp=True, norm=True, ll=1000,
            hist=False, train_split=True):

    check_folder('./data')

    # change head for variables
    dic = locals()
    for var in dic.keys():
        head = head.replace('{' + var + '}', str(dic[var]))

    cts = load_file(fname)
    cts = cts.astype(int)

    if sp:
        import tensorly.contrib.sparse as tl
        from tensorly.contrib.sparse.decomposition import non_negative_parafac

        s = np.max(cts[:, :-1], axis=0)
        X = sparse.COO(np.transpose(cts[:, :-1] - 1), cts[:, -1], shape=tuple(s))
    else:
        import tensorly as tl
        from tensorly.decomposition import non_negative_parafac

        X = cts

    print(X.shape)
    if train_split:
        # cts = pd.read_csv(fname + '.csv', header=None, index_col=None)
        ind = pd.read_csv(indF + '.csv', header=None)

        # training set
        indT, _ = np.where(ind > 0)
        indT = indT[indT < X.shape[0]]
        X = X if 'r8p_sparse3' in fname else X[indT]

    if 'r8' in fname:
        a1, a2 = 2, 1
    else:
        a1, a2 = 1, 2

    if 'min' in fselect:
        # remove slices with min or fewer occurances
        X1 = X.astype(int).sum(axis=a2) if 'r8' in fname else X.astype(bool).max(axis=a2)
        cols = impose(X1.sum(axis=0) > fmin, sp)
        # print(np.where((X.astype(bool).sum(axis=(0, 2)) <= fmin).todense())[0])
        X = X[:, :, cols] if 'r8' in fname else X[:, cols, :]
    if 'max' in fselect:
        # remove slices with max or more occurances
        X1 = X.astype(int).sum(axis=a2) if 'r8' in fname else X.astype(bool).max(axis=a2)
        cols = impose(X1.sum(axis=0) < fmax, sp)
        # print((X.astype(bool).sum(axis=(0, 2)) >= fmin).todense())
        X = X[:, :, cols] if 'r8' in fname else X[:, cols, :]
    # clean up pathways
    X2 = X.astype(int).sum(axis=a1) if 'r8' in fname else X.astype(bool).max(axis=a1)
    cols = impose(X2.sum(axis=0) > 0, sp)
    X = X[:, cols, :] if 'r8' in fname else X[:, :, cols]
    X2 = X.astype(int).sum(axis=a1) if 'r8' in fname else X.astype(bool).max(axis=a1)
    _, cols = np.unique(impose(X2, sp), axis=1, return_index=True)
    X = X[:, cols, :] if 'r8' in fname else X[:, :, cols]
    if 'thresh' in fselect:
        # change elements below threashold to zero
        X[X < thresh] = 0
    if 'simp_dupe' in fselect:
        # simpler version of dupe
        gp = impose(X.sum(axis=0) > 0, sp)
        if 'r8' in fname:
            _, cols = np.unique(gp, axis=0, return_index=True)
            X = X[:, cols, :]
        else:
            _, cols = np.unique(gp, axis=1, return_index=True)
            X = X[:, :, cols]
    elif 'dupe' in fselect:
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

    """
    cols = (X.astype(bool).max(axis=2).sum(axis=1) > 0)
    cols = impose(cols, sp)
    X = X[cols, :, :]
    """
    print(X.shape)
    print('Sparsity: {0}'.format(X.astype(bool).sum()/np.prod(X.shape)))
    if hist:
        check_folder('./plots')
        import matplotlib.pyplot as plt
        bins = X.max()
        if sp:
            plt.hist(X.data, bins=bins)
        else:
            plt(X, bins=bins)
        plt.savefig('./plots/{0}_hist.png'.format(head))

    Xsp = np.c_[X.coords.T, X.data.T]
    np.savetxt('Xsp.csv', Xsp, delimiter=',')

    phiT = tl.tensor(X, dtype=tl.float32)
    if decomp:
        start_time = time.time()
        weights, factors = non_negative_parafac(phiT, rank=rank, init='random')
        np.savetxt('./data/{0}_{1}_weights.csv'.format(head, rank), impose(weights, sp), delimiter=',')
        for i, f in enumerate(factors):
            np.savetxt('./data/{0}_{1}_{2}.csv'.format(head, rank, i), impose(f, sp), delimiter=',')
        print("Decomp Time: {0}".format(time.time() - start_time))
    else:
        file = './data/{0}_{1}_weights.csv'.format(head, rank)
        weights = load_csv(file)
        if sp:
            weights = sparse.COO(weights)
        factors = []
        for i in range(3):
            file = './data/{0}_{1}_{2}.csv'.format(head, rank, i)
            factors.append(load_csv(file))
            if sp:
                factors[i] = sparse.COO(factors[i])

    if norm or ll > 0:
        if sp:
            import tensorly as tl
            weights = weights.todense()
            for i in range(3):
                factors[i] = factors[i].todense()
        splits = np.max(np.array(ind)) if train_split else 1

    # norm of difference between decomposed tensor and original
    if norm:
        factors_t = list(factors)
        # training set
        psq = 0
        for j in range(X.shape[0]):
            x = X[j]
            factors_t[0] = np.expand_dims(factors[0][j, :], axis=0)
            temp = tl.kruskal_to_tensor((weights, factors_t))[0]
            psq += np.power(x - temp, 2).sum()

        if splits>1:
            print('Train Norm: {0}'.format(np.sqrt(psq * (splits - 1) / splits)))
            print('Valid Norm: {0}'.format(np.sqrt(psq / splits)))
        else:
            print('Norm: {0}'.format(np.sqrt(psq)))

    # log-likelihood
    if ll > 0:
        # transform/normalize tensor
        for i in range(1, 3):
            fsum = np.sum(factors[i], axis=0)
            weights *= fsum
            factors[i] /= fsum
        fsum = np.sum(weights)
        factors[0] *= fsum
        weights /= fsum
        psum = np.sum(factors[0], axis=1)
        factors[0] /= psum[:, None]

        # fit parameters
        # lam = psum.mean()
        alpha = np.mean(factors[0], axis=0)

        # generate samples
        # sim_cts = stats.poisson.rvs(lam, size=ll)
        sim_dist = stats.dirichlet.rvs(alpha, size=ll)

        eps = 1/X.shape[1]*X.shape[2]
        l = 0
        for j in range(X.shape[0]):
            start_time = time.time()
            x = impose(X[j], sp)
            indF = np.where(x > 0)
            x = np.reshape(x, -1)
            x = x[np.where(x > 0)]
            factors_t = [sim_dist, factors[1][indF[0], :]*factors[2][indF[1], :]]
            tens = tl.kruskal_to_tensor((weights, factors_t))
            tens = np.reshape(tens, (ll, -1)) + eps
            p = x*(np.log(tens)-np.log(1+eps*tens.shape[1]))
            lP = np.sum(p, axis=1)
            p = np.exp(lP)
            m = 0
            k = 1
            m = k * np.mean(lP)
            p = np.exp(lP - m)
            w = p/np.sum(p)
            # handling of underflow/overflow error
            while np.sum(p) == 0 or np.sum(w) == 0 or np.isinf(np.sum(p)) or np.isinf(np.sum(w)):
                if np.sum(p) == 0 or np.sum(w) == 0:
                    k = k * 1.5
                else:
                    k = k / 2
                m = k * np.mean(lP)
                p = np.exp(lP - m)
                w = p / np.sum(p)
            l += np.log(np.sum(w * p)) + m
            if np.isinf(l):
                print(np.log(np.sum(w * p)))
                print(np.log(np.sum(p)))
                print(m)
                print(p)
                break
            # print("{0}, {1}, {2}".format(j, l, time.time() - start_time))
        print('Log-likelihood: {0}'.format(l/X.shape[0]))

"""
for i in range(12):
    ten_dec(fname='toy.mat', rank=5, fselect='', train_split=False)
"""
# ten_dec(fname='cancerSparseND4.csv', indF='cancerCVInd', fselect='min max', rank=200, fmin=200, fmax=2000)
# ten_dec(fname='r8p_sparse.csv', indF='r8CVInd', fselect='min max', rank=200, fmin=200, fmax=2000)
# ten_dec(fname='r8p_sparse3.csv', indF='r8CVInd', fselect='', rank=200, fmin=200, fmax=2000)
# ten_dec(fname='asdSparseND.csv', indF='asdCVInd', fselect='', rank=200, fmin=200, fmax=2000)
ten_dec(fname='cnn_sparse2.csv', indF='cnnCVInd', fselect='', rank=200, fmin=30000, fmax=50000)
