# coherence.py
#
# Author: Adam Sandler
# Date: 2/10/21
#
# Computes model coherence
#
#
# Dependencies:
#   Packages: gensim, numpy, pandas

from gensim.corpora import Dictionary
from gensim.matutils import Dense2Corpus
from gensim.models.coherencemodel import CoherenceModel
import math
import numpy as np
import pandas as pd
import pickle
import scipy.io
import sparse


def load_csv(f):
    try:
        d = pd.read_csv(f, header=None, dtype=float)
        d2 = d.iloc[0, 0]
    except ValueError:
        d, d2 = [], float('nan')
    if math.isnan(d2):
        d = pd.read_csv(f, header=0, index_col=0, dtype={0: str})
    return np.array(d)


def load_file(f, s=''):
    if '.pik' in f:
        with open(f, "rb") as f:
            d = pickle.load(f)[0]
    elif '.mat' in f:
        mdict = scipy.io.loadmat(f)  # import dataset from matlab
        d = mdict.get(s)
    else:
        d = load_csv(f)
    return d


def impose(x, sp):
    return x.todense() if sp else x


def get_tl(fname, d):
    if 'PAM' in fname:
        if (d == 1 and 'pwy' not in fname.lower()) or (d == 2 and 'pwy' in fname.lower()):
            tl = [None, [0], list(range(1, 11)), list(range(11, 21))]
        else:
            tl = [None, list(range(10)), list(range(10, 20)), list(range(20, 30))]
    elif 'IndepTrees' in fname or 'hLDA' in fname:
        tl = [None, 0, 1, 2]
    else:
        tl = [None]
    if 'L2' in fname and len(tl) > 1:
        tl = tl[:-1]
    return tl


# compute coherence over topics
def coh(X, t, n=5, eps=1e-5, topics=None, mean=True, meas='uci'):
    x = (X > 0).astype(int)  # convert to whether or not a word occurs
    dw = np.dot(np.transpose(x), x) + eps  # word co-occurrence
    p = np.log(dw)  # log probabilities
    if 'mass' not in meas.lower():
        p -= np.log((X.shape[0] * (1 + eps)))
    d = np.diagonal(p)  # diagonal is occurrence of words
    if 'mass' in meas.lower():
        s = np.log(dw + 1) - d.transpose()
    else:
        s = p - np.add.outer(d, d)  # compute UCI/PMI score

    if topics is None:
        nt = t.shape[1]  # number of topics
        topics = range(nt)
    else:
        nt = len(topics)
    c = np.zeros(nt)  # initialize topic coherence vector
    ui = []  # list of unique words
    for i in range(nt):
        idx = np.argsort(-t[:, topics[i]])[:n]  # get top n words in topic
        c[i] = np.sum(np.tril(s[np.ix_(idx, idx)]))  # get topic coherence
        ui.extend(idx)
    nu = len(np.unique(ui))/len(ui)  # % of top words that are unique

    c = np.mean(c) if mean else c

    return c, nu


def coherence(fname, counts, indF, dim=1, splits=10, fmax=math.inf, fmin=0, sp=False, root='./data/',
              coh_meas='uci', topics=None):

    coh_tr, coh_te, nu = np.zeros(splits), np.zeros(splits), np.zeros(splits)

    cf = load_file(counts, 'sparse')

    if sp:
        s = np.max(cf[:, :-1], axis=0)
        cf = sparse.COO(np.transpose(cf[:, :-1] - 1), cf[:, -1], shape=tuple(s))

    ind = pd.read_csv(indF + '.csv', header=None)
    rows = np.where(ind > 0)[0]
    ind = ind.iloc[rows, 0]

    cf = cf[rows] > 0
    t, t2 = list(range(1, cf.ndim)), list(range(2, cf.ndim))
    t.remove(dim)
    ct2 = np.max(cf, axis=tuple(t2))

    cols = impose(np.logical_and(fmin < ct2.sum(axis=0), ct2.sum(axis=0) < fmax), sp)
    cts = cf[:, cols, :] if sp else cf[:, cols]
    if dim == 2:
        cols = impose(cts.sum(axis=[0, 1]) > 0, sp)
        cts = cts[:, :, cols]
        gp = impose(cts.sum(axis=0) > 0, sp)
        _, cols = np.unique(gp, axis=1, return_index=True)
        cts = cts[:, :, cols]
    phi = np.max(cts, axis=tuple(t))
    phi = impose(phi, sp)

    if 'pwy' in fname.lower():
        dim = (dim % 2) + 1

    if '{i}' not in fname:
        psi = load_file(root + fname, 'psi')

    for i in range(splits):

        if '{i}' in fname:
            psi = load_file(root + fname.replace('{i}', str(i+1)), 'psi')

        if '{i}' in fname:
            p = psi[(0, 0)] if psi.shape == (1, 1) else psi
        elif psi.shape == (splits, 1):
            p = psi[(i, 0)]
        else:
            p = psi

        if p.shape[1] == 1:
            if 'hLDA' in fname:
                p = p[(0, 0)]
            else:
                p = p[(dim - 1, 0)]

        """
        # t = {str(c): [{str(r): p[r, c]} for r in range(p.shape[0])] for c in range(p.shape[1])}
        t = [[str(r) for r in range(p.shape[0]) if p[r, c] > 1e-5] for c in range(p.shape[1])]

        # training set
        rowsT = np.where(ind != (i + 1))
        X_corp = Dense2Corpus(np.array(phi[rowsT[0], :]), documents_columns=False)

        # valid set
        rowsV = np.where(ind == (i + 1))
        X_testcorp = Dense2Corpus(np.array(phi[rowsV[0], :]), documents_columns=False)

        dic = Dictionary.from_corpus(X_corp)

        cm = CoherenceModel(topics=t, corpus=X_corp, dictionary=dic, coherence=coh_meas)
        coh_tr[i] = cm.get_coherence()
        cm = CoherenceModel(topics=t, corpus=X_testcorp, dictionary=dic, coherence=coh_meas)
        coh_te[i] = cm.get_coherence()
        """
        if isinstance(topics, int):
            tree = load_file(root + fname.replace('{i}', str(i+1)), 'tree')
            if topics == 0:
                t = [0]
            elif topics == 1:
                if 'CP' in fname:
                    t = np.array(tree[(0, 0)][0]) - 1
                elif 'hLDA' in fname:
                    t = np.array(tree[(0, 0)][(0, 0)][0]) - 1
                else:
                    t = np.array(tree[(dim - 1, 0)][(0, 0)][0]) - 1
            else:
                if 'CP' in fname:
                    t1 = np.array(tree[(0, 0)][0]) - 1
                elif 'hLDA' in fname:
                    t1 = np.array(tree[(0, 0)][(0, 0)][0]) - 1
                else:
                    t1 = np.array(tree[(dim - 1, 0)][(0, 0)][0]) - 1
                t1 = np.append(t1, 0)
                t = list(range(p.shape[1]))
                [t.remove(i) for i in t1]
        else:
            t = topics

        # training
        rowsT = np.where(ind != (i + 1))
        coh_tr[i], nu[i] = coh(phi[rowsT[0], :], p, topics=t, meas=coh_meas)
        print(coh_tr[i])

        # validation/test
        rowsV = np.where(ind == (i + 1))
        coh_te[i], _ = coh(phi[rowsV[0], :], p, topics=t, meas=coh_meas)

    return coh_tr, coh_te, nu



