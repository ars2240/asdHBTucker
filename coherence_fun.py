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


def load_file(f):
    if '.pik' in f:
        with open(f, "rb") as f:
            d = pickle.load(f)[0]
    elif '.mat' in f:
        mdict = scipy.io.loadmat(f)  # import dataset from matlab
        d = mdict.get('sparse')
    else:
        d = load_csv(f)
    return d


def coherence(fname, counts, indF, dim=1, splits=10, fmax=math.inf, fmin=0, sp=False, root='./data/',
              coh_meas='u_mass'):

    coh_tr = np.zeros(splits)
    coh_te = np.zeros(splits)

    cts = load_file(counts)

    if sp:
        s = np.max(cts[:, :-1], axis=0)
        cts = sparse.COO(np.transpose(cts[:, :-1] - 1), cts[:, -1], shape=tuple(s))

    t = list(range(1, cts.ndim))
    t.remove(dim)
    cts = np.amax(cts, axis=tuple(t))

    ind = pd.read_csv(indF + '.csv', header=None)
    rows = np.where(ind > 0)[0]
    phi = cts[rows, :]
    ind = ind.iloc[rows, 0]

    cols = np.logical_and(fmin < phi.sum(axis=0), phi.sum(axis=0) < fmax)
    phi = phi[:, cols]

    if '{i}' not in fname:
        psi = load_file(root + fname)

    for i in range(splits):

        if '{i}' in fname:
            psi = load_file(root + fname.replace('{i}', str(i+1)))

        if '{i}' in fname:
            p = psi[(0, 0)] if psi.shape == (1, 1) else psi
        elif psi.shape == (splits, 1):
            p = psi[(i, 0)]
        else:
            p = psi

        (r, c) = np.nonzero(p > .01)
        t = [[str(r[x]) for x in range(len(r)) if c[x] == y] for y in np.unique(c)]

        # training set
        rowsT = np.where(ind != (i + 1))
        X_corp = Dense2Corpus(np.array(phi[rowsT, :]), documents_columns=False)

        # valid set
        rowsV = np.where(ind == (i + 1))
        X_testcorp = Dense2Corpus(np.array(phi[rowsV, :]), documents_columns=False)

        dic = Dictionary.from_corpus(X_corp)

        cm = CoherenceModel(topics=t, corpus=X_corp, dictionary=dic, coherence=coh_meas)
        coh_tr[i] = cm.get_coherence()
        cm = CoherenceModel(topics=t, corpus=X_testcorp, dictionary=dic, coherence=coh_meas)
        coh_te[i] = cm.get_coherence()

    return coh_tr, coh_te



