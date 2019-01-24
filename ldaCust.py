# ldaCust.py
#
# Author: Adam Sandler
# Date: 1/22/18
#
# Computes LDA decomposition
#
#
# Dependencies:
#   Packages: numpy, pandas

import logging
import math
import numpy as np
import pandas as pd
from lda import LDA


"""""
def get_doc_topic(corpus, model):
    doc_topic = list()
    for doc in corpus:
        gamma, _ = model.inference([doc])
        topics = gamma[0] / sum(gamma[0])
        doc_topic.append(topics)
    return doc_topic
"""""


def lda(fname, indF, nTopics=20, iterations=50, fmax=math.inf):
    cts = pd.read_csv(fname + '.csv', header=0, index_col=0, dtype={0: str})
    ind = pd.read_csv(indF + '.csv', header=None)
    patID = cts.index
    gvID = cts.columns
    rows = np.where(ind > 0)[0]
    splits = np.max(np.array(ind))
    patID = patID[rows]
    phi = cts.iloc[rows]
    ind = ind.iloc[rows, 0]

    for i in range(1, splits+1):

        ofname = 'cancer_py_cust_gvLDA_' + str(nTopics) + '_' + str(i)

        # training set
        rowsT = np.where(ind != i)
        X = np.asarray(phi.iloc[rowsT])
        cols = X.sum(axis=0) < fmax
        X = X[:, cols]

        # valid set
        rowsV = np.where(ind == i)
        X_test = np.asarray(phi.iloc[rowsV])
        X_test = X_test[:, cols]

        lda = LDA(nTopics)
        patTop, gvTop = lda.train(X, iters=iterations)
        ofname = 'data/' + ofname
        gvTop = pd.DataFrame(gvTop)
        gvTop.columns = np.asarray(gvID)[cols]
        gvTop.to_csv(ofname + '_genes.csv')
        pd.DataFrame(lda.alpha).to_csv(ofname + '_alpha.csv')
        patTop = pd.DataFrame(patTop)
        patTop.index = patID[rowsT]
        patTop.to_csv(ofname + '_train.csv')
        patTop = lda.predict(X_test, iters=iterations)
        patTop = pd.DataFrame(patTop)
        patTop.index = patID[rowsV]
        patTop.to_csv(ofname + '_valid.csv')

