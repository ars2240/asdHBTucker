# ldaGVcancer.py
#
# Author: Adam Sandler
# Date: 12/19/18
#
# Computes LDA decomposition
#
#
# Dependencies:
#   Packages: gensim, numpy, pandas

from gensim.matutils import Dense2Corpus
from gensim.models.ldamodel import LdaModel
import logging
import math
import numpy as np
import pandas as pd


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_doc_topic(corpus, model):
    doc_topic = list()
    for doc in corpus:
        topics = model.__getitem__(doc, eps=0)
        topics = np.asarray(topics)
        doc_topic.append(topics[:, 1])
    return doc_topic


"""""
def get_doc_topic(corpus, model):
    doc_topic = list()
    for doc in corpus:
        gamma, _ = model.inference([doc])
        topics = gamma[0] / sum(gamma[0])
        doc_topic.append(topics)
    return doc_topic
"""""


def lda(fname, indF, nTopics=20, passes=1, iterations=50, fmax=math.inf, head='cancer_py_gen_gvLDA_'):
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

        # initialize log
        ofname = head + str(nTopics) + '_' + str(i)
        ch = logging.FileHandler('logs/' + ofname + '.log', mode='w')
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s : %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # training set
        rowsT = np.where(ind != i)
        X = np.asarray(phi.iloc[rowsT])
        cols = X.sum(axis=0) < fmax
        X = X[:, cols]
        X_corp = Dense2Corpus(np.array(X), documents_columns=False)

        # valid set
        rowsV = np.where(ind == i)
        X_test = np.asarray(phi.iloc[rowsV])
        X_test = X_test[:, cols]
        X_testcorp = Dense2Corpus(np.array(X_test), documents_columns=False)

        lda = LdaModel(X_corp, nTopics, alpha='auto', passes=passes, iterations=iterations)
        ofname = 'data/' + ofname
        lda.save(ofname + '_model')
        gvTop = pd.DataFrame(lda.get_topics())
        gvTop.columns = np.asarray(gvID)[cols]
        gvTop.to_csv(ofname + '_genes.csv')
        pd.DataFrame(lda.alpha).to_csv(ofname + '_alpha.csv')
        patTop = pd.DataFrame(get_doc_topic(X_corp, lda))
        patTop.index = patID[rowsT]
        patTop.to_csv(ofname + '_train.csv')
        patTop = pd.DataFrame(get_doc_topic(X_testcorp, lda))
        patTop.index = patID[rowsV]
        patTop.to_csv(ofname + '_valid.csv')

        logger.removeHandler(ch)  # stop log

