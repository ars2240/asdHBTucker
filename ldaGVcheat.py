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


def lda(fname, indF, nTopics=20, passes=1, iterations=50):
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
        ofname = 'cancer_py_gvLDA_' + str(nTopics) + '_' + str(i)
        ch = logging.FileHandler('logs/' + ofname + '.log', mode='w')
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s : %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # training set
        rowsT = np.where(ind != i)
        X = phi
        X_corp = Dense2Corpus(np.array(X), documents_columns=False)

        # valid set
        rowsV = np.where(ind == i)
        X_test = phi.iloc[rowsV]
        X_testcorp = Dense2Corpus(np.array(X_test), documents_columns=False)

        lda = LdaModel(X_corp, nTopics, alpha='auto', passes=passes, iterations=iterations)
        ofname = 'data/' + ofname
        lda.save(ofname + '_2_cheat_model')
        gvTop = pd.DataFrame(lda.get_topics())
        gvTop.columns = gvID
        gvTop.to_csv(ofname + '_2_cheat_genes.csv')
        pd.DataFrame(lda.alpha).to_csv(ofname + '_2_cheat_alpha.csv')
        patTop = pd.DataFrame(get_doc_topic(X_corp, lda))
        patTop.index = patID
        patTop.to_csv(ofname + '_2_cheat_train.csv')
        patTop = pd.DataFrame(get_doc_topic(X_testcorp, lda))
        patTop.index = patID[rowsV]
        patTop.to_csv(ofname + '_2_cheat_valid.csv')

        logger.removeHandler(ch)  # stop log

