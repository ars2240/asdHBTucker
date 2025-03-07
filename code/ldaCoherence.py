# ldaCoherence.py
#
# Author: Adam Sandler
# Date: 9/18/18
#
# Computes LDA model coherence
#
#
# Dependencies:
#   Packages: gensim, numpy, pandas

from gensim.corpora import Dictionary
from gensim.matutils import Dense2Corpus
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import math
import numpy as np
import pandas as pd

nTopics = 20  # number of LDA topics
header = 'cancer_py_LDAgen_gvLDA_'  # file header
coh_meas = 'u_mass'  # coherence measure
fname = 'cancerLDAGenNumber'  # count file name
indF = 'cancerGenCVInd'  # index file name

# min & max allowable feature sum
fmax = math.inf
fmin = 0

coh_tr = np.zeros(10)
coh_te = np.zeros(10)

cts = pd.read_csv(fname + '.csv', header=0, index_col=0, dtype={0: str})
# cts = pd.read_csv(fname + '.csv', header=None, index_col=None)
ind = pd.read_csv(indF + '.csv', header=None)
patID = cts.index
gvID = cts.columns
rows = np.where(ind > 0)[0]
splits = np.max(np.array(ind))
patID = patID[rows]
phi = cts.iloc[rows]
ind = ind.iloc[rows, 0]

for i in range(1, 11):
    fname = './data/' + header + str(nTopics) + '_' + str(i)
    genes = pd.read_csv(fname + '_genes.csv', header=0, index_col=0)

    # training set
    rowsT = np.where(ind != i)
    X = np.asarray(phi.iloc[rowsT])
    cols = np.logical_and(fmin < X.sum(axis=0), X.sum(axis=0) < fmax)
    X = X[:, cols]
    X_corp = Dense2Corpus(np.array(X), documents_columns=False)

    # valid set
    rowsV = np.where(ind == i)
    X_test = np.asarray(phi.iloc[rowsV])
    X_test = X_test[:, cols]
    X_testcorp = Dense2Corpus(np.array(X_test), documents_columns=False)

    dic = Dictionary.from_corpus(X_corp)

    model = LdaModel.load(fname + '_model')  # load model

    cm = CoherenceModel(model=model, corpus=X_corp, dictionary=dic, coherence=coh_meas)
    coh_tr[i-1] = cm.get_coherence()
    cm = CoherenceModel(model=model, corpus=X_testcorp, dictionary=dic, coherence=coh_meas)
    coh_te[i - 1] = cm.get_coherence()

print("Mean Train Coherence: " + str(np.mean(coh_tr)))
print("Mean Test Coherence: " + str(np.mean(coh_te)))
