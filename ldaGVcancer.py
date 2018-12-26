# ldaGVcancer.py
#
# Author: Adam Sandler
# Date: 12/7/18
#
# Computes LDA decomposition for cancer dataset
#
#
# Dependencies:
#   Packages: gensim, numpy, pandas
#   Files: ldaGV
#   Data: cancerSparseGenes, cancerCVInd

from ldaGV import lda

nTopics = 50  # number of topics for LDA
fname = 'cancerNumber'  # count file name
indF = 'cancerCVInd'  # index file name

lda(fname, indF, nTopics, passes=100)

