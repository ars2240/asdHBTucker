# generatedDataClass.py
#
# Author: Adam Sandler
# Date: 3/12/19
#
# Creates classes and Computes LDA decomposition on generated data
#
#
# Dependencies:
#   Packages: matplotlib, numpy, scipy, sklearn
#   Files: ldaCust
#   Data: asdHBTucker

# load packages
from ldaGV import lda

nTopics = 20  # number of topics for LDA
fname = 'cancerHBTGenNumberGV'  # count file name
indF = 'cancerGenCVInd'  # index file name

# ldaDecompositions
lda(fname, indF, nTopics, iterations=100, head='cancer_py_HBTgenGV_gvLDA_')
