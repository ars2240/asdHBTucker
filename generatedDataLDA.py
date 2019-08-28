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
import pandas as pd

nTopics = 20  # number of topics for LDA
fname = 'cancerGenNumber'  # count file name
indF = 'cancerGenCVInd'  # index file name

# import and format data
cts = pd.read_csv(fname + '.csv', header=0, index_col=0, dtype={0: str})
ind = pd.read_csv(indF + '.csv', header=None)

# ldaDecompositions
lda(fname, indF, nTopics, iterations=100)
