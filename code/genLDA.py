# genLDA.py
#
# Author: Adam Sandler
# Date: 9/12/19
#
# Creates classes and Computes LDA decomposition on generated data
#
#
# Dependencies:
#   Packages: numpy, pandas, scipy
#   Data: cancerNumber

# load packages
import numpy as np
import pandas as pd
import scipy.stats as st

fname = 'cancerNumber'  # count file name
tops = 20  # number of topics

cts = pd.read_csv(fname + '.csv', header=0, index_col=0, dtype={0: str})

genTops = np.random.dirichlet(1/tops*np.ones(tops), size=cts.shape[0])
pd.DataFrame(genTops).to_csv('cancerLDAGenTops.csv')

ctsColSum = cts.sum(axis=0)
ctsColSum /= ctsColSum.sum()
genTopGVs = np.random.dirichlet(ctsColSum, size=tops)
pd.DataFrame(genTopGVs).to_csv('cancerLDAGenTopGVs.csv')

ctsRowSum = cts.sum(axis=1)
lam = ctsRowSum.mean()
genCts = st.poisson.rvs(lam, size=cts.shape[0])

ctsGen = np.zeros(cts.shape)
for i in range(0, cts.shape[0]):
    top = np.random.binomial(genCts[i], genTops[i, :])
    for j in range(0, tops):
        ctsGen[i, :] += np.random.binomial(top[j], genTopGVs[j, :])

pd.DataFrame(ctsGen).to_csv('cancerLDAGenNumber.csv')
