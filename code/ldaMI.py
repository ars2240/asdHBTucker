import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

nTopics = 20
splits = 10
indF = 'cancerCVInd'
labelF = 'cancerLabel'

ind = pd.read_csv(indF + '.csv', header=None)
label = pd.read_csv(labelF + '.csv', header=0, index_col=0)
rows = np.where(ind > 0)[0]
ind = ind.iloc[rows, 0]
cancer = label.iloc[rows, 0]

for i in range(1, splits+1):
    fname = 'data/cancer_gvLDA_' + str(nTopics) + '_' + str(i)
    lda = pd.read_csv(fname + '_train.csv', header=0, index_col=0, dtype={0: str})
    rowsT = np.where(ind != i)
    y = cancer.iloc[rowsT]
    mi = pd.DataFrame(mutual_info_classif(lda, y))
    mi.to_csv(fname + '_mi.csv')
