# generatedDataClass.py
#
# Author: Adam Sandler
# Date: 4/5/19
#
# Creates classes and Computes LDA decomposition on generated data
#
#
# Dependencies:
#   Packages: numpy, pandas, sklearn
#   Data: asdHBTucker

# load packages
import math
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import random

nTopics = 20  # number of topics for LDA
rfname = 'cancerGenNumber2'  # random count file name
rlabF = 'cancerGenLabel'  # random labels file name
fname = 'cancerGenNumber'  # count file name
indF = 'cancerGenCVInd'  # index file name
C = 1  # penalty parameter

# import and format data
cts = pd.read_csv(fname + '.csv', header=None)
ind = pd.read_csv(indF + '.csv', header=None)
patID = cts.index
gvID = cts.columns
rows = np.where(ind > 0)[0]
splits = np.max(np.array(ind))
patID = patID[rows]
phi = cts.iloc[rows]
ind = ind.iloc[rows, 0]

# random data
rows = random.sample(range(0, len(cts.index)), 4)
rcts = cts.iloc[rows]
rlabs = range(0, 4)

model = SVC(C=C, kernel='linear', probability=True)  # SVM classifier
model.fit(rcts, rlabs)

probas = model.predict_proba(cts)
y_hat = np.argmax(probas, axis=1)

# randomly flip 10% of the samples
rows = random.sample(range(0, len(y_hat)), math.ceil(len(y_hat)/10))
y_hat[rows] = np.random.randint(0, 4, len(rows))

pd.DataFrame(y_hat).to_csv('cancerHBTuckerGenDataLabel.csv')
