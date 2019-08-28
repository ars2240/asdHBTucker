# generatedDataClass.py
#
# Author: Adam Sandler
# Date: 10/28/19
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
from sklearn.preprocessing import StandardScaler
import random

fname = 'cancerGenNumber10k'  # count file name
indF = 'cancerGenCVInd10k'  # index file name
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

model = SVC(C=C, kernel='linear')  # SVM classifier

m = 0
iters = 0
while m < .15:
    # random data
    rows = random.sample(range(0, len(cts.index)), 4)
    rcts = cts.iloc[rows]
    rlabs = range(0, 4)
    rcts.to_csv('cancerHBTuckerGenDataRoots10k.csv')

    # normalize
    scaler = StandardScaler()
    scaler.fit(cts)
    tcts = scaler.transform(cts)
    trcts = scaler.transform(rcts)

    # model = LogisticRegression(C=C)  # Logistic Regression classifier
    model.fit(trcts, rlabs)

    pd.DataFrame(model.support_vectors_).to_csv('cancerHBTuckerGenDataSVs10k.csv')

    y_hat = model.predict(tcts)

    # randomly flip 10% of the samples
    #rows = random.sample(range(0, len(y_hat)), math.ceil(len(y_hat) / 10))
    #y_hat[rows] = np.random.randint(0, 4, len(rows))

    # get class distribution
    cls = []
    for i in set(y_hat):
        cls.append(list(y_hat).count(i))
    dis = cls / np.sum(cls)
    print(dis)
    m = np.min(dis)

    iters += 1
    pd.DataFrame(y_hat).to_csv('cancerHBTuckerGenDataLabel10k.csv')

print(iters)
