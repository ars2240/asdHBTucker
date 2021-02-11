# coherence.py
#
# Author: Adam Sandler
# Date: 2/10/21
#
# Computes model coherence
#
#
# Dependencies:
#   Packages: gensim, numpy, pandas

from coherence_fun import coherence
import numpy as np

nTopics = 20  # number of LDA topics
fname = 'cancer_tensorlyCP_nonNeg_0_1000_10_1.csv'  # file header
coh_meas = 'u_mass'  # coherence measure
counts = 'cancerNumber.csv'  # count file name
indF = 'cancerCVInd'  # index file name

coh_tr, coh_te = coherence(fname, counts, indF, fmin=0, fmax=1000, coh_meas=coh_meas)

print("Mean Train Coherence: " + str(np.mean(coh_tr)))
print("Mean Test Coherence: " + str(np.mean(coh_te)))



