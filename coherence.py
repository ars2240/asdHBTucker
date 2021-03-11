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
fname = 'cancerHBTCV3KB10_L2_tpl10_{i}_Pwy_hLDA.mat'
# fname = 'cancerHBTCV3KB10_L2_tpl10_{i}_PAM_Level_Pwy.mat'
# fname = 'cancerHBTCV3KB10_L3_tpl0.1_{i}_Cartesian_IndepTrees.mat'
# fname = 'cancer_tensorlyCP_nonNeg_200_2000_200_1.csv'
coh_meas = 'u_mass'  # coherence measure
counts = 'cancerSparse.csv'  # count file name
indF = 'cancerCVInd'  # index file name

print('Mean Train Coherence\tMean Test Coherence\tUnique %')
for t in [None, 0, 1]:
# for t in [None, [0], list(range(1, 11)), list(range(11, 21))]:
# for t in [None, list(range(10)), list(range(10, 20)), list(range(20, 30))]:
    coh_tr, coh_te, nu = coherence(fname, counts, indF, fmin=200, fmax=2000, coh_meas=coh_meas, dim=1,
                                   topics=t, sp=True)

    print('{0}\t{1}\t{2}'.format(np.mean(coh_tr), np.mean(coh_te), np.mean(nu)))



