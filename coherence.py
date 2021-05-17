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

from coherence_fun import coherence, get_tl
import numpy as np

nTopics = 20  # number of LDA topics
fname = 'asdHLDACV3KB10_L1_tpl10_{i}_IndepTrees_Cartesian_Pwy.mat'
# fname = 'cancerHBTCV3KB10_L2_tpl10_{i}_IndepTrees_CP_Genes.mat'
# fname = 'cancerHBTCV3KB10_L3_tpl10_{i}_hLDA.mat'
# fname = 'cancerHBTCV3KB10_L2_tpl10_{i}_PAM_Cartesian_Genes.mat'
# fname = 'cancerHBTCV3KB10_L_tpl10_{i}_Level_PAM.mat'
# fname = 'cancerHBTCV3KB10_L2_tpl0.1_{i}_Cartesian_IndepTrees.mat'
# fname = 'cancer_tensorlyCP_nonNeg_200_2000_200_2.csv'
coh_meas = 'u_mass'  # coherence measure
# counts = 'cancerSparseND4.csv'  # count file name
counts = 'asdSparse.csv'
# indF = 'cancerCVInd'  # index file name
indF = 'asdCVInd'

print('Mean Train Coherence\tMean Test Coherence\tUnique %')
for d in [1, 2]:
    tl = get_tl(fname, d)
    for t in tl:
        coh_tr, coh_te, nu = coherence(fname, counts, indF, fmin=200, fmax=2000, coh_meas=coh_meas, dim=d,
                                       topics=t, sp=True)

        print('{0}\t{1}\t{2}'.format(np.mean(coh_tr), np.mean(coh_te), np.mean(nu)))



