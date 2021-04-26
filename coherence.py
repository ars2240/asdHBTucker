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
# fname = 'cancerHBTCV3KB10_L3_tpl10_{i}_IndepTrees_Cartesian_Genes_weighted.5x.mat'
# fname = 'cancerHBTCV3KB10_L2_tpl10_{i}_IndepTrees_CP_Genes.mat'
# fname = 'cancerHBTCV3KB10_L3_tpl10_{i}_hLDA.mat'
# fname = 'cancerHBTCV3KB10_L2_tpl10_{i}_PAM_Level_Pwy.mat'
fname = 'cancerHBTCV3KB10_L2_tpl10_{i}_Level_PAM.mat'
# fname = 'cancerHBTCV3KB10_L2_tpl0.1_{i}_Cartesian_IndepTrees.mat'
# fname = 'cancer_tensorlyCP_nonNeg_200_2000_200_2.csv'
coh_meas = 'u_mass'  # coherence measure
counts = 'cancerSparseND4.csv'  # count file name
indF = 'cancerCVInd'  # index file name

print('Mean Train Coherence\tMean Test Coherence\tUnique %')
for d in [2]:
    if 'PAM' in fname:
        if (d == 1 and 'pwy' not in fname.lower()) or (d == 2 and 'pwy' in fname.lower()):
            tl = [None, [0], list(range(1, 11)), list(range(11, 21))]
        else:
            tl = [None, list(range(10)), list(range(10, 20)), list(range(20, 30))]
    elif 'IndepTrees' in fname or 'hLDA' in fname:
        tl = [None, 0, 1, 2]
    else:
        tl = [None]
    if 'L2' in fname:
        tl = tl[:-1]
    for t in tl:
        coh_tr, coh_te, nu = coherence(fname, counts, indF, fmin=200, fmax=2000, coh_meas=coh_meas, dim=d,
                                       topics=t, sp=True)

        print('{0}\t{1}\t{2}'.format(np.mean(coh_tr), np.mean(coh_te), np.mean(nu)))



