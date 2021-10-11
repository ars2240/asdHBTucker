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

# fname = 'cancerHBTCV3KB10_L2_tpl10_{i}_IndepTrees_Cartesian_Genes_weighted48x_48x_cohmass.mat'
fname = 'r8HBTCV3KB10_L2_tpl10_{i}_IndepTrees_Cartesian_Genes_t1000_cohmass.mat'
# fname = 'r8HLDACV3KB10_L2_tpl10_{i}_IndepTrees_Cartesian_Genes_coh.mat'
# fname = 'asdHBTCV3KB10_L2_tpl10_{i}_IndepTrees_Cartesian_Genes.mat'
# fname = 'cancerHBTCV3KB10_L2_tpl10_{i}_IndepTrees_CP_Genes_coh.mat'
# fname = 'cancerHBTCV3KB10_L2_tpl10_{i}_PAM_Cartesian_Genes.mat'
# fname = 'asdHLDACV3KB10_L2_tpl10_{i}_IndepTrees_Cartesian_Pwy.mat'
# fname = 'cancerHBTCV3KB10_L2_tpl10_{i}_hLDA.mat'
# fname = 'cancerHBTCV3KB10_L3_tpl10_{i}_PAM_Level_Pwy.mat'
# fname = 'cancerHBTCV3KB10_L3_tpl10_{i}_Level_PAM.mat'
# fname = 'cancerHBTCV3KB10_L3_tpl0.1_{i}_Cartesian_IndepTrees.mat'
# fname = 'r8_tensorlyCP_nonNeg_200_2000_200_2.csv'
meas = ['uci', 'umass']  # coherence measures
# counts = 'cancerSparseND4.csv'  # count file name
# counts = 'asdSparseND.csv'
counts = 'r8_sparse.csv'
# indF = 'cancerCVInd'  # index file name
# indF = 'asdCVInd'
indF = 'r8CVInd'

for coh_meas in meas:
    print('Mean Train Coh\tMean Test Coh\tUnique %')
    for d in [1, 2]:
        tl = get_tl(fname, d)
        for t in tl:
            coh_tr, coh_te, nu = coherence(fname, counts, indF, fmin=200, fmax=2000, fmin2=100, coh_meas=coh_meas,
                                           dim=d, topics=t, sp=True, rmz=False)

            print('{0}\t{1}\t{2}'.format(np.mean(coh_tr), np.mean(coh_te), np.mean(nu)))



