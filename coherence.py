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

# fname = 'cancerHBTCV3KB10_L2_tpl10_{i}_IndepTrees_Cartesian_Genes_weighted32x_32x_cohmass.mat'
fname = 'r8pHBTCV3KB10_L2_tpl10_{i}_IndepTrees_Cartesian_Genes_t50_cohmass.mat'
# fname = 'r8HLDACV3KB10_L2_tpl10_{i}_IndepTrees_Cartesian_Pwy_coh.mat'
# fname = 'asdHBTCV3KB10_L2_tpl10_{i}_IndepTrees_Cartesian_Genes.mat'
# fname = 'cancerHBTCV3KB10_L2_tpl10_{i}_IndepTrees_CP_Genes_coh.mat'
# fname = 'cancerHBTCV3KB10_L2_tpl10_{i}_PAM_Cartesian_Genes.mat'
# fname = 'asdHLDACV3KB10_L2_tpl10_{i}_IndepTrees_Cartesian_Pwy.mat'
# fname = 'cancerHBTCV3KB10_L2_tpl10_{i}_hLDA.mat'
# fname = 'cancerHBTCV3KB10_L3_tpl10_{i}_PAM_Level_Pwy.mat'
# fname = 'cancerHBTCV3KB10_L3_tpl10_{i}_Level_PAM.mat'
# fname = 'cancerHBTCV3KB10_L3_tpl0.1_{i}_Cartesian_IndepTrees.mat'
# fname = 'r8_tensorlyCP_nonNeg_200_2000_200_{d}.csv'
meas = ['uci', 'umass']  # coherence measures
# counts = 'cancerSparseND4.csv'  # count file name
# counts = 'asdSparseND.csv'
counts = 'r8p_sparse.csv'
# indF = 'cancerCVInd'  # index file name
# indF = 'asdCVInd'
indF = 'r8CVInd'
bad = None
# bad = [180, 194, 234]
# bad = [0, 29, 180, 186, 194, 224, 234, 246, 247]
# bad = [0, 29, 64, 81, 180, 186, 194, 224, 234, 242, 244, 245, 246, 247]
# bad = np.genfromtxt('r8_badwF.csv', delimiter=',')
dims = [1, 2]

for coh_meas in meas:
    print('Mean Train Coh\tMean Test Coh\tUnique %')
    for d in dims:
        tl = get_tl(fname, d)
        for t in tl:
            coh_tr, coh_te, nu = coherence(fname, counts, indF, fmin=200, fmax=2000, fmin2=100, coh_meas=coh_meas,
                                           dim=d, topics=t, sp=True, rmz=False, bad=bad)

            print('{0}\t{1}\t{2}'.format(np.mean(coh_tr), np.mean(coh_te), np.mean(nu)))



