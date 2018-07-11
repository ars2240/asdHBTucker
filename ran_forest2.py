# ran_forest2.py
#
# Author: Adam Sandler
# Date: 7/6/18
#
# Computes Random Forest Classifier
#
# Uses CV data
#
# Dependencies:
#   Packages: matplotlib, numpy, scipy, sklearn
#   Files: roc_cv
#   Data: asdHBTucker

# load packages
from roc_cv2 import roc
import scipy.io
from sklearn.ensemble import RandomForestClassifier

fname = 'asdHBTuckerCVData'
mdict = scipy.io.loadmat(fname)  # import dataset from matlab


# #############################################################################
# Classification and ROC analysis

print('%6s\t %6s\t %6s\t %6s\t %6s\t %6s' % ('dset', 'N', 'depth', 'mean', 'stdev', 'pval'))

N_v = [20]
d_v = [None]

for N in N_v:
    for d in d_v:
        # Run classifier with cross-validation and plot ROC curves
        classifier = RandomForestClassifier(max_depth=d, n_estimators=N)

        pname = 'ranforest_AUC_' + str(N) + '_' + str(d) + '_' + str(fname)
        mean_auc, std_auc, p_val, mean_auc_tr, std_auc_tr, p_val_tr = roc(classifier, mdict, pname, fselect='MI',
            nfeat=50)

        if d is None:
            print('%6s\t %6d\t %6s\t %0.4f\t %0.4f\t %0.4f' % ('valid', N, 'None', mean_auc, std_auc, p_val))
            print('%6s\t %6d\t %6s\t %0.4f\t %0.4f\t %0.4f' % ('train', N, 'None', mean_auc_tr, std_auc_tr, p_val_tr))
        else:
            print('%6s\t %6d\t %6d\t %0.4f\t %0.4f\t %0.4f' % ('valid', N, d, mean_auc, std_auc, p_val))
            print('%6s\t %6d\t %6d\t %0.4f\t %0.4f\t %0.4f' % ('train', N, d, mean_auc_tr, std_auc_tr, p_val_tr))
