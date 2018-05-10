# logistic_feature_select.py
#
# Author: Adam Sandler
# Date: 5/8/18
#
# Computes logistic regression tests with feature selection
#
# Dependencies:
#   Packages: matplotlib, numpy, scipy, sklearn
#   Data: asdHBTucker

# load packages
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy import interp
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

fname = 'asdHBTucker_gam0.1'
mdict = scipy.io.loadmat(fname)  # import dataset from matlab

# reformat data
phi = mdict.get('phi')
s = phi.shape
X = np.reshape(phi, [s[0], s[1]*s[2]])
y = np.empty((s[0],))
y[::2] = 1
y[1::2] = 0

# split set into training & test sets
X, X_test, y, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

nfeat_v = [10, 25, 50, 75, 100, 200]

# #############################################################################
# Classification and ROC analysis

print('%6s\t %6s\t %6s\t %6s\t %6s' % ('dset', 'nfeat', 'mean', 'stdev', 'pval'))

for nfeat in nfeat_v:
    X_new = SelectKBest(mutual_info_classif, k=nfeat).fit_transform(X, y)

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=10, random_state=12345)
    classifier = LogisticRegression(C=1e15)

    tprs = []
    aucs = []
    aucs_tr = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X_new, y):
        probas_ = classifier.fit(X_new[train], y[train]).predict_proba(X_new[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        # Compute AUC on Training set
        probas_ = classifier.fit(X_new[train], y[train]).predict_proba(X_new[train])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[train], probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        aucs_tr.append(roc_auc)

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right", prop={'size': 6})
    plt.savefig('plots/logreg_AUC_feat_select_' + str(nfeat) + '_' + str(fname) + '.png')

    results = stats.ttest_1samp(aucs, popmean=0.5)
    p_val = results[1]

    results = stats.ttest_1samp(aucs_tr, popmean=0.5)
    p_val_tr = results[1]

    print('%6s\t %6d\t %0.4f\t %0.4f\t %0.4f' % ('valid', nfeat, mean_auc, std_auc, p_val))
    print('%6s\t %6d\t %0.4f\t %0.4f\t %0.4f' % ('train', nfeat, np.mean(aucs_tr), np.std(aucs_tr), p_val_tr))
