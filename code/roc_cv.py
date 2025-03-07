# roc_cv.py
#
# Author: Adam Sandler
# Date: 6/7/18
#
# Computes ROC for each CV, returns plot in /plots/ folder, and
# mean, stDev, and p-value for both train & validation sets
#
# Modified from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
#
# Dependencies:
#   Packages: matplotlib, numpy, scipy, sklearn
#   Data: asdHBTucker

import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from scipy import stats
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold


def roc(classifier, X, y, pname, splits=10, random_state=12345):

    cv = StratifiedKFold(n_splits=splits, random_state=random_state)

    tprs = []
    aucs = []
    aucs_tr = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        # Compute AUC on Training set
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[train])
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
    plt.savefig('plots/' + pname + '.png')

    results = stats.ttest_1samp(aucs, popmean=0.5)
    p_val = results[1]

    results = stats.ttest_1samp(aucs_tr, popmean=0.5)
    p_val_tr = results[1]

    return mean_auc, std_auc, p_val, np.mean(aucs_tr), np.std(aucs_tr), p_val_tr
