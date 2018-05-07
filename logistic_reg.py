# logistic_reg.py
#
# Author: Adam Sandler
# Date: 5/6/18
#
# Computes logistic regression tests with regularization parameter
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
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

fname = 'asdHBTucker_gam0.5'
mdict = scipy.io.loadmat(fname)  # import dataset from matlab

# reformat data
phi = mdict.get('phi')
s = phi.shape
X = np.reshape(phi, [s[0], s[1]*s[2]])
y = np.empty((s[0],))
y[::2] = 1
y[1::2] = 0

# split set into training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

lam_v = [1E-9, 1E-6, 1E-3, 1, 1E3, 1E6, 1E9]

# #############################################################################
# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves
for lam in lam_v:
    cv = StratifiedKFold(n_splits=5)
    classifier = LogisticRegression(C=1 / lam)

    tprs = []
    aucs = []
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
    plt.savefig('plots/logreg_AUC_' + str(lam) + '_' + str(fname) + '.png')

    results = stats.ttest_1samp(aucs, popmean=0.5)
    p_val = results[1]

    print(r'Lambda = %6.3e, Mean ROC (AUC = %0.4f $\pm$ %0.4f), p-value = %0.4f' % (lam, mean_auc, std_auc, p_val))
