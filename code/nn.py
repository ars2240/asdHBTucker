# gbm.py
#
# Author: Adam Sandler
# Date: 6/9/18
#
# Computes Nueral Network classifier
#
# Dependencies:
#   Packages: matplotlib, numpy, scipy, sklearn, xgboost torch
#   Files: roc_cv_nn
#   Data: asdHBTucker

# load packages
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
from sklearn.model_selection import train_test_split
from roc_cv_nn import roc

fname = 'asdHBTucker_gam0.1'
mdict = sio.loadmat(fname)  # import dataset from matlab

# reformat data
phi = mdict.get('phi')
s = phi.shape
X = np.reshape(phi, [s[0], s[1]*s[2]])
y = np.empty((s[0],))
y[::2] = 1
y[1::2] = 0

# split set into training & test sets
X, _, y, _ = train_test_split(X, y, test_size=0.3, random_state=12345)

# re-format data
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()

# #############################################################################
# Classification and ROC analysis

print('%6s\t %6s\t %6s\t %6s' % ('dset', 'mean', 'stdev', 'pval'))


# Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(s[1]*s[2], 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x):
        x = x.view(-1, s[1]*s[2])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


pname = 'nn_AUC__' + str(fname)
mean_auc, std_auc, p_val, mean_auc_tr, std_auc_tr, p_val_tr = roc(Net(), X, y, pname)

print('%6s\t %0.4f\t %0.4f\t %0.4f' % ('valid', mean_auc, std_auc, p_val))
print('%6s\t %0.4f\t %0.4f\t %0.4f' % ('train', mean_auc_tr, std_auc_tr, p_val_tr))
