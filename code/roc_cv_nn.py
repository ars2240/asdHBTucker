# roc_cv_nn.py
#
# Author: Adam Sandler
# Date: 6/9/18
#
# Computes ROC for each CV, returns plot in /plots/ folder, and
# mean, stDev, and p-value for both train & validation sets
# for neural networks
#
# Modified from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
#
# Dependencies:
#   Packages: matplotlib, numpy, scipy, sklearn, torch
#   Data: asdHBTucker

import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from scipy import stats
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import torch
import torch.utils.data as utils_data


# my training algorithm
def train_model(model, train_data, X, y, nb_epochs=10, tol=0.001, batch_size=100, lr=0.001, momentum=0.9, weight_decay=0):
    """
    Trains the model using SGD.

    Inputs:
        model: Neural network model
        nb_epochs: number of epochs (int)
        batch_size: batch size (int)
        lr: learning rate/steplength (float)

    """

    # initialize train loader, optimizer, and loss
    train_loader = utils_data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    loss = torch.nn.BCELoss()

    for e in range(nb_epochs):

        # Training loop!
        for i, data in enumerate(train_loader):
            # get inputs
            inputs, labels = data

            # set to training mode
            model.train()

            # zero gradient
            optimizer.zero_grad()

            # forward pass and compute loss
            ops = model(inputs)
            loss_fn = loss(ops.view(-1), labels)

            # compute gradient and take step in optimizer
            loss_fn.backward()
            optimizer.step()

        model.eval()  # set to evaluation mode

        # evaluate training loss and training accuracy
        ops = model(X)
        train_loss = loss(ops.view(-1), y).item()
        if e > 0:
            dif = (train_loss_o - train_loss)/train_loss
            if abs(dif) < tol:
                lr *= 1/2
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        train_loss_o = train_loss

        #print('Epoch:', e + 1, 'Train Loss:', train_loss, 'Training Accuracy:', train_acc)


# Define Testing Function
def test_probs(net, X):
    """
    Generates test probabilities

    Inputs:
        X: feature data (FloatTensor)
        y: labels (LongTensor)

    """

    # constructs model
    model = net
    loss = torch.nn.CrossEntropyLoss()

    # loads weights
    model.load_state_dict(torch.load('trained_model.pt'))

    # compute loss and accuracy
    ops = model(X)
    return ops.data.view(-1)


def roc(net, X, y, pname, splits=10, random_state=12345):

    # set seed
    np.random.seed(1226)
    torch.manual_seed(1226)

    # Parameters
    epochs = 25
    lr = 0.001
    momentum = 0.9
    batch_size = 128
    weight_decay = 0
    tol = 0.001

    cv = StratifiedKFold(n_splits=splits, random_state=random_state)

    tprs = []
    aucs = []
    aucs_tr = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):

        train_data = utils_data.TensorDataset(X[train], y[train])

        model = net

        # Train model
        train_model(model, train_data, X[train], y[train], nb_epochs=epochs, tol=tol, batch_size=batch_size, lr=lr,
                    momentum=momentum, weight_decay=weight_decay)

        torch.save(model.state_dict(), 'trained_model.pt')

        probas_ = test_probs(net, X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        # Compute AUC on Training set
        probas_ = test_probs(net, X[train])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[train], probas_)
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
