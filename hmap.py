# hmap.py
#
# Author: Adam Sandler
# Date: 5/15/20
#
# Creates & saves heatmaps
#
#
# Dependencies:
#   Packages: matplotlib, numpy

# load packages
import matplotlib.pyplot as plt
import numpy as np
import os


# check if folder exists; if not, create it
def check_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def heatmap(x, y, tail=''):
    check_folder('plots')

    # change data type
    x = np.array(x)
    y = np.array(y)

    for i in np.unique(y):
        x2 = x[y == i, :]
        fig, ax = plt.subplots()
        plt.imshow(x2, cmap='gray', aspect='auto')
        # We want to show all ticks...
        ax.set_xticks(np.arange(x2.shape[1]))
        ax.set_xticklabels(['%.2e' % j for j in np.mean(x2, axis=1)], fontsize=3)
        ax.set_title('Class {0}, {1} Patients, {2} Topics'.format(i, x2.shape[0], x2.shape[1]))
        fig.tight_layout()
        plt.savefig(os.path.join('plots', 'heatmap{0}_{1}.pdf'.format(tail, i)), dpi=300)
        plt.close()
