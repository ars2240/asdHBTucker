# file_manip.py
#
# Author: Adam Sandler
# Date: 8/16/21
#
# File manipulation functions
#
#
# Dependencies:
#   Packages: gensim, numpy, pandas

import math
import numpy as np
import os
import pandas as pd
import pickle
import scipy.io
import sparse


def load_csv(f):
    if os.path.exists(f):
        try:
            d = pd.read_csv(f, header=None, dtype=float)
            d2 = d.iloc[0, 0]
        except ValueError:
            d, d2 = [], float('nan')
        if math.isnan(d2):
            d = pd.read_csv(f, header=0, index_col=0, dtype={0: str})
        return np.array(d)
    else:
        raise Exception('Cannot find file: ' + f)


def load_file(f, s=''):
    if '.pik' in f:
        with open(f, "rb") as f:
            d = pickle.load(f)[0]
    elif '.mat' in f:
        mdict = scipy.io.loadmat(f)  # import dataset from matlab
        d = mdict.get(s)
    else:
        d = load_csv(f)
    return d


# make sure matrix is sparse or not
def impose(x, sp):
    return x.todense() if sp else x


# check if folder exists; if not, create it
def check_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


