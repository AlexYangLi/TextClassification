# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: metrics.py

@time: 2019/2/8 10:24

@desc:

"""

import numpy as np
from sklearn.metrics import accuracy_score


def eval_acc(y_true, y_pred):
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)

    acc = accuracy_score(y_true, y_pred)
    return acc