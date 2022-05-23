# -*- coding: utf-8 -*-
"""
Created on Mon May 23 13:45:25 2022

@author: ashas
"""

import numpy as np

s = 4


def runge_kutta_custom(f,x,h,y0,s):
    a = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0.75, 0.5, 0.75, 0]
    ], dtype=np.float64)
    b = np.array([0.1, 0.1, 0.4, 0.4], dtype=np.float64)
    c = np.array([0.75, 0.25, 0.75, 0.5], dtype=np.float64)

    y = np.full(x.shape[0], 0, dtype=np.float64)
    y[0] = y0
    for i in range(x.shape[0] - 1):
        k = np.full(s, 0, dtype=np.float64)