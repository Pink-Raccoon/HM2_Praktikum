# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:01:19 2022

@author: ashas
"""

import numpy as np
import matplotlib.pyplot as plt

u = 2000
m0 = 10000
q = 100
g = 9.8
t = 30
h = 10
a=0
n=3


def f(t):
    v = u*np.log((m0)/(m0-q*t))-g*t
    return v

def Tf(f, a, b, n):
    xi = np.array([a + i * ((b - a) / n) for i in range(1, n)], dtype=np.float64)
    h = (b - a) / n
    return h * ((f(a) + f(b)) / 2 + np.sum(f(xi)))

print('Die Höhe beträgt: ',Tf(f, a, t, n))

