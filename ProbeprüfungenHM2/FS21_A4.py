# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 01:10:28 2022

@author: Asha
"""

import numpy as np
import matplotlib.pyplot as plt

a = 0.0
b = np.pi
h = 0.06
n = int((b-a)/h)

def f(x):
    return np.sin(x)


def Tf(f, a, b, n):
    xi = np.array([a + i * ((b - a) / n) for i in range(1, n)], dtype=np.float64)
    h = (b - a) / n
    return h * ((f(a) + f(b)) / 2 + np.sum(f(xi)))


print('TRAPEZ-REGEL')

print('Wert mit Trapezregel       = ' + str(Tf(f, a, b, n)))

print('abs fehler =',abs(Tf(f, a, b, n)-2))

