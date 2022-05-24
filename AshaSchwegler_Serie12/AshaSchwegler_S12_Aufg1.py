# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:22:01 2022

@author: ashas
"""

import numpy as np

def f(x,y):
    return x**2+(0.1*y)

def AshaSchwegler_S12_Aufg1(f,a,b,n,y0):
    h  = (b-a)/n
    x = np.arange(a, b + h, step=h, dtype=np.float64)
    y = y = np.full(x.shape[0], 0, dtype=np.float64)
    y[0] = y0
    for i in range(x.shape[0] - 1):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + (h / 2.0), y[i] + (h / 2.0) * k1)
        k3 = f(x[i] + (h / 2.0), y[i] + (h / 2.0) * k2)
        k4 = f(x[i] + h, y[i] + h * k3)
        print('k1 =',k1,'\n')
        print('k2 =',k2,'\n')
        print('k3 =',k3,'\n')
        print('k4 =',k4,'\n')
        

        y[i + 1] = y[i] + h * (1 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return x, y 

a = -1.5
b = 1.5
n = 5
y0 = 0

print(AshaSchwegler_S12_Aufg1(f, a, b, n, y0))




