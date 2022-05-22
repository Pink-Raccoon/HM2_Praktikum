#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 06:55:14 2021

@author: miec
"""

import numpy as np

def f(x):
    x1 = x[0,0]
    x2 = x[1,0]
    return np.array([[2.*x1 + 4.*x2], 
                     [4.*x1 + 8.*x2**3]])

def Df(x):
    x2 = x[1,0]
    return np.array([[2., 4.],
                     [4., 24.*x2**2]])

def newton(x):
    for i in np.arange(1,6):
        delta = np.linalg.solve(Df(x),-f(x))
        x = x + delta
        print('x'+str(i),'=', x.reshape(2))
    return x
        
#%%
x0 = np.array([[4],[2]])
print('x0 =', x0.reshape(2))
newton(x0)

#%%
x0= np.array([[-4],[-2]])
print('x0 =', x0.reshape(2))
newton(x0)

#%%
x0= np.array([[1],[0.4]])
print('x0 =', x0.reshape(2))
newton(x0)
