#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 06:55:14 2021

@author: miec
"""

import numpy as np
import sympy as sp

x1, x2 = sp.symbols('x1 x2')

x = sp.Matrix([x1,x2])

f1 = 2*x1+4*x2
f2 = 4*x1+8*x2**3
f = sp.Matrix([f1,f2])

Df = f.jacobian(x)

f = sp.lambdify([[[x1],[x2]]], f, 'numpy')
Df = sp.lambdify([[[x1],[x2]]], Df, 'numpy')


def newton(x):
    for i in np.arange(1,6):
        delta = np.linalg.solve(Df(x),-f(x))
        x = x + delta
        print('x'+str(i)+' =', x.reshape(2))
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