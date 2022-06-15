# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 21:30:10 2022

@author: kimla
"""

import numpy as np

def sum_rechteck(f, a, b, n):
    h = (b-a)/n
    res = 0.
    for i in range(n):
        #print('f',i,' =', f(a + (i+0.5)*h))
        res = res + f(a + (i+0.5)*h)
    return h * res

def sum_trapez(f, a, b, n):
    h = (b-a)/n
    res = 0.5*(f(a) + f(b))
    for i in range(1,n):
        res = res + f(a + i*h)
    return h * res

def sum_simpson(f, a, b, n):
    h = (b-a)/n
    res = 0.
    for i in range(n):
        res = res + f(a + (i+0.5)*h)
    res = 2*res
    for i in range(1,n):
        res = res + f(a+i*h)
    res = 2*res
    res = res + f(a) + f(b)
    return res*h/6.


def f(x):
    return np.log(x**2)

orig = np.log(16) - 2

#%% Rechteck
Rf = sum_rechteck(f,1,2,92)
print('Rf =', Rf)
print('Rf err=', abs(orig - Rf))

#%% Trapez
Tf = sum_trapez(f,1,2,130)
print('Tf =', Tf)
print('Tf err=', abs(orig - Tf))

#%% Simpson
Sf = sum_simpson(f,1,2,5)
print('Sf =', Sf)
print('Sf err=', abs(orig - Sf))