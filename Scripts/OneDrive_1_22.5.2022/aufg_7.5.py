#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 07:08:38 2021

@author: miec
"""

import numpy as np

def sum_rechteck(f, a, b, n):
    h = (b-a)/n
    res = 0.
    for i in range(n):
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

def gauss_3(f,a,b):
    m = (a+b)/2
    r = (b-a)/2
    w0 = 8/9
    w1 = 5/9
    r1 = np.sqrt(0.6)*r    
    return r*(w1*f(m - r1) + w0*f(m) + w1*f(m + r1))


def f(x):
    return np.exp(-x**2)

Rf = sum_rechteck(f,0,0.5,3)
Tf = sum_trapez(f,0,0.5,3)
Sf = sum_simpson(f,0,0.5,3)
G3f = gauss_3(f,0,0.5)

print('Rf =', Rf)
print('Tf =', Tf)
print('Sf =', Sf)
print('G3f =', G3f)

I = sum_simpson(f,0,0.5,1000)
print('|I-Rf| =', abs(I-Rf))
print('|I-Tf| =', abs(I-Tf))
print('|I-Sf| =', abs(I-Sf))
print('|I-G3f| =', abs(I-G3f))