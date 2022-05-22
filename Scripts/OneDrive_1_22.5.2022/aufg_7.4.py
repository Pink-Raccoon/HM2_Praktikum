#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 06:50:32 2021

@author: miec
"""

import numpy as np

def sum_trapez(f, a, b, n):
    h = (b-a)/n
    res = 0.5*(f(a) + f(b))
    for i in range(1,n):
        res = res + f(a + i*h)
    return h * res

def f(x):
    return np.exp(-x**2)

Tf = sum_trapez(f,0,0.5,46)

print(Tf)