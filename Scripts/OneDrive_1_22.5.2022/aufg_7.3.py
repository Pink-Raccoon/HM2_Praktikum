#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 09:15:14 2021

@author: miec
"""

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


def f(x):
    return 1/x

a = 2
b = 4
n = 4

rf = sum_rechteck(f,a,b,n)
tf = sum_trapez(f,a,b,n)
sf = sum_simpson(f,a,b,n)

print('rf =', rf)
print('tf =', tf)
print('sf =', sf)
print('check =', sf - (tf + 2*rf)/3.)

