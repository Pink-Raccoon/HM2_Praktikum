#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 06:55:14 2021

@author: miec
"""

import numpy as np

def p(k1,k2,k3,r):
    return k1*np.exp(k2*r) + k3*r

def f(x):
    k1 = x[0,0]
    k2 = x[1,0]
    k3 = x[2,0]
    return np.array([[p(k1,k2,k3,1.) - 10.],
                     [p(k1,k2,k3,2.) - 12.],
                     [p(k1,k2,k3,3.) - 15.]])

def Df(x):
    k1 = x[0,0]
    k2 = x[1,0]
    k3 = x[2,0]
    return np.array([[np.exp(   k2),    k1*np.exp(   k2), 1.],
                     [np.exp(2.*k2), 2.*k1*np.exp(2.*k2), 2.],
                     [np.exp(3.*k2), 3.*k1*np.exp(3.*k2), 3.]])

def newton(x, tol, kmax=0):
    i=0
    res = np.linalg.norm(f(x))
    print('||f(x' + str(i) + ')|| =',  res)
    while res > tol:
        delta = np.linalg.solve(Df(x),-f(x))
        k = 0
        while k <= kmax and  np.linalg.norm(f(x + 0.5**k*delta)) > res:
            k = k+1
        if k > kmax:
            print('no k found')
            k = 0
        x = x + 0.5**k*delta
        res = np.linalg.norm(f(x))
        i = i+1
        print('k =', k)
        print('x' + str(i) + ' = \n', x)
        print('||f(x' + str(i) + ')|| =',  res)
    return x
        

x0 = np.array([[10.],[0.1],[-1.]])
tol = 1e-5
kmax = 4
x = newton(x0, tol, kmax)
k1 = x[0,0]
k2 = x[1,0]
k3 = x[2,0]

def g(r):
    return p(k1,k2,k3,r) - 500.

def dg(r):
    return k1*k2*np.exp(k2*r) + k3

def newton1D(r, tol):
    while np.abs(g(r)) > tol:
        r = r - g(r)/dg(r)
    return r
    
r0 = 15
r = newton1D(r0, tol)
print('r =', r)

    

