# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 23:10:35 2021

HM2, Bsp. 6.10, Manuell

@author: roor / knaa /miec
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 1, 2, 3, 4])
y = np.array([3, 1, 0.5, 0.2, 0.05])

def fp(lamda, x):
    return lamda[0]*np.exp(lamda[1]*x)

def fp_a(lamda, x):
    return np.exp(lamda[1]*x)

def fp_b(lamda, x):
    return lamda[0]*np.exp(lamda[1]*x)*x

def g(lamda):
    return np.array([y-fp(lamda, x)]).T

def Dg(lamda):
    return np.array([-fp_a(lamda, x), -fp_b(lamda, x)]).T

def gauss_newton(g, Dg, lam0, tol, nmax):
    n=0
    lam=np.copy(lam0)
    increment = tol+1
    err_func = np.linalg.norm(g(lam),2)**2
    
    while increment > tol and n < nmax:
        [Q,R] = np.linalg.qr(Dg(lam))
        delta = np.linalg.solve(R,-Q.T @ g(lam))
        lam = lam+delta
        err_func = np.linalg.norm(g(lam),2)**2
        increment = np.linalg.norm(delta,2)
        n = n+1
        print('Iteration: ',n)
        print('lambda = ',lam.flatten())
        print('Inkrement = ',increment)
        print('Fehlerfunktional =', err_func)
    return(lam,n)

lamda0 = np.array([[1, -1.5]]).T
tol = 1e-5
nmax = 100
[lamda_solution,n] = gauss_newton(g, Dg, lamda0, tol, nmax)
print(lamda_solution.flatten(),',', n)

xPlot = np.linspace(x[0], x[-1])

plt.figure(1)
plt.clf()
plt.plot(x, y, 'o')
plt.plot(xPlot, fp(lamda_solution, xPlot))
plt.show()

#%%

lamda0 = np.array([[2.0, 2.0]]).T
[lamda_solution,n] = gauss_newton(g, Dg, lamda0, tol, nmax)
print(lamda_solution.flatten(),',', n)