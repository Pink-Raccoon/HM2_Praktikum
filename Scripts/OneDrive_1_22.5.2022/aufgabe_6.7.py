# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 17:12:05 2021

HM2, Aufg. 6.7

@author: knaa / miec
"""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

x = np.array([1.,2.,3.,4.])
y = np.array([7.1, 7.9, 8.3, 8.8])

# Definition von fp
a, b = sp.symbols('a b')
def fp(x):
    return a*sp.log(x+b)

# Definition von g und Berechnung von Dg
g = sp.Matrix([y[k]-fp(x[k]) for k in range(len(x))])
Dg = g.jacobian(sp.Matrix([a, b]))

# Numerische Funktionen 
g = sp.lambdify([[[a],[b]]], g, "numpy")
Dg = sp.lambdify([[[a],[b]]], Dg, "numpy")

# Gauss-Newton Verfahren
def gauss_newton(g, Dg, lam0, tol):
    k=0
    lam=np.copy(lam0)
    increment = tol+1
    err_func = np.linalg.norm(g(lam),2)**2
    
    while increment > tol:
        [Q,R] = np.linalg.qr(Dg(lam))
        delta = np.linalg.solve(R,-Q.T @ g(lam))
        lam = lam+delta
        err_func = np.linalg.norm(g(lam),2)**2
        increment = np.linalg.norm(delta,2)
        k = k+1
        print('Iteration: ',k)
        print('lam = ',lam.flatten())
        print('Residuum = ',increment)
        print('Fehlerfunktional =', err_func)
    return(lam,k)

# Aufruf
tol = 1e-5
lam0 = np.array([[1,1]]).T
[lam,k] = gauss_newton(g, Dg, lam0, tol)

print('lam_' + str(k) + ' = ' + str(lam.flatten()))

def f(x):
    return lam[0]*np.log(x+lam[1])

x_fine=np.arange(float(x[0]),float(x[-1])+0.1,0.1)
plt.plot(x,y,'o')
plt.plot(x_fine, f(x_fine))
plt.xlabel('x'),plt.ylabel('y=f(x)'),plt.legend(['data','f(x)=a*log(x+b)'])








