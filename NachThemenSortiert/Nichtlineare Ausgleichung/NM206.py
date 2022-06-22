# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:40:39 2022

@author: ashas
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math
x = np.array([0,14,28,42,56])
y = np.array([29,2072,15798,25854,28997])

#a 
'''
N(0) = N0, lim von N(t) = G
'''

N0 = 29
G = 30000
c = 0.3049

tol = 1e-5
max_iter = 30
pmax = 10
damping = 1

lam0 = np.array([G,N0,c], dtype=np.float64)
p = sp.symbols('p:{n:d}'.format(n=lam0.size))




def f(t,p):
    return p[0]/(((p[0]-p[1])/p[1])*sp.exp(-p[2]*t)+1) 

g = sp.Matrix([y[k]-f(x[k],p) for k in range(len(x))])
Dg = g.jacobian(p)

g = sp.lambdify([p], g, 'numpy')
Dg = sp.lambdify([p], Dg, 'numpy')

k = 0
lam = np.copy(lam0)
increment = tol + 1
err_func = np.linalg.norm(g(lam)) ** 2

while increment > tol and k <= max_iter:
    # QR-Zerlegung von Dg(lam)
    [Q, R] = np.linalg.qr(Dg(lam))
    
    delta = np.linalg.solve(R, (-Q.T @ g(lam).flatten()))  # Achtung: flatten() braucht es, um aus dem Spaltenvektor delta wieder
    # eine "flachen" Vektor zu machen, da g hier nicht mit Spaltenvektoren als Input umgehen kann
    # hier kommt die Däfmpfung, falls damping = 1
    p = 0
    while damping == 1 and p <= pmax and np.linalg.norm(g(lam + delta / (2 ** p))) ** 2 >= np.linalg.norm(g(lam)) ** 2:
        p += 1

    if p > pmax:
        p = 0

    # Update des Vektors Lambda
    lam = lam + delta / (2 ** p)
    err_func = np.linalg.norm(g(lam)) ** 2
    #xopt = scipy.optimize.fmin(err_func, lam0)
    increment = np.linalg.norm(delta)
    k = k + 1
    print('Iteration: ', k)
    print('lambda = ', lam)
    print('Inkrement = ', increment)
    print('Fehlerfunktional =', err_func)
    print()

t = sp.symbols('t')
F = f(t, lam)
F = sp.lambdify([t],F,'numpy')
t = np.linspace(-5, 70,75)



plt.figure(1)
plt.grid()
plt.title('Nicht-lineare Ausgleichrechnung COVID-19')
plt.scatter(x,y,marker='x',label='Messdaten')
plt.plot(t, F(t), color = 'black', label='gedämptes Newton Verfahren')
plt.legend()
plt.show()










