# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 16:54:35 2022

@author: ashas
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

x = np.array([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0])
y = np.array([39.55,46.55,50.13,51.75,55.25,56.79,56.78,59.13,57.76,59.39,60.08])

plt.figure(1)
plt.grid()
plt.scatter(x,y,marker='x',label='Messdaten')
plt.legend()
plt.show()

A = 0
Q = 60
tau = 2.0 / 5


tol = 1e-7
max_iter = 30
pmax = 5
damping = 1


lam0 = np.array([A,Q,tau], dtype=np.float64)
p = sp.symbols('p:{n:d}'.format(n=lam0.size))


def U(x,p):
    return (p[0]+(p[1]-p[0]) * (1 - sp.exp(-x/p[2])))

g = sp.Matrix([y[k]-U(x[k],p) for k in range(len(x))])
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









