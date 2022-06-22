# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 20:49:47 2022

@author: ashas
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sy

x = np.array([25,35,45,55,65],dtype=np.float64)
y = np.array([47,114,223,81,20],dtype=np.float64)

A0 = 10.0**8
f0 = 50.0
c0 = 600.0

def f(x,a):
    return (a[0]/(((x**2-a[1]**2)**2)+a[2]**2))
    
    

lam0 = np.array([A0,f0,c0],dtype=np.float64)

    
a = sy.symbols('a:{n:d}'.format(n=lam0.size))

tol = 1e-3    # Fehlertoleranz
max_iter = 30  # Maximale Iterationen

g = sy.Matrix([y[k] - f(x[k], a) for k in range(x.shape[0])])  # Fehlerfunktion für alle (xi, yi)
Dg = g.jacobian(a)

g_lambda = sy.lambdify([a], g, 'numpy')
Dg_lambda = sy.lambdify([a], Dg, 'numpy')




k = 0
lam = np.copy(lam0)
increment = abs(lam[k+1]-lam[k])
err_func = np.linalg.norm(g_lambda(lam)) ** 2
while increment > tol and k <= max_iter:
    # QR-Zerlegung von Dg(lam)
    [Q, R] = np.linalg.qr(Dg_lambda(lam))
    delta = np.linalg.solve(R, -Q.T @ g_lambda(lam)).flatten()
    # Achtung: flatten() braucht es, um aus dem Spaltenvektor delta wieder
    # eine "flachen" Vektor zu machen, da g hier nicht mit Spaltenvektoren als Input umgehen kann

    lam = lam + delta
    err_func = np.linalg.norm(g_lambda(lam)) ** 2
    increment = np.linalg.norm(delta)
    k = k + 1
    print('Iteration: ', k)
    print('lambda = ', lam)
    print('Inkrement = ', increment)
    print('Fehlerfunktional =', err_func)
    print()


pmax = 5
damping = 1

while increment > tol and k <= max_iter:
    # QR-Zerlegung von Dg(lam)
    [Q, R] = np.linalg.qr(Dg_lambda(lam))
    
    delta = np.linalg.solve(R, (-Q.T @ g_lambda(lam).flatten()))  # Achtung: flatten() braucht es, um aus dem Spaltenvektor delta wieder
    # eine "flachen" Vektor zu machen, da g hier nicht mit Spaltenvektoren als Input umgehen kann
    # hier kommt die Däfmpfung, falls damping = 1
    p = 0
    while damping == 1 and p <= pmax and np.linalg.norm(g_lambda(lam + delta / (2 ** p))) ** 2 >= np.linalg.norm(g_lambda(lam)) ** 2:
        p += 1

    if p > pmax:
        p = 0

    # Update des Vektors Lambda
    lam = lam + delta / (2 ** p)
    err_func = np.linalg.norm(g_lambda(lam)) ** 2
    #xopt = scipy.optimize.fmin(err_func, lam0)
    increment = np.linalg.norm(delta)
    k = k + 1
    print('Iteration: ', k)
    print('lambda = ', lam)
    print('Inkrement = ', increment)
    print('Fehlerfunktional =', err_func)
    print()
    


t = sy.symbols('t')
F = f(t, lam)
F = sy.lambdify([t],F,'numpy')
t = np.linspace(20, 70,90)





plt.figure(1)
plt.grid()
plt.scatter(x,y,marker='x',label='Messdaten')

plt.plot(t,F(t), label= 'gedämpft')
plt.legend()
plt.show()