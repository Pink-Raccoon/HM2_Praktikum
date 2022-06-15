# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:40:39 2022

@author: ashas
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
x = np.array([0,14,28,42,56])
y = np.array([29,2072,15798,25854,28997])

#a 
'''
N(0) = N0, lim von N(t) = G
'''

plt.figure(1)
plt.grid()
plt.title('Ausgleich')
plt.scatter(x,y,marker='X',label= 'Messpunkte')
# plt.plot(x,f(x,lam_qr),color='r')
# plt.plot(x,val,color='orange', label='Polyfit')
plt.xlabel('t')
plt.ylabel('N')
plt.legend()
plt.show()

#b N0 = 29, Gn=28000

#c 


y0 = 29.0
c = 0.304
G = 30000.0
lam0 = np.array([G,y0,c], dtype='float')  # Startvektor für Iteration
a = sy.symbols('a:{n:d}'.format(n=lam0.size))

tol = 1e-5     # Fehlertoleranz
max_iter = 30  # Maximale Iterationen
pmax = 10      # Maximale Dämpfung
t = np.linspace(-5,70)
def f(t,a):
    return a[0]/((a[0]-a[1])/(a[1])*np.exp(-a[2]*t)+1)
    
g = sy.Matrix([y[k] - f(x[k], a) for k in range(x.shape[0])])  # Fehlerfunktion für alle (xi, yi)
Dg = g.jacobian(a)

g_lambda = sy.lambdify([a], g, 'numpy')
Dg_lambda = sy.lambdify([a], Dg, 'numpy')

k = 0
lam = np.copy(lam0)
increment = tol + 1
err_func = np.linalg.norm(g_lambda(lam)) ** 2

while increment > tol and k <= max_iter:
    # QR-Zerlegung von Dg(lam)
    [Q, R] = np.linalg.qr(Dg_lambda(lam))
    delta = np.linalg.solve(R, -Q.T @ g_lambda(lam)).flatten()
    # Achtung: flatten() braucht es, um aus dem Spaltenvektor delta wieder
    # eine "flachen" Vektor zu machen, da g hier nicht mit Spaltenvektoren als Input umgehen kann

    # Dämpfung
    p = 0
    while p <= pmax and np.linalg.norm(g_lambda(lam + delta / (2 ** p))) ** 2 >= np.linalg.norm(g_lambda(lam)) ** 2:
        p += 1

    if p > pmax:
        p = 0

    # Update des Vektors Lambda
    lam = lam + delta / (2 ** p)

    err_func = np.linalg.norm(g_lambda(lam)) ** 2
    increment = np.linalg.norm(delta)
    k = k + 1
    print('Iteration: ', k)
    print('lambda = ', lam)
    print('Inkrement = ', increment)
    print('Fehlerfunktional =', err_func)
    print()





