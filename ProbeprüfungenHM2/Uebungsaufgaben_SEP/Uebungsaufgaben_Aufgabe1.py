# -*- coding: utf-8 -*-


import sympy as sy
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,1.6,2], dtype=np.float64)  # Messwerte xi
y = np.array([40,250,800], dtype=np.float64)  # Messwerte yi


def f(x, a):
    return a[0] + a[1]*sy.exp(a[2] * x)


lam0 = np.array([1,2,3], dtype=np.float64)  # Startvektor für Iteration
a = sy.symbols('a:{n:d}'.format(n=lam0.size))

tol = 1e-5     # Fehlertoleranz
max_iter = 0  # Maximale Iterationen

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

    lam = lam + delta
    err_func = np.linalg.norm(g_lambda(lam)) ** 2
    increment = np.linalg.norm(delta)
    k = k + 1
    print('Iteration: ', k)
    print('f(a,b,c)',g,'\n')
    print('Df(a,b,c)',Dg,'\n')
    print('f(1,2,3)', g_lambda(lam),'\n')
    print('Df(1,2,3)', Dg_lambda(lam),'\n')
    print('delta0',delta,'\n')
    print('x^1 = ', lam)
    print('Inkrement = ', increment)
    print('Fehlerfunktional =', err_func)
    print()



t0 = 2.2
tol = 1e-4
increment = tol + 1
def g(t):
    return lam[0] + lam[1] * np.exp(lam[2]*t)

def f(t):
    return g(t)-1600

def g_abl(t):
    return lam[1]*lam[2]* np.exp(lam[2]*t)


t = np.full(x.shape[0]+2,0,dtype=np.float64)
t[0] = t0





for i in range(len(t)-1):
    t[i+1] = t[i] - (f(t[i]))/(g_abl(t[i]))
    print('t{}='.format(i),t[i],'\n')
    #print('g=',g(t)-1600,'\n')
    if f(t[i]) >= tol:
        break
    i+=1
print('t=',t)

    

        
        
        
     

