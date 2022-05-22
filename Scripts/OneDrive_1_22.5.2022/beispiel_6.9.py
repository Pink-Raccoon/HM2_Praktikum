# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 15:17:19 2021

HM2, Bsp. 6.9

@author: knaa / miec
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

a, b, X = sp.symbols('a b x')

x = np.array([0, 1, 2, 3, 4])
y = np.array([3, 1, 0.5, 0.2, 0.05])

f = a*sp.exp(b*X)

E = 0
for i in range(len(x)):
    u = y[i] - f.subs(X,x[i])
    E = E + u*u
    
g1 = sp.diff(E,a)
g2 = sp.diff(E,b)

g = sp.Matrix([g1,g2])
lam = sp.Matrix([a,b])        # Achtung: die Unbekannten sind a und b (nicht x)
Dg  = g.jacobian(lam)

g = sp.lambdify([[[a],[b]]], g, "numpy")
Dg = sp.lambdify([[[a],[b]]], Dg, "numpy")

# Newton's Methode für Systeme
def newton(f, Df, x0, tol):
    n=0
    x=np.copy(x0)
    err = np.linalg.norm(f(x),2)
    
    while err > tol:
        delta = np.linalg.solve(Df(x),-f(x))
        x = x+delta
        err = np.linalg.norm(f(x),2)
        n = n+1
    return(x,n)

# Aufruf
tol = 1e-5
x0 = np.array([[3,-1]]).T
[xn,n] = newton(g, Dg, x0, tol)

lam = np.reshape(xn,(2))
print('lamda = ', lam)

err = E.subs(a,lam[0]).subs(b,lam[1])
print('E(f) =', err)

f = f.subs(a,lam[0]).subs(b,lam[1])
print('f = ', f)
f = sp.lambdify(X, f, "numpy")

#plot
x_fine=np.arange(x[0],x[-1]+0.1,0.1)
plt.plot(x,y,'o')
plt.plot(x_fine,f(x_fine))
plt.xlabel('x'),plt.ylabel('y=f(x)'),plt.legend(['data','f(x)=a*exp(b*x)'])


