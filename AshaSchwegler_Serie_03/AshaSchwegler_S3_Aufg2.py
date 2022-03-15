# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 21:25:50 2022

@author: Asha
"""


import sympy as sp
import numpy as np
import matplotlib.pyplot


x, y = sp.symbols('x y')

f1 = (x**2/186**2) - (y**2/300**2 - 186**2)-1
f2 = ((y-500)**2/279**2) - ((x-300)**2/(500**2-279**2)-1)

#Plot zeigt nicht 4 Schnittpunkte an

p1 = sp.plot_implicit(sp.Eq(f1, 0), (x, -2000, 2000), (y, -2000, 2000))
p2 = sp.plot_implicit(sp.Eq(f2, 0), (x, -2000, 2000), (y, -2000, 2000))

p1.append(p2[0])
p1.show()


X = sp.Matrix([x,y])
f = sp.Matrix([f1,f2])

Df = f.jacobian(X)

print("f:\n",f,"\n")
print("Df:\n",Df,"\n")

# b
f = sp.lambdify([([x],[y])], f, "numpy")
Df = sp.lambdify([([x],[y])], Df, "numpy")

def newt(x, nmax, tol):
    n=0
    while n<nmax and tol<np.linalg.norm(f(x)):
        n+=1
        delta = np.linalg.solve(Df(x), -f(x))        
        x = x + delta        
    return x
        
nmax = 100
x0 = np.array([[-600],[1300]])
print(newt(x0, nmax, 1e-5))