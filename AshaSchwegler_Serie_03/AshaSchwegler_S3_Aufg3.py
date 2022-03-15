# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 19:11:58 2022

@author: Asha
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot


x1, x2, x3 = sp.symbols('x1 x2 x3')

f1 = x1 + x2**2 - x3**2 - 13
f2 = sp.log(x2/4) + sp.exp(0.5*x3 - 1) - 1
f3 = (x2 - 3)**2 - x3**3 +7




X = sp.Matrix([x1, x2, x3])
f = sp.Matrix([f1,f2,f3])

Df = f.jacobian(X)

print("f:\n",f,"\n")
print("Df:\n",Df,"\n")

# b
f = sp.lambdify([([x1],[x2],[x3])], f, "numpy")
Df = sp.lambdify([([x1],[x2],[x3])], Df, "numpy")

def newt(x, nmax, tol):
    n=0
    while n<nmax and tol<np.linalg.norm(f(x)):
        n+=1
        delta = np.linalg.solve(Df(x), -f(x))        
        x = x + delta        
    return x
        
nmax = 100
x0 = np.array([[1.5],[3],[2.5]],dtype='float')
print(newt(x0, nmax, 1e-5))