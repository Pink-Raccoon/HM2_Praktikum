# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 15:37:36 2022

@author: ashas
"""

import sympy as sp

x1, x2, x3 = sp.symbols ('x1 x2 x3')

f1 = x1 + x2**2 - x3**2 - 13
f2 = sp.log(x2/4) + sp.exp(0.5*x3 - 1) - 1
f3 = (x2 - 3)**2 - x3**3 + 7

a = f1.subs([(x1,1.5),(x2,3),(x3,2.5)])
b = f2.subs([(x1,1.5),(x2,3),(x3,2.5)])
c = f3.subs([(x1,1.5),(x2,3),(x3,2.5)])

f = sp.Matrix([a,b,c])
f_x0 = f.evalf()



X = sp.Matrix([x1, x2, x3])
g = sp.Matrix([(f1),(f2),(f3)])
Df = g.jacobian(X)
Df0 = Df.subs([(x1,1.5),(x2,3),(x3,2.5)])

A = sp.Matrix([x1, x2, x3])
A0 = A.subs([(x1,1.5),(x2,3),(x3,2.5)])


g_x = f_x0 + (Df0 * (X-A0))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 


print(g_x.evalf())