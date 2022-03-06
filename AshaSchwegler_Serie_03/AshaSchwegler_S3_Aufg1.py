# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 16:19:23 2022

@author: ashas
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 15:37:36 2022

@author: ashas
"""

import sympy as sp
import numpy as np

x1, x2 = sp.symbols ('x1 x2')
Delta1, Delta2 = sp.symbols ('Delta1 Delta2')

f1 = 20 - 18*x1 - 2*x2**2
f2 = -4*x2 * (x1-x2**2)

g = sp.Matrix([(f1),(f2)])

A = g.subs([(x1,1.1),(x2,0.9)])
f_x0 = A.evalf()




X = sp.Matrix([x1, x2])

Df = g.jacobian(X)
Df0 = Df.subs([(x1,1.1),(x2,0.9)])

Delt = sp.Matrix([Delta1, Delta2])

X_one = sp.linsolve(Df0*Delt + f_x0, Delta1, Delta2)




