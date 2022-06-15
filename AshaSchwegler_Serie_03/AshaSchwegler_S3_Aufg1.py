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

x0 = sp.Matrix([[1.1],[0.9]],dtype='float')

g = sp.Matrix([(f1),(f2)])

A = g.subs([(x1,1.1),(x2,0.9)])
f_x0 = A.evalf()



 
X = sp.Matrix([x1, x2])

Df = g.jacobian(X)
Df0 = Df.subs([(x1,1.1),(x2,0.9)])

Delt = sp.Matrix([Delta1, Delta2])

delt0 = sp.linsolve(Df0*Delt + f_x0, Delta1, Delta2)






arr_delta0 = np.array([[delt0.args[0][0]],[delt0.args[0][1]]])

x_one = x0 + arr_delta0

A_1 = g.subs([(x1,0.996),(x2,1.026)],dtype='float')
f_x1= A_1.evalf()



fx1_matr=np.array([[f_x1]],dtype='float')

normx1 = np.linalg.norm(fx1_matr)
x1x0 = np.array([[x_one - x0]],dtype='float')
normx1x0= np.linalg.norm(x1x0)


Df1 = Df.subs([(x1,0.996),(x2,1.026)])
delt1 = sp.linsolve(Df1*Delt + f_x1, Delta1, Delta2)
arr_delt1 = np.array([[delt1.args[0][0]],[delt1.args[0][1]]])

x_two = x0 + arr_delt1

A_2 = g.subs([(x1,1.104),(x2,0.875)],dtype='float')
f_x2 = A_2.evalf()

fx2_matr = np.array([[f_x2]],dtype='float')

normx2 = np.linalg.norm(fx2_matr)
x2x1 = np.array([[x_two - x_one]],dtype='float')
normx2x1= np.linalg.norm(x2x1)

print(normx1x0)