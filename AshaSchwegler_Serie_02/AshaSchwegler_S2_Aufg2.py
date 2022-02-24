# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 20:27:14 2022

@author: Asha
"""

import sympy as sp


x, y = sp.symbols ('x y')


#1a
f1 = 5*x*y
f2 = x**2*y**2 + x + 2*y


f = sp.Matrix([f1,f2])


X = sp.Matrix([x,y])
Df = f.jacobian(X)
Df0 = Df.subs([(x,1),(y,2)])

print(Df0)


#1b

x1,x2,x3 = sp.symbols ('x1 x2 x3')

func1 = sp.log(x1**2+x2**2) + x3**2
func2 = sp.exp(x2**2+x3**2) + x1**2
func3 = 1/(x3**2+x1**2) + x2**2

func = sp.Matrix ([func1,func2,func3])



A = sp.Matrix([x1,x2,x3])
Df_a = func.jacobian(A)
Df1 = Df_a.subs([(x1,1),(x2,2),(x3,3)])

print('')

print(Df1)