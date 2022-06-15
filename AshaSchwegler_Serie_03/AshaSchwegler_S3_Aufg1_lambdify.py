# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 10:02:56 2022

@author: ashas
"""



import sympy as sp
import numpy as np

x1, x2 = sp.symbols ('x1 x2')


f1 = 20 - 18*x1 - 2*x2**2
f2 = -4*x2 * (x1-x2**2)

g = sp.Matrix([(f1),(f2)])
func = sp.lambdify([x1,x2],g,"numpy")
 
X = sp.Matrix([x1, x2])
Df = g.jacobian(X)

x0 = sp.Matrix([[1.1],[0.9]],dtype='float')


f_x0 = func(1.1,0.9)
print('f_x0 = ',f_x0, '\n')

func_df = sp.lambdify([x1,x2],Df,"numpy")
df0_lambd = func_df(1.1,0.9)
print('df0_lambd = ',df0_lambd,'\n' )


delt0 = np.linalg.solve(df0_lambd,-f_x0,)
print('delt0 = ',delt0,'\n')


# def newton_verfahren(X,A):



x_one = x0 + delt0
print('x_one = ', x_one, '\n')

f_x1 = func(0.996,1.026)
print('f_x1 = ',f_x1, '\n')

func_df1 = sp.lambdify([x1,x2],Df,"numpy")
df1_lambd = func_df(0.996,1.026)
print('df1_lambd = ',df1_lambd,'\n' )

delt1 = np.linalg.solve(df1_lambd,-f_x1,)
print('delt1 = ',delt1,'\n')

normx1 = np.linalg.norm(f_x1)
print('Norm: f(x_1) = ',normx1,'\n')
x1x0 = np.array([[x_one - x0]],dtype='float')
normx1x0= np.linalg.norm(x1x0)
print('Norm: x_1 - x_0 = ',normx1x0,'\n')

x_two = x0 + delt1
print('x_two = ',x_two,'\n')

f_x2 = func(1.1039,0.875)
print('f_x2 = ',f_x2, '\n')

normx2 = np.linalg.norm(f_x2)
print('Norm: f(x_2) = ',normx2,'\n')
x2x1 = np.array([[x_two - x_one]],dtype='float')
normx2x1= np.linalg.norm(x2x1)
print('Norm: x_2 - x_1 = ',normx1x0,'\n')

'''
------------------------------------------------------------------
'''

# Abbruchkriterium a): Abbruch nach einer bestimmten Anzahl Iterationen
def is_finished_max_iterations(f, x, n_max):
    return x.shape[0] - 1 >= n_max

def is_finished(f, x):
    return is_finished_max_iterations(f, x, 3)


x = sp.Matrix([x1, x2])
x0 = np.array([1.1,0.9])  # Startwert

# Sympy-Funktionen kompatibel mit Numpy machen
f_lambda = sp.lambdify([(x1, x2)], g, "numpy")
Df_lambda = sp.lambdify([(x1, x2)], Df, "numpy")

x_approx = np.empty(shape=(0, 2), dtype=np.float64)  # Array mit Lösungsvektoren x0 bis xn
x_approx = np.append(x_approx, [x0], axis=0)  # Start-Vektor in Array einfügen
print('x({}) = {}\n'.format(0, x0))


def newton_verfahren_lin(f_lambda, x_approx):

    while not is_finished(f_lambda, x_approx):
        i = x_approx.shape[0] - 1
        print('ITERATION ' + str(i + 1))
        print('-------------------------------------')
    
        x_n = x_approx[-1]  # x(n) (letzter berechneter Wert)
    
        print('𝛅({}) ist die Lösung des LGS Df(x({})) * 𝛅({}) = -1 * f(x({}))'.format(i, i, i, i))
        print('Df(x({})) = \n{},\nf(x({})) = \n{}'.format(i, Df_lambda(x_n), i, f_lambda(x_n)))
        
        [Q, R] = np.linalg.qr(Df_lambda(x_n))
        delta = np.linalg.solve(R, -Q.T @ f_lambda(x_n)).flatten()  # 𝛅(n) aus Df(x(n)) * 𝛅(n) = -1 * f(x(n))
        print('𝛅({}) = \n{}\n'.format(i, delta))
    
        print('x({}) = x({}) + 𝛅({})'.format(i+1, i, i))
        x_next = x_n + delta.reshape(x0.shape[0], )                  # x(n+1) = x(n) + 𝛅(n)
        print('x({}) = {}'.format(i+1, x_next))
        print('‖f(x({}))‖₂ = {}'.format(i+1, np.linalg.norm(f_lambda(x_next), 2)))
        print('‖x({}) - x({})‖₂ = {}\n'.format(i + 1, i, np.linalg.norm(x_next - x_n, 2)))
    
        x_approx = np.append(x_approx, [x_next], axis=0)
    return x_approx

print(newton_verfahren_lin(f_lambda, x_approx))

