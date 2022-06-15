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

x0 = np.array([1.5,3,.5],dtype='float')

k_max = 4

X = sp.Matrix([x1, x2, x3])
f = sp.Matrix([f1,f2,f3])

Df = f.jacobian(X)

print("f:\n",f,"\n")
print("Df:\n",Df,"\n")

# b
def is_finished_max_iterations(f, x, n_max):
    return x.shape[0] - 1 >= n_max

def is_finished_max_residual(f, x, eps):
    if x.shape[0] < 1:
        return False

    return np.linalg.norm(f(x[-1]), 2) <= 1.0 * eps

def is_finished(f, x):
    return is_finished_max_iterations(f, x, 9)      # Abbruchkriterium a): Abbruch nach einer bestimmten Anzahl Iterationen
    # return is_finished_relative_error(f, x, 1e-5)  # Abbruchkriterium b): Abbruch, wenn ‖x(n+1) - x(n)‖₂ ≤ ‖x(n+1)‖₂ * 𝛜
    # return is_finished_absolute_error(f, x, 1e-5)  # Abbruchkriterium c): Abbruch, wenn ‖x(n+1) - x(n)‖₂ ≤ 𝛜
    return is_finished_max_residual(f, x, 10**-5)    # Abbruchkriterium d): Abbruch, wenn ‖f(x(n+1))‖₂ ≤ 𝛜


f_lambda = sp.lambdify([(x1, x2, x3)], f, "numpy")
Df_lambda = sp.lambdify([(x1, x2, x3)], Df, "numpy")
x_approx = np.empty(shape=(0, 3), dtype=np.float64)  # Array mit Lösungsvektoren x0 bis xn
x_approx = np.append(x_approx, [x0], axis=0)  # Start-Vektor in Array einfügen
print('x({}) = {}\n'.format(0, x0))

def gedämpft_newt(f_lambda, x_approx):
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
    
        x_next = x_n + delta.reshape(x0.shape[0], )  # x(n+1) = x(n) + 𝛅(n) (provisorischer Kandidat, falls Dämpfung nichts nützt)
    
        # Finde das minimale k ∈ {0, 1, ..., k_max} für welches 𝛅(n) / 2^k eine verbessernde Lösung ist (vgl. Skript S. 107)
        last_residual = np.linalg.norm(f_lambda(x_n), 2)  # ‖f(x(n))‖₂
        print('Berechne das Residuum der letzten Iteration ‖f(x(n))‖₂ = ' + str(last_residual))
    
        k = 0
        k_actual = 0
        while k <= k_max:
            print('Versuche es mit k = ' + str(k) + ':')
            new_residual = np.linalg.norm(f_lambda(x_n + (delta.reshape(x0.shape[0], ) / (2 ** k))), 2)  # ‖f(x(n) + 𝛅(n) / 2^k)‖₂
            print('Berechne das neue Residuum ‖f(x(n) + 𝛅(n) / 2^k)‖₂ = ' + str(new_residual))
    
            if new_residual < last_residual:
                print('Das neue Residuum ist kleiner, verwende also k = ' + str(k))
    
                delta = delta / (2**k)
                print('𝛅({}) = 𝛅({}) / 2^{} = {}'.format(i, i, k, delta.T))
    
                x_next = x_n + delta.reshape(x0.shape[0], )  # x(n+1) = x(n) + 𝛅(n) / 2^k
                print('x({}) = x({}) + 𝛅({})'.format(i + 1, i, i))
    
                k_actual = k
                break  # Minimales k, für welches das Residuum besser ist wurde gefunden -> abbrechen
            else:
                print('Das neue Residuum ist grösser oder gleich gross, versuche ein anderes k (bzw. k = 0 wenn k_max erreicht ist)')
    
            print()
            k += 1
    
        x_approx = np.append(x_approx, [x_next], axis=0)
    
        print('x({}) = {} (k = {})'.format(x_approx.shape[0] - 1, x_next, k_actual))
        print('‖f(x({}))‖₂ = {}'.format(i + 1, np.linalg.norm(f_lambda(x_next), 2)))
        print('‖x({}) - x({})‖₂ = {}\n'.format(i + 1, i, np.linalg.norm(x_next - x_n, 2)))
    
    return x_approx


def newt(x, nmax, tol):
    n=0
    while n<nmax and tol<np.linalg.norm(f_lambda(x)):
        n+=1
        delta = np.linalg.solve(Df_lambda(x), -f_lambda(x))        
        x = x + delta        
    return x
        
nmax = 100

print(gedämpft_newt(f_lambda, x_approx))



print('my stuff: ',newt(x0, nmax, 1e-5),'\n')