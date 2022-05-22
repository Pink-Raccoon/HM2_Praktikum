# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:55:17 2022

@author: ashas
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from AshaSchwegler_S11_Aufg1 import AshaSchwegler_S11_Aufg1





def f(x, y):
    return x**2/y

def euler(f, x, y_euler, dx, i, maxX):
    m = f(x, y_euler[i])
    y_euler = np.append(y_euler, m*dx + y_euler[i])
    if x+dx >= maxX:
        return y_euler
    return euler(f, x + dx, y_euler, dx, i+1, maxX)

def y_euler(f, x, y_euler, dx, i, maxX):
    m = f(x, y_euler[i])
    y_euler = np.append(y_euler, m*dx + y_euler[i])
    if x+dx >= maxX:
        return y_euler
    return y_euler(f, x + dx, y_euler, dx, i+1, maxX)


def mittelpunktverfahren(f, x, y_mittelpunkt, dx, i, maxX):
    m = f(x, y_mittelpunkt[i])
    mm = f(x + dx/2, y_mittelpunkt[i] + dx/2*m)
    y_mittelpunkt = np.append(y_mittelpunkt, mm*dx + y_mittelpunkt[i])
    if x+dx >= maxX:
        return y_mittelpunkt
    return mittelpunktverfahren(f, x + dx, y_mittelpunkt, dx, i+1, maxX)

def modifiziertenEulerverfahren(f, x, y_modeuler, dx, i, maxX):
    m = f(x, y_modeuler[i])
    mm = f(x + dx, y_modeuler[i] + dx*m)
    y_modeuler = np.append(y_modeuler, dx/2 * (m + mm) + y_modeuler[i])
    if x+dx >= maxX:
        return y_modeuler
    return modifiziertenEulerverfahren(f, x + dx, y_modeuler, dx, i+1, maxX)
   
def AshaSchwegler_S11_Aufg3(f, a, b, n, y0):
    x = np.linspace(a, b, n+1)
    dx = (b-a)/n
    y_euler = euler(f, a, np.array([y0]), dx, 0, b)
    y_mittelpunkt = mittelpunktverfahren(f, a, np.array([y0]), dx, 0, b)
    y_modeuler = modifiziertenEulerverfahren(f, a, np.array([y0]), dx, 0, b)
    return [x, y_euler, y_mittelpunkt, y_modeuler]


(x, y_euler, y_mittelpunkt, y_modeuler) = AshaSchwegler_S11_Aufg3(f, 0, 7, 10, 2)

print("x: ", x)
print()
print("y_euler: ", y_euler)
print()
print("y_mittelpunkt: ", y_mittelpunkt)
print()
print("y_modeuler: ", y_modeuler)


plt.plot(x, y_euler, label='Eulerverfahren')
plt.plot(x, y_mittelpunkt, label='Mittelpunktverfahren')
plt.plot(x, y_modeuler, label='Modifizieres Eulerverfahren')
plt.legend()
AshaSchwegler_S11_Aufg1(f, 0, 7, 2, 15, 0.5, 1)
plt.show()