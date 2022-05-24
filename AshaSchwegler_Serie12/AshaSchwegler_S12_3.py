# -*- coding: utf-8 -*-
"""
Created on Mon May 23 23:57:12 2022

@author: Asha
"""

import numpy as np
from AshaSchwegler_S12_Aufg1 import AshaSchwegler_S12_Aufg1
import matplotlib.pyplot as plt


a = 0
b = 10
h = 0.1
y0 = 2
x = np.arange(a, b + h, step=h, dtype=np.float64)


def f(x,y):
    return x**2/y


def f_exakt(x):
    return np.sqrt(((2*x**3)/3)+4)

def euler(f,x,h,y0):
    n = int((b-a)/h)
    y = np.full(x.shape[0], 0, dtype=np.float64)
    y[0] = y0
    for i in range(n):

        y[i+1] = y[i] + h*f(x[i],y[i])
        
    return y


def mittelpunkt(f,x,h,y0):
    n = int((b-a)/h)
    y = np.full(x.shape[0], 0, dtype=np.float64)
    y[0] = y0
    for i in range(n):
        x_mitte = x[i] + h/2.0
        y_mitte = y[i] + h/2.0 * f(x[i],y[i])
        x[i+1] = x[i]+h
        y[i+1] = y[i] + h * f(x_mitte,y_mitte)
    return y


def mod_euler(f,x,h,y0):
    n = int((b-a)/h)
    y = np.full(x.shape[0], 0, dtype=np.float64)
    y_euler = np.full(x.shape[0], 0, dtype=np.float64)
    y[0] = y0
    for i in range (n):
        y_euler[i+1] = y[i] + h*f(x[i],y[i])
        k1 = f(x[i],y[i])
        k2 = f(x[i+1],y_euler[i+1])
        y[i+1] = y[i] + h*(k1+k2)/2
    return y

def globaler_fehler(x,y):
    return np.abs(f(x,y),f(x))


y_exakt = f_exakt(x)
y_eulerverfahren = euler(f,x,h,y0)
y_mittelpunktverfahren = mittelpunkt(f,x,h,y0)
y_mod_euler = mod_euler(f,x,h,y0)
x,y_runge_kutta = AshaSchwegler_S12_Aufg1(f,a,b,100,y0)
print('x = ' + str(x),'\n')
print('y_euler = ' + str(y_eulerverfahren),'\n')
print('y_midpoint = ' + str(y_mittelpunktverfahren),'\n')
print('y_mod_euler = ' + str(y_mod_euler),'\n')
print('y_runge_kutta  = ' + str(y_runge_kutta),'\n')

plt.figure(0)
plt.title("Klassische Runga-Kutta Verfahren vs. Exakte Lösung")
plt.plot(x, y_exakt, label='Exakt')
plt.plot(x, y_eulerverfahren, label='Eulerverfahren')
plt.plot(x, y_mittelpunktverfahren, label='Mittelpunktverfahren')
plt.plot(x, y_mod_euler, label='Modifiziertes Eulerverfahren')
plt.plot(x, y_runge_kutta, label='Runge-kutte')
plt.legend()
plt.show()


plt.figure(1)
plt.title('Global Error')
plt.semilogy()
plt.plot(x, np.abs(y_eulerverfahren - y_exakt), label='Euler')
plt.plot(x, np.abs(y_mittelpunktverfahren - y_exakt), label='Mittelpunkt')
plt.plot(x, np.abs(y_mod_euler - y_exakt), label='Mod. Euler')
plt.plot(x, np.abs(y_runge_kutta - y_exakt), label='Runge-Kutta')
plt.legend()
plt.show()