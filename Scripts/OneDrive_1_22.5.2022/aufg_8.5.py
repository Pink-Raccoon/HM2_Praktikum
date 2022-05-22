#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 07:44:30 2021

@author: miec
"""

import numpy as np
import matplotlib.pyplot as plt

def f(t,y): return t**2 + 0.1*y
def y(t): return -10*t**2 - 200*t - 2000 + 1722.5*np.exp(0.05*(2*t+3))

def runge_kutta_4(f, x0, y0, xn, n):
    h = (xn-x0)/n
    x = np.linspace(x0, xn, n+1)
    y = np.empty(n+1)
    y[0] = y0
    for i in range(n):
        k1 = f(x[i],y[i])
        k2 = f(x[i] + 0.5*h, y[i] + 0.5*h*k1)
        k3 = f(x[i] + 0.5*h, y[i] + 0.5*h*k2)
        k4 = f(x[i] + h, y[i] + h*k3)
        y[i+1] = y[i] + h*(k1+2*k2+2*k3+k4)/6
    return x, y

x = np.linspace(-1.5,1.5)
x_runge_kutta, y_runge_kutta = runge_kutta_4(f, -1.5,0,1.5,5)

plt.plot(x, y(x), '-',x_runge_kutta, y_runge_kutta, '--.')
plt.grid()
plt.xlabel('t')
plt.ylabel('y')
plt.legend(['y(t)', 'Klassisches Runge Kutta Verfahren'])
plt.show()