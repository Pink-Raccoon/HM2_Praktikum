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

def euler(f, x0, y0, xn, n):
    h = (xn-x0)/n
    x = np.linspace(x0, xn, n+1)
    y = np.empty(n+1)
    y[0] = y0
    for i in range(n):
        y[i+1] = y[i] + h*f(x[i],y[i])
    return x, y

x = np.linspace(-1.5,1.5)
x_euler, y_euler = euler(f, -1.5,0,1.5,5)

print('x_euler =', x_euler)
print('y_euler =', y_euler)

plt.plot(x, y(x), x_euler, y_euler)
plt.grid()
plt.xlabel('t')
plt.ylabel('y')
plt.legend(['y(t)', 'Euler'])
plt.show()