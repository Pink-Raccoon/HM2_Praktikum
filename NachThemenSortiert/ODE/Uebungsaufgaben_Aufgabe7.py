# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 14:54:57 2022

@author: Asha
"""

import numpy as np
import matplotlib.pyplot as plt

y0 = 0
a = 0
b = 6
h = 0.2

x = np.arange(a, b + h, step=h, dtype=np.float64)

def f(y,t):
    return (0.1*y) + np.sin(2*t)    





def interpolate_euler(f, x, h, y0):
    y = np.full(x.shape[0], 0, dtype=np.float64)
    y[0] = y0

    for i in range(x.shape[0] - 1):
        y[i + 1] = y[i] + h * f(x[i], y[i])

    return y


def interpolate_runge_kutta(f, x, h, y0):
    y = np.full(x.shape[0], 0, dtype=np.float64)
    y[0] = y0

    for i in range(x.shape[0] - 1):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + (h / 2.0), y[i] + (h / 2.0) * k1)
        k3 = f(x[i] + (h / 2.0), y[i] + (h / 2.0) * k2)
        k4 = f(x[i] + h, y[i] + h * k3)

        y[i + 1] = y[i] + h * (1 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        print('y{}'.format(i),y[i])

    return y

plt.figure(1)
plt.grid()
plt.plot(x,interpolate_euler(f, x, h, y0), label='Euler')
plt.plot(x,interpolate_runge_kutta(f, x, h, y0), label = 'Runge-Kutta')
plt.legend()
plt.show()

plt.figure(2)
plt.grid()
plt.semilogy()
plt.plot(x,abs(interpolate_euler(f, x, h, y0)-interpolate_runge_kutta(f, x, h, y0)), label='Differenz')

plt.legend()
plt.show()