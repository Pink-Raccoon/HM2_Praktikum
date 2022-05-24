# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:52:44 2022

@author: ashas
"""

import numpy as np
from AshaSchwegler_S12_Aufg1 import AshaSchwegler_S12_Aufg1
import matplotlib.pyplot as plt

a = 1
b = 6
h = 0.01
n = (b-a)/h
y0 = 5
x = np.arange(a, b + h, step=h, dtype=np.float64)

def f(x,y):
    return 1-(y/x)

def f_exakt(x):
    return (x/2) +(9/(2*x))


x_runga,y_runga = AshaSchwegler_S12_Aufg1(f, a, b, n, y0)
y_exakt = f_exakt(x)
#c
from AshaSchwegler_S12Aufg2c import runge_kutta_custom
y_custom = runge_kutta_custom(f,x,h,y0,4)

plt.figure(0)
plt.title("Klassische Runga-Kutta Verfahren vs. Exakte Lösung")
plt.plot(x_runga, y_runga, label='Runge-Kutta (Numerisch)')
plt.plot(x, y_exakt, label='Exakt')
plt.plot(x, y_custom, label='Runge-kutte-custom')
plt.legend()
plt.show()


plt.figure(1)
plt.title("Exakte Lösung")
plt.plot(x, y_exakt, label='Exakt')
plt.legend()
plt.show()


plt.figure(2)

plt.title('Runge-kutte-custom')
plt.plot(x, y_custom, label='Runge-kutte-custom')
plt.legend()
plt.show()


