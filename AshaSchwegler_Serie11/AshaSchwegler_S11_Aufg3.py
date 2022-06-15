# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:55:17 2022

@author: ashas
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from AshaSchwegler_S11_Aufg1 import AshaSchwegler_S11_Aufg1

a = 0
b = 1.4
h = 0.7
n = int((b-a)/h)
x = np.linspace(a,b,n+1)
y0 = 2

def f(x, y):
    return x**2/y

def euler(f,x,h,y0):
    y = np.full(x.shape[0], 0, dtype=np.float64)
    y[0] = y0

    for i in range(n):
        y[i + 1] = y[i] + h * f(x[i], y[i])
        
        print('i{} ='.format(i),y )

    return y




def euler_mittel(f,x,h,y0):
    y = np.full(x.shape[0], 0, dtype=np.float64)
    y[0] = y0
    
    for i in range(n):
        x_halb = x[i] + h/2
        y_halb = y[i] + h/2 * f(x[i],y[i])
        
        y[i+1] = y[i] + h * f(x_halb,y_halb)
        
    return y

def euler_mod(f,x,h,y0):
    y = np.full(x.shape[0],0,dtype=np.float64)
    y[0] = y0
    y_euler = np.empty(n+1,dtype=np.float64)
    for i in range(n):
        y_euler[i+1] = y[i] + h * f(x[i],y[i])
        k1 = f(x[i],y[i])
        k2 = f(x[i+1], y_euler[i+1])
        
        y[i+1] = y[i] + h * ((k1+k2)/2)
        
    return y
   
def AshaSchwegler_S11_Aufg3(f, a, b, n, y0):
    x = np.linspace(a, b, n+1)
    dx = (b-a)/n
    y_euler = euler(f, x,h,y0)
    y_mittelpunkt = euler_mittel(f, x,h,y0)
    y_modeuler = euler_mod(f, x,h,y0)
    return [x, y_euler, y_mittelpunkt, y_modeuler]


(x, y_euler, y_mittelpunkt, y_modeuler) = AshaSchwegler_S11_Aufg3(f, a,b,n,y0)

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
AshaSchwegler_S11_Aufg1(f, 0,1.4, 2, 8, 0.5, 1)
plt.show()