# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:58:38 2022

@author: ashas
"""

import numpy as np
import matplotlib.pyplot as plt

a = 0
b = 10
y0 = 2
h = 0.1
n = int((b-a)/h)
x = np.linspace(a,b,n+1)

def f(x,y):
    return x**2/y

def f_exakt(x):
    return np.sqrt(((2*x**3)/4)+4)

def klass_kutta(f,a,b,n,y0):
    y = np.full(x.shape[0],0,dtype= np.float64)
    y[0] = y0
    h_halb = h/2
    
    for i in range(n):
        k1 = f(x[i],y[i])
        k2 = f(x[i]+h_halb, y[i]+h_halb * k1)
        k3 = f(x[i]+h_halb, y[i]+h_halb * k2)
        k4 = f(x[i]+h_halb, y[i]+h * k3)
        
        y[i+1] = y[i] + h * (1/6)*(k1 + 2*k2 + 2*k3 + k4)
        
        
    return y

def euler(f,x,h,y0):
    y = np.full(x.shape[0], 0, dtype=np.float64)
    y[0] = y0

    for i in range(n):
        y[i + 1] = y[i] + h * f(x[i], y[i])
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

def AshaSchwegler_S12_Aufg3(f,x,h,y0):
    y_euler = euler(f, x,h,y0)
    y_mittelpunkt = euler_mittel(f, x,h,y0)
    y_modeuler = euler_mod(f, x,h,y0)
    y_kutta = klass_kutta(f, a, b, n, y0)
    
    return [y_euler,y_mittelpunkt,y_modeuler,y_kutta]

[y_euler,y_mittelpunkt,y_modeuler,y_kutta] = AshaSchwegler_S12_Aufg3(f,x,h,y0)
absoluterFehlerEuler = abs(y_euler-f_exakt(x))
absoluterFehleryMittelpunkt = abs(y_mittelpunkt-f_exakt(x))
absoluterFehlerModeuler = abs(y_modeuler-f_exakt(x))
absoluterFehlerKutta = abs(y_kutta-f_exakt(x))


# print('Absoluter Fehler von y_euler =',absoluterFehlerEuler,'\n')
# print('Absoluter Fehler von y_mittelpunkt =', absoluterFehleryMittelpunkt,'\n')
# print('Absoluter Fehler von y_modeuler =', absoluterFehlerModeuler,'\n')
# print('Absoluter Fehler von y_kutta =', absoluterFehlerKutta,'\n')

plt.figure(1)
plt.grid()
plt.plot(x,y_euler,label="Euler")
plt.plot(x,y_mittelpunkt,label="Mittelpunktverfahren")
plt.plot(x,y_modeuler,label="Mod.Eulerverfahren")
plt.plot(x,y_kutta,label="Klass.Runge-KuttaVerfahren")
plt.plot(x,f_exakt(x),label="exakt")


plt.legend()
plt.show()

plt.figure(2)
plt.grid()
plt.semilogy()
plt.plot(x,absoluterFehlerEuler,label="absoluterFehlerEuler")
plt.plot(x,absoluterFehleryMittelpunkt,label="absoluterFehleryMittelpunkt")
plt.plot(x,absoluterFehlerModeuler,label="absoluterFehlerModeuler")
plt.plot(x,absoluterFehlerKutta,label="absoluterFehlerKutta")