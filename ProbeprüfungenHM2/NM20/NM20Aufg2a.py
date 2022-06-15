# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:49:49 2022

@author: ashas
"""

import numpy as np
import matplotlib.pyplot as plt
from NM20Aufg2b import kutta_allgemein
a= 1
b = 6
h = 0.01
y0 = 5

n= np.int((b-a)/h)
x = np.arange(a, b + h, step=h, dtype=np.float64)
a_kutt = np.array([
    [0,0,0,0],
    [0.5,0,0,0],
    [0.75,0.75,0,0],
    [1,1,1,0]], dtype=np.float64)


b_kutt = np.array([1/10,4/10,4/10,1/10],dtype=np.float64)
c=np.array([0.25,0.5,0.5,0.75],dtype=np.float64)

n_kutt = 4
s=4
def f(t,y):
    return 1-(y/t)

def f_exakt(t):
    return (t/2) + (9/(2*t))






def runge_kutta_klass(f,x,n,y0):
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
y= runge_kutta_klass(f,x,n,y0)
y_custom = kutta_allgemein(f,x,a_kutt,b_kutt,c,h,y0,s)

absFehlerKlassisch = abs(y-f_exakt(x))

absFehlerCustom = abs(y_custom-f_exakt(x))


plt.figure(1)
plt.grid()
plt.plot(x,y,label="Runge-Kutta")
plt.plot(x,f_exakt(x),label="Exakte Lösung")
plt.plot(x,y_custom,label="Runge-Kutta Custom")
plt.legend()
plt.show()

#c
plt.figure(1)
plt.grid()
plt.semilogy()
plt.plot(x,absFehlerKlassisch,label="Absoluter Fehler des klassischen Runge-Kutta Verfahrens")
plt.plot(x,absFehlerCustom,label="Absoluter Fehler des Custom Runge-Kutta Verfahrens")
plt.legend()
plt.show()