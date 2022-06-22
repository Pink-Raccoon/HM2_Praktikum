# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:26:43 2022

@author: ashas
"""
import numpy as np
import matplotlib.pyplot as plt
a = 0
b = 20
h = 0.1
n = np.int((b-a)/h)

x0 = 0
v0 = 100
m = 97000
rows = 2

x = np.arange(a, b + h, step=h, dtype=np.float64)
z = np.zeros([rows,n+1])

x[0] = a
z[:,0] =np.array([0.,100.])

def f(x,z):
   return np.array([z[1],((-5*z[1]**2)-570000)/m])
    
def mittel(f,a,b):
    for i in range(x.shape[0] - 1):
        x[i+1] = x[i] + h
        x_halb = x[i] + h/2
        y_halb = z[:,i] + h/2 * f(x[i],z[:,i])
        
        z[:,i+1] = z[:,i] + h * f(x_halb,y_halb) 
    return z

z = mittel(f, a, b)




plt.figure(1)
plt.plot(x,z[0,:],label='Bremsweg')
plt.plot(x,z[1,:],label='Geschwindigkeit')
plt.legend()
plt.show()

#c

# Bremsweg beträgt ungefähr 800m und die Zeit ungefähr 20sek