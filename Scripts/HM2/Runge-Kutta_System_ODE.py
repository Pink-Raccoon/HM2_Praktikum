# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 11:37:42 2022

@author: ashas
"""

import numpy as np
import matplotlib.pyplot as plt

a = 0.
b = 1.
h = 0.1
n = np.int((b-a)/h)
rows = 4

x = np.zeros(n+1)
z = np.zeros([rows,n+1])

x[0] = a
z[:,0] =np.array([0.,2.,0.,0.])

def f(x,z): 
    return np.array([z[1], z[2], z[3], np.sin(x)+5-1.1*z[3]+0.1*z[2]+0.3*z[0]])

for i in range(x.shape[0]-1):
    x[i+1]=x[i]+h
    k1 = f(x[i], z[:,i]) 
    k2 = f(x[i] + (h / 2.0), z[:,i] + (h / 2.0) * k1)
    k3 = f(x[i] + (h / 2.0), z[:,i] + (h / 2.0) * k2)
    k4 = f(x[i] + h, z[:,i] + h * k3)
    z[:,i+1] = z[:,i] + h*(1/6)*(k1+2*k2+2*k3+k4)
    
plt.figure(1)
plt.title("Lösung mit Runge-Kutta")
plt.plot(x,z[0,:],x,z[1,:],x,z[2,:],x,z[3,:]) 
plt.legend(["Lösung y(x)", "y'(x)","y''(x)","y'''(x)"]) 

plt.show()