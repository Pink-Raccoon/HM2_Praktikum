# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 09:17:15 2022

@author: ashas
"""
import numpy as np
import matplotlib.pyplot as plt
import math

v_rel = 2600
m_a = 300000
m_E = 80000
a = 0
b = 190
h = 0.1
y0 = 0
g = 9.8
mue = (m_a - m_E)/b
n = np.int((b-a)/h)
rows = 3

x = np.zeros(n+1)
z = np.zeros([rows,n+1])

x[0] = a
z[:,0] =np.array([0,0,0])


def f(x,z):
    return np.array([z[0],z[1],[v_rel * 
  (mue)/(m_a-mue*x)-g - math.exp(-z[0]/8000)/(m_a-mue*x)]])


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