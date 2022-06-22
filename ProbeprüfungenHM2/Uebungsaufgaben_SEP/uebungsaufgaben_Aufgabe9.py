# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 16:36:45 2022

@author: Asha
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
a = 0
b = 3
h = 0.05
n = np.int((b-a)/h)
rows = 2

x = np.zeros(n+1)
z = np.zeros([rows,n+1])

x[0] = a
z[:,0] =np.array([20.0,0.0])

def f(x,z): 
    return np.array([z[1], -0.1*z[1]*abs(z[1])-10])

for i in range(x.shape[0]-1):
    x[i+1]=x[i]+h
    k1 = f(x[i], z[:,i]) 
    k2 = f(x[i] + (h / 2.0), z[:,i] + (h / 2.0) * k1)
    k3 = f(x[i] + (h / 2.0), z[:,i] + (h / 2.0) * k2)
    k4 = f(x[i] + h, z[:,i] + h * k3)
    z[:,i+1] = z[:,i] + h*(1/6)*(k1+2*k2+2*k3+k4)
    
print('z =',z)
    
plt.figure(1)
plt.title("Lösung mit Runge-Kutta")
plt.plot(x,z[0,:],x,z[1,:]) 
plt.legend(["Lösung y(x)", "y'(x)"]) 

plt.show()

#%%

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
a = 0
b = 8
h = 0.05
n = np.int((b-a)/h)
rows = 2

x = np.zeros(n+1)
z = np.zeros([rows,n+1])

x[0] = a
z[:,0] =np.array([20.0,0.0])

def f(x,z): 
    return np.array([z[1], -0.1*z[1]*abs(z[1])-10])



for i in range(x.shape[0]-1):

    if z[0:0,i] < 0 and z[1:1,i] < 0:
         -1*z[i:i,i]
    
    x[i+1]=x[i]+h
    k1 = f(x[i], z[:,i]) 
    k2 = f(x[i] + (h / 2.0), z[:,i] + (h / 2.0) * k1)
    k3 = f(x[i] + (h / 2.0), z[:,i] + (h / 2.0) * k2)
    k4 = f(x[i] + h, z[:,i] + h * k3)
    z[:,i+1] = z[:,i] + h*(1/6)*(k1+2*k2+2*k3+k4)
    
print('z =',z)

plt.figure(2)
plt.title("Lösung mit Runge-Kutta")
plt.plot(x,z[0,:],x,z[1,:]) 
plt.legend(["Lösung y(x)", "y'(x)"]) 

plt.show()