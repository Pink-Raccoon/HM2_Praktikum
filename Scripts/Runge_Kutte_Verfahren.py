# -*- coding: utf-8 -*-
"""
Created on Sun May 22 12:55:57 2022

@author: Asha
"""

import numpy as np
import matplotlib.pyplot as plt

a = 0.
b = 1.
h = 0.1
n = np.int((b-a)/h)
y0 = 0
rows = 4

x = np.zeros(n+1)
z = np.zeros([rows,n+1])
y0 = np.array([0.,2.,0.,0.])

x[0] = a
z[:,0] =np.array([0.,2.,0.,0.])

print('x= ',x,'\n')
print('z= ',z,'\n')

def f(x,z): 
    res = np.empty(z.shape)
    res[:-1] = z[1:]
    res[-1] = np.sin(x)+5-1.1*z[3]+0.1*z[2]+0.3*z[0]
    return res
print(f(x[0],z[:,0]),'\n')


for i in range(0,n):
    x[i+1]=x[i]+h
    z[:,i+1]=z[:,i]+h*f(x[i],z[:,i])
    
    print(x[i+1],'\n')
    print(z[:,i+1]) 
    
    

plt.plot(x,z[0,:],x,z[1,:],x,z[2,:],x,z[3,:]), plt.legend(["Lösung y(x)", "y'(x)","y''(x)","y'''(x)"])   


def runge_kutta_verfahren(f,x0,y0,xn,n):
    h = (xn-x0)/n
    x = np.linspace(x0,xn, n+1)
    y= np.empty((n+1,y0.size))
    y[0] = y0    
    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + (h / 2.0), y[i] + (h / 2.0) * k1)
        k3 = f(x[i] + (h / 2.0), y[i] + (h / 2.0) * k2)
        k4 = f(x[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h * (1 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        z[:,i+1] = z[:,i] + h * (1 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            
        print('K1 = ',k1,'\n')
        print('K2 = ',k2,'\n')
        print('K3 = ',k3,'\n')
        print('K4 = ',k4,'\n')
        print('y[' + str(i+1) + '] =',y[i+1],'\n')
        
    return x,y
            
            




x,y = runge_kutta_verfahren(f,a,y0,b,n)

plt.figure(1)
plt.plot(x, y, label='Runge-Kutta')
plt.legend()
plt.show()