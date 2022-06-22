# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 21:35:03 2022

@author: Asha
"""

import numpy as np
import matplotlib.pyplot as plt

v_rel = 2600.0
m_A = 300000.0
m_E = 80000.0
tE = 190.0
g = 9.81
mue = (m_A-m_E)/tE

a = 0
b = tE

h = 0.1

n = int((b-a)/h)
rows = 2

x = np.zeros(n+1)
z = np.zeros([rows,n+1])

x[0] = a
y0 = np.array([0.,0.])
z[:,0] =np.array([0.,0.])

def f(t,z): 
  y = np.array([z[1], v_rel * (mue / (m_A-mue*t))-g-(np.exp(-z[0]/8000)/(m_A-mue*t))*z[1]**2])
  return y
#%%
for i in range(x.shape[0]-1):
    x[i+1]=x[i]+h
    k1 = f(x[i], z[:,i]) 
    k2 = f(x[i] + (h / 2.0), z[:,i] + (h / 2.0) * k1)
    k3 = f(x[i] + (h / 2.0), z[:,i] + (h / 2.0) * k2)
    k4 = f(x[i] + h, z[:,i] + h * k3)
    

    z[:,i+1] = z[:,i] + h*(1/6)*(k1+2*k2+2*k3+k4)
    
    
plt.figure(1)
plt.title("Höhe")
plt.plot(x,z[0,:]) 
plt.legend(["Lösung h(t)"]) 

plt.show()

plt.figure(2)
plt.title("Geschwindigkeit")
plt.plot(x,z[1,:]) 
plt.legend(["h'(t)"]) 
plt.show()


# plt.figure(3)
# plt.title("Geschwindigkeit")
# plt.plot(x,f(x,z[1,:])) 
# plt.legend(["h'(t)"]) 
# plt.show()


#%%
def kutta(f,x,h):
    for i in range(x.shape[0]-1):
        x[i+1]=x[i]+h
        k1 = f(x[i], z[:,i]) 
        k2 = f(x[i] + (h / 2.0), z[:,i] + (h / 2.0) * k1)
        k3 = f(x[i] + (h / 2.0), z[:,i] + (h / 2.0) * k2)
        k4 = f(x[i] + h, z[:,i] + h * k3)
        z[:,i+1] = z[:,i] + h*(1/6)*(k1+2*k2+2*k3+k4)
        
    return z

def euler_mod(f,x,h):

    z_euler = z = np.zeros([rows,n+1])

    for i in range(n):
        z_euler[:,i+1] = z[:,i] + h * f(x[i],z[:,i])
        k1 = f(x[i],z[:,i])
        k2 = f(x[i+1], z_euler[:,i+1])
        
        z[:,i+1]=z[:,i] + h * ((k1+k2)/2)

        
    return z
y = euler_mod(f, x, h)
kutt = kutta(f,x,h)
plt.figure(4)
plt.title("Euler")
plt.semilogy()
plt.plot(x,(abs(kutt[0,:]-y[0,:])))
plt.plot(x,(abs(kutt[1,:]-y[1,:])))

plt.legend(["euler", "relative Abweichung"])  

 