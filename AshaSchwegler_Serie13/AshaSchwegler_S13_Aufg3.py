# -*- coding: utf-8 -*-
"""
Created on Sun May 22 20:10:29 2022

@author: Asha
"""

import numpy as np
import matplotlib.pyplot as plt

h = 0.1
a = 0
b = 20
n = int((b-a)/h)
m = 97000
rows = 2

x = np.zeros(n+1)
z = np.zeros([rows,n+1])

x[0] = a
z[:,0] =np.array([0.,100.])



def f(x,z):
   return np.array([z[1],((-5*z[1]**2)-570000)/m])




for i in range(x.shape[0] - 1):
    x[i+1] = x[i] + h
    x_halb = x[i] + h/2
    y_halb = z[:,i] + h/2 * f(x[i],z[:,i])
    
    z[:,i+1] = z[:,i] + h * f(x_halb,y_halb)  



plt.figure(1)
plt.plot(x,z[0,:],label='Bremsweg')
plt.plot(x,z[1,:],label='Geschwindigkeit')
plt.legend()
plt.show()


# braucht 20Sekunden um zum Stillstand zu kommen und hat einen Bremsweg von 800m