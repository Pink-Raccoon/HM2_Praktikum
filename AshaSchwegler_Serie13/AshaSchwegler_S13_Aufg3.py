# -*- coding: utf-8 -*-
"""
Created on Sun May 22 20:10:29 2022

@author: Asha
"""

import numpy as np
import matplotlib.pyplot as plt

h = 0.1
a = 0.
b = 20.
n = np.int((b-a)/h)
m = 97000.

rows = 2
x = np.zeros(n+1)
z = np.zeros([rows,n+1])
y0 = 0

x[0] = a
z[:,0] =np.array([0.,100])





def f(x,z):
    res = np.empty(z.shape)
    res[:-1] = z[1:]
    res[-1] =(-5.*z[1]**2.-270000.)/m
    return res
    
print(f(x[0],z[:,0]),'\n')




  



def mittelpunktverfahren(f, x, h, y0):
    y = np.full(x.shape[0], 0, dtype=np.float64)
    y[0] = y0

    for i in range(x.shape[0] - 1):
        y[i + 1] = y[i] + h * f(x[i] + (h / 2.0), y[i] + (h / 2.0) * f(x[i], y[i]))
    return y


y = mittelpunktverfahren(f, x, h, y0)

plt.figure(1)
plt.plot(x,y[1,:],x,y[0,:]), plt.legend(["Lösung x(t)", "v(t)"])
plt.legend()
plt.show()


# braucht 20Sekunden um zum Stillstand zu kommen und hat einen Bremsweg von 500m