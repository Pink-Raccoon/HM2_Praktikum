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
n = 200
m = 97000


x = np.linspace(a,b,n+1)









def f(x,z):
    res = np.empty(z.shape)
    res[:-1] = z[1:]
    res[-1] =((-5*z[1]**2)-270000)/m
    return res
    





  



def mittelpunktverfahren(f, x, h, y0):
    y_mittel = np.empty((n+1, y0.size))
    y_mittel[0] = y0
   

    for i in range(x.shape[0] - 1):
        x_mitte = x[i] + h/2
        y_mitte = y_mittel[i] + (h/2)*f(x[i], y_mittel[i])
        y_mittel[i + 1] = y_mittel[i] + h * f(x_mitte,y_mitte)
        print('y[',str(i),']=y',y_mittel,'\n')
    return y_mittel

y0 = np.array([0.,100])
y_mittel = mittelpunktverfahren(f, x, h, y0)

plt.figure(1)

plt.plot(x,y_mittel[:,0],label='Bremsweg')
plt.plot(x,y_mittel[:,1],label='Bremszeit')



plt.legend()
plt.show()


# braucht 20Sekunden um zum Stillstand zu kommen und hat einen Bremsweg von 1400m