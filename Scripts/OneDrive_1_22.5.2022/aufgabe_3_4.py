#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 18:18:46 2020

@author: miec
"""

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2-2

x = np.arange(0,2,0.001)
y = f(x)
plt.plot(x,y)
plt.grid()
plt.show()

#%%

def newton(x):
    return x/2+1/x
    
n = np.arange(11)
x = np.zeros(11)
x[0] = 2
for i in n[:-1]:
    x[i+1] = newton(x[i])
    print(x[i+1])

plt.figure()
plt.plot(n,x)
plt.show()

#%%

error = np.abs(x-np.sqrt(2))
plt.figure()
plt.plot(n,error)
plt.show()

print(error[4])

#%%

error = np.abs(x-np.sqrt(2))
plt.figure()
plt.semilogy(n,error)
plt.ylim([1e-15,1])
plt.show()