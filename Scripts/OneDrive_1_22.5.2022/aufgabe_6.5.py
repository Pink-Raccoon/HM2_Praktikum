# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 13:57:51 2021

HM2, Aufg. 6.5

@author: knaa / miec
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3,4])
y = np.array([6,6.8,10.,10.5])

def f1(x):
    return x

def f2(x):
    return 1

def f(lam,x):
    return lam[0]*f1(x)+lam[1]*f2(x)

A =np.empty([4,2])
A[:,0] = f1(x)
A[:,1] = f2(x)

[Q,R] = np.linalg.qr(A)
print('Q =')
print(Q)
print(Q.T @ Q)
print('R =')
print(R)

#%% solve

lam = np.linalg.solve(R, Q.T @ y)
print('lam =', lam)

#%% plot

x_fine = np.linspace(x[0],x[-1])
plt.plot(x,y,'o')
plt.plot(x_fine,f(lam, x_fine))
plt.xlabel('x'), plt.ylabel('y=f(x)'), plt.legend(['data','f(x)=a*x+b'])

