# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 16:16:45 2022

@author: vukim
"""

import numpy as np


def Matrix_A(h,n):
    A = np.zeros((n-1, n-1))
    
    A[0,0] = 2*(h[0]+h[1])
    A[0,1] = h[1]
    
    if(n < 4):
        A[1,0] = h[1]
        A[1,1] = 2*(h[0]+h[1])
        return A
    else:
        j = 0
        i = 1
        while (i < n-1):
            if (i == n-2):
                A[i,j] = h[i]
                A[i,j+1] = 2*(h[i]+h[i+1])
            else:
                A[i,j] = h[i]
                A[i,j+1] = 2*(h[i+1]+h[i])
                A[i,j+2] = h[i+1]
            i = i+1
            j = j+1
        return A

def S(x,y,xx):
    n = len(x)-1
    a = y[0 : n]
    c = np.zeros(len(x))
    b = np.zeros(n)
    d = np.zeros(n)
    h = np.zeros((n, 1))
    yy = np.zeros(len(xx))
    z = np.zeros((n-1))

    for i in range(n):
        h[i,0] = x[i+1,0] - x[i,0]
        
    for i in range(1,n):
        z[i-1] = 3 * (y[i+1,0] - y[i,0]) / h[i,0] - 3 * (y[i,0] - y[i-1,0]) / h[i-1,0]
                
    A = Matrix_A(h, n)
    
    c[1:n] = np.linalg.solve(A, z)
   
    for i in range(n):
        b[i] = (y[i+1,0] - y[i,0]) / h[i,0] - (h[i,0]/3) * (c[i+1] + 2*c[i])
        d[i] = (c[i+1] - c[i]) / (3*h[i])
    
  
    for i in range(len(xx)):
        for j in range(n):
            if (xx[i] >= x[j,0] and xx[i] <= x[j+1,0]):
                yy[i] = a[j,0] + b[j] * (xx[i]-x[j,0]) + c[j] * (xx[i]-x[j,0])**2 + d[j] * (xx[i]-x[j,0])**3
     
    return yy

'''
x = np.array([[4.],[6],[8],[10]])
y = np.array([[6.],[3],[9],[0]])
n = len(x)-1
xx = np.arange(x[0,0],x[n,0],0.1)

yy = S(x,y,xx)


import matplotlib.pyplot as plt

plt.plot(xx, yy)
plt.title('kubischen Splinefunktion')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
'''