# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 18:22:22 2022

@author: Asha
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.array([
        [33.00, 53.00, 3.32, 3.42, 29.00],
        [31.00, 36.00, 3.10, 3.26, 24.00],
        [33.00, 51.00, 3.18, 3.18, 26.00],
        [37.00, 51.00, 3.39, 3.08, 22.00],
        [36.00, 54.00, 3.20, 3.41, 27.00],
        [35.00, 35.00, 3.03, 3.03, 21.00],
        [59.00, 56.00, 4.78, 4.57, 33.00],
        [60.00, 60.00, 4.72, 4.72, 34.00],
        [59.00, 60.00, 4.60, 4.41, 32.00],
        [60.00, 60.00, 4.53, 4.53, 34.00],
        [34.00, 35.00, 2.90, 2.95, 20.00],
        [60.00, 59.00, 4.40, 4.36, 36.00],
        [60.00, 62.00, 4.31, 4.42, 34.00],
        [60.00, 36.00, 4.27, 3.94, 23.00],
        [62.00, 38.00, 4.41, 3.49, 24.00],
        [62.00, 61.00, 4.39, 4.39, 32.00],
        [90.00, 64.00, 7.32, 6.70, 40.00],
        [90.00, 60.00, 7.32, 7.20, 46.00],
        [92.00, 92.00, 7.45, 7.45, 55.00],
        [91.00, 92.00, 7.27, 7.26, 52.00],
        [61.00, 62.00, 3.91, 4.08, 29.00],
        [59.00, 42.00, 3.75, 3.45, 22.00],
        [88.00, 65.00, 6.48, 5.80, 31.00],
        [91.00, 89.00, 6.70, 6.60, 45.00],
        [63.00, 62.00, 4.30, 4.30, 37.00],
        [60.00, 61.00, 4.02, 4.10, 37.00],
        [60.00, 62.00, 4.02, 3.89, 33.00],
        [59.00, 62.00, 3.98, 4.02, 27.00],
        [59.00, 62.00, 4.39, 4.53, 34.00],
        [37.00, 35.00, 2.75, 2.64, 19.00],
        [35.00, 35.00, 2.59, 2.59, 16.00],
        [37.00, 37.00, 2.73, 2.59, 22.00]])



"""
mCH =lam1 · TTank + lam2 · TBenzin + lam3 · pTank + lam4 · pBenzin +lam5
"""


[n,m] = np.shape(data)
y = data[:,m-1]
x = range (n)
A = np.zeros([n,m])
A[:,:m-1] = data[:,:m-1]
A[:,m-1] = np.ones(n)
print(A)

lam = np.linalg.solve(A.T @ A, A.T @ y)
print('lambda Normalgleichung: ', lam, '\n')

def f(A,lam):
    yy = 0
    for i in range(m):
        yy += lam[i]*A[:,i]
    return yy

yy= f(A,lam)



plt.plot(x,y)
plt.plot(x,yy)
plt.legend(["Messpunkte","Ausgleich"])
plt.show()