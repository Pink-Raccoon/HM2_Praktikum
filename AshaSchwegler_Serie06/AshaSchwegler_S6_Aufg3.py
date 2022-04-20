# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 21:29:02 2022

@author: Asha
"""
import numpy as np
import matplotlib.pyplot as plt
data=np.array([
    [1971, 2250.],
    [1972, 2500.],
    [1974, 5000.],
    [1978, 29000.],
    [1982, 120000.],
    [1985, 275000.],
    [1989, 1180000.],
    [1989, 1180000.],
    [1993, 3100000.],
    [1997, 7500000.],
    [1999, 24000000.],
    [2000, 42000000.],
    [2002, 220000000.],
    [2003, 410000000.],   
    ])


#3.1

[n,m] = np.shape(data)
y = data[:,m-1]
x =data[:,:m-1].T
print('x: ',x, '\n')
A = np.zeros([n,m])

def f2(x):
    return(x-1970)

def f1(x):
    return 1

A[:,:m-1] = f1(x)
A[:,m-1] = f2(x)
print('A = ',A, '\n')

y_neu = np.log10(y)
print('y_neu: ', y_neu, '\n')

lam = lam = np.linalg.solve(A.T @ A, A.T @ y_neu)
print('lambda Normalgleichung: ', lam, '\n')

def f(A,lam):
    yy = 0
    for i in range(m):
        yy += lam[i]*A[:,i]
    return yy

print (f(A,lam))


# plt.plot(x,y.T)
# plt.plot(x,yy)
# plt.yscale("log")
# plt.legend(["Messpunkte", "Ausgleich"])
# plt.show()
