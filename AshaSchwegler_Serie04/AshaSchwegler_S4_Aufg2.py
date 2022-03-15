# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 19:59:50 2022

@author: Asha
"""

import numpy as np
import matplotlib.pyplot as plt

def lagrange(x,y,x_int):
    y_int = 0
    i_int = 1
    i = 0
    j = 1
    for i in range(len(y)):
        for j in range(len(x)):
            if (x[i] != x[j]):
                i_int *= (x_int-x[j])/(x[i]-x[j])
        print("I",i,"=",i_int)
        y_int += i_int * y[i]
        i_int = 1
    return y_int

def plotLagrange(x,y,x_vec):
    y_vec = np.array([])
    for i in range(0,len(x_vec)):        
        y_vec = np.append(y_vec,np.array([lagrange(x,y,x_vec[i])]))
    plt.plot(x_vec,y_vec)
   
    plt.ylim(-100,250) 

    plt.scatter(x,y)    
    plt.legend(["Lagrange" ])
    plt.show()      


x = [0,2500,5000,10000]
y = [1013,747,540,226]
x_int = 3750

y_int = lagrange(x,y,x_int)
print(y_int)



