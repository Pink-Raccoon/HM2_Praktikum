# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 15:26:58 2022

@author: Asha
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
L = 1.0
R = 80.0
C = 4.0*10.0**-4.0
U = 100

q0 = 0
q_abl_0 =0

h = 0.01

n = 1

rows = 2

x = np.zeros(n+1)
z = np.zeros([rows,n+1])


z[:,0] =np.array([0.,0.])



def f(x,z):
   return np.array([z[1],(U-R*z[1]+(1/C)*z[0])*(1/L)])




for i in range(x.shape[0] - 1):
    x[i+1] = x[i] + h
    x_halb = x[i] + h/2
    y_halb = z[:,i] + h/2 * f(x[i],z[:,i])
    
    z[:,i+1] = z[:,i] + h * f(x_halb,y_halb)  



print('z1 =',z)
