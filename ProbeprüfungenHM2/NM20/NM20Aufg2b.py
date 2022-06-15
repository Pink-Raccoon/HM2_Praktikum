# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:11:15 2022

@author: ashas
"""

import numpy as np
import matplotlib.pyplot as plt



def kutta_allgemein(f,x,a_kut,b_kut,c,h,y0,s):
    y = np.full(x.shape[0],0,dtype= np.float64)
    y[0] = y0
    
    for i in range(x.shape[0]-1):
        k = np.full(s,0,dtype=np.float64)
        for n in range(s):
            k[n] = f(x[i]+c[n]*h,y[i]+h*np.sum([a_kut[n][m]*k[m] for m in range(1,n-1)]))
            y[i+1] = y[i] + h*np.sum([b_kut[n]*k[n] for n in range(1,s)])
            
    return y