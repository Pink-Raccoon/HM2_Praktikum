# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:55:17 2022

@author: ashas
"""

import numpy as np
import math
from AshaSchwegler_S11_Aufg1 import AshaSchwegler_S11_Aufg1

def f(x,y):
    return x**2 / y

def f_exakt(x):
    return math.sqrt(((2*x**3)/3)+4)

def euler(f, x0, y0, xn, n):
    h = (xn-x0)/n
    x = np.linspace(x0, xn, n+1)
    y = np.empty(n+1)
    x[0] = x0
    y[0] = y0
    for i in range(n):
        y[i+1] = y[i] + h*f(x[i],y[i])
    AshaSchwegler_S11_Aufg1(f,x0,xn,y0,y[n],)
    return x, y

def mittelpunktverfahren(f, x0, y0, xn, n):
    
    h = (xn-x0)/n
    x = np.linspace(x0, xn, n+1)
    y = np.empty(n+1)
    x[0] = x0
    y[0] = y0
    for i in range(n):
        x_halbe = x[i] + h
        y_halbe = y[i] + h * f(x[i],y[i])
        y[i+1] = y[i] + h*f(x_halbe,y_halbe)
    return x, y

def modifiziertenEulerverfahren(f, x0, y0, xn, n):
    h = (xn-x0)/n
    x = np.linspace(x0, xn, n+1)
    y = np.empty(n+1)
    y[0] = y0
    x[0] = x0    
    for i in range(n):
        k1 = f(x[i],y[i])
        y_euler = y[i] + h * k1
        k2 = f(x[i+1],y_euler[i+1])
        y[i+1] = y[i] + h*((k1+k2)/2)
    return x, y

def AshaSchwegler_S11_Aufg3(f,a,b,n,y0):
   
    
    
    