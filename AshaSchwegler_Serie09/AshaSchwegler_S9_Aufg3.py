# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 16:46:56 2022

@author: ashas
"""

import numpy as np

def sumTrapez(a,b,n):
    print("1.Schritt: Berechne Tj0 mit Summierten Trapezregel", '\n')
    print("--------------------------------------------------", '\n')
    print("a = " + str(a), '\n')
    print("b = " + str(b), '\n')
    print("b = " + str(n), '\n')
    h = float((b-a)/n)
    result = 0
    for i in range (1,n-1):
        result += f(a+i*h,10)
    result += (f(a,10)+f(b,10))/2
    result *= h
    return result

def AshaSchwegler_S9_Aufg3(f,a,b,m):
    n = 2**
   