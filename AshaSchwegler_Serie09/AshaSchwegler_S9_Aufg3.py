# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 16:46:56 2022

@author: ashas
"""

import numpy as np

def sumTrapez(f, a, b, n):
    print("     Startwert a = " + str(a))
    print("       Endwert b = " + str(b))
    print("Anzahl Schritte n = " + str(n))
    sum = 0
    h = (b-a)/n
    print("  Schrittweite h = (b - a) / n = " + str(h))
    for i in range(1, n):
        sum += f(a + i*h)
    return h * ((f(a) + f(b))/2 + sum)

def AshaSchwegler_S9_Aufg3(f, a, b, m):
    print("1. Berechne die Tj0 mit der Trapezregel:")
    print("----------------------------------------")
    n = m + 1
    T = np.zeros((n, n))
    for j in range(n):
        T[j,0] = sumTrapez(f, a, b, 2**j)
        print("2. Berechne die Tjk aus den Tj0:")
        print("--------------------------------")
    for k in range(1,n):
        for j in range(m-k,-1,-1):
            T[j,k] = (4**k*T[j+1, k-1] - T[j,k-1]) / (4**k - 1)
            print("T{}{} = (4^{} * T{}{} - T{}{}) / (4^{} - 1) = {}".format(j, k, k, j + 1, k - 1, j, k - 1, k, T[j, k]))
    print('\nTij = \n', T)
    return T[0,m]

def f(x):
    return np.cos(x**2)




print(AshaSchwegler_S9_Aufg3(f, 0, np.pi, 4))