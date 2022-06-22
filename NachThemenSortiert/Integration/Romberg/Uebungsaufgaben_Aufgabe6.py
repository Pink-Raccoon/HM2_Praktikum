# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 14:09:34 2022

@author: Asha
"""

import numpy as np

np.set_printoptions(precision=2)
# np.set_printoptions(suppress = True)

a = 0
b = 40
n = 8

def f(t):
    return 2*np.exp(-((t/10)-2)**4)


def Tf(f, a, b, h, j):
    print("     Startwert a = " + str(a))
    print("       Endwert b = " + str(b))
    print("Anzahl Schritte n = " + str(n))

    

    print("  Schrittweite h = (b - a) / n = " + str(h))

    xi = np.array([a + i * h for i in range(1, n)], dtype=np.float64)

    print("xi = " + str(xi))

    T = h * ((f(a) + f(b)) / 2 + np.sum(f(xi)))

    print("T{}0 = h * ((f(a) + f(b)) / 2 + SUM(f(xi))) = {}\n".format(j, T))

    return T


def romberg_extrapolate(f, a, b, m):
    print("1. Berechne die Tj0 mit der Trapezregel:")
    print("----------------------------------------")
    T = np.zeros((m + 1, m + 1), dtype=np.float64)
 

    T[0:, 0] = [Tf(f, a, b, 2 ** j, j) for j in range(m + 1)]

    print("2. Berechne die Tjk aus den Tj0:")
    print("--------------------------------")
    for k in range(1, m + 1):
        for j in range(m + 1 - k):
            T[j, k] = (4**k * T[j + 1, k - 1] - T[j, k - 1]) / (4**k - 1)
            print("T{}{} = (4^{} * T{}{} - T{}{}) / (4^{} - 1) = {}".format(j, k, k, j + 1, k - 1, j, k - 1, k, T[j, k]))

    print('\nTij = \n', T)
    return T[0, m]






T = romberg_extrapolate(f, a, b, 3)

print('\nIntegral mit Romberg-Extrapolation: ', round(T,2))



