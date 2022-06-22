# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 18:15:29 2022

@author: ashas
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.array([0,10,20,30,40,50,60,70,80,90,100,110])
y = np.array([76,92,106,123,137,151,179,203,227,250,281,309])


n = len(x) #Anz. Gleichungen
m = 4 # a,b,c,d, Anzahl Unbekannten


def f1(x):
    return x**3
def f2(x):
    return x**2
def f3(x):
    return x
def f4(x):
    return 1

def f(x, lam):
    return lam[0]*f1(x) + lam[1]*f2(x) + lam[2]*f3(x) + lam[3]*f4(x)

def f_other(x,lam):
    return lam[0]*f2(x) + lam[1]*f3(x) + lam[2]*f4(x)

# def f(x,a):
#     return a[0]*x + a[1]*x**2 + a[3]*x**3+ a[4]*x**4 + 1

A = np.zeros([n,m]) # Zeilen = anzahl x, Spalten = anzahl koeff.
A[:,0] = f1(x)
A[:,1] = f2(x)
A[:,2] = f3(x)
A[:,3] = f4(x)


A_other = np.zeros([n,3])
A_other[:,0] = f2(x)
A_other[:,1] = f3(x)
A_other[:,2] = f4(x)


[Q,R] = np.linalg.qr(A)
[Q_other, R_other] = np.linalg.qr(A_other)
lam_qr = np.linalg.solve(R, Q.T @ y)
lam_qr_other = np.linalg.solve(R_other, Q_other.T@y)

print('a0 =', lam_qr[0],'\n')
print('a1 =', lam_qr[1],'\n')
print('a2 =', lam_qr[2],'\n')
print('a3 =', lam_qr[3],'\n')



print('b0 =', lam_qr_other[0],'\n')
print('b1 =', lam_qr_other[1],'\n')
print('b2 =', lam_qr_other[2],'\n')

Fehlerfunktional_3 = np.linalg.norm(y-A@lam_qr,2)**2
Fehlerfunktional_2 = np.linalg.norm(y-A_other@lam_qr_other,2)**2




plt.figure(1)
plt.scatter(x,y,marker='x',color='black',label='Messdaten')
plt.plot(x,f(x,lam_qr),'--',color='red', label='Ausgleichfunktion')
plt.plot(x,f_other(x,lam_qr_other), '--',label= 'Ausgl.x^2')
plt.legend()
plt.show()
