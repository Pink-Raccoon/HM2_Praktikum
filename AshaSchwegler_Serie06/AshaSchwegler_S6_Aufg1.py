# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 17:07:07 2022

@author: Asha
"""

import numpy as np
import matplotlib.pyplot as plt


"""
Ausgleichsfunktion : a*x**2 + b*x +c
"""
x = np.array([0,10,20,30,40,50,60,70,80,90,100])
y = np.array([999.9,999.7,998.2,995.7,992.2,988.1,983.2,977.8,971.8,965.3,958.4])
n = len(x) #Anz. Gleichungen
m = 3 # a,b,c Anzahl Unbekannten

#Basisfunktionen
def f1(x):
    return x**2

def f2(x):
    return x

def f3(x):
    return 1

def f(x, lam):
    return lam[0]*f1(x) + lam[1]*f2(x) + lam[2]*f3(x)

A = np.zeros([n,m])
A[:,0] = f1(x)
A[:,1] = f2(x)
A[:,2] = f3(x)

print('A= ', A, '\n')
# a) Ohne QR 
lam = np.linalg.solve(A.T @ A, A.T @ y)
print('lambda: ', lam, '\n')

# a) mit QR

[Q,R] = np.linalg.qr(A)
print('Q: ', Q, '\n')
print ('R: ', R, '\n')
lam_qr = np.linalg.solve(R, Q.T @ y)

print('QR-Lösung: ', lam_qr, '\n')

# plt.plot(x,y)

# plt.plot(x,f(x,lam))
# plt.plot(x,f(x,lam_qr),'--')

# plt.legend(["Messpunkte","Regression", "Regression_QR"])

# plt.xlabel("[C°]")
# plt.ylabel("[g/l]")


# plt.show()

# b)
kond_normgl = np.linalg.cond(A.T @ A, np.inf)
print('Konditionszahl Normalgleichung: ', kond_normgl, '\n')

kond_r = np.linalg.cond(R,np.inf)
print('Konditionszahl R: ',kond_r, '\n')

diff_cond = kond_normgl - kond_r

print('Differenz der beiden Konditionszahlen beträgt: ', diff_cond, '\n')

# R ist um 154496188.02927807 besser konditioniert als A.T@A

# c)
poly = np.polyfit(x-x[0],y,len(x)-1)
val = np.polyval(poly, x-x[0])
plt.plot(x,y)

plt.plot(x,f(x,lam))
plt.plot(x,f(x,lam_qr),'--')
plt.plot(x,val)
plt.legend(["Messpunkte","Regression", "Regression_QR", "Polyfit"])

plt.xlabel("[C°]")
plt.ylabel("[g/l]")

plt.show()

# d) Fehler in Polyfit ist am kleinsten
def E(f,y): 
    diff = (y - f)
    return diff.T @ diff

print('Grösse Fehlerfunktional in Normalgleicung: ', E(f(x,lam),y), '\n')       
print('Grösse Fehlerfunktional in QR: ', E(f(x,lam_qr),y), '\n')          
print('Grösse Fehlerfunktional in Polyfit: ', E(val,y), '\n') 
