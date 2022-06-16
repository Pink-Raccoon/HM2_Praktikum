# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:09:27 2022

@author: ashas
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sy

x = np.array([500,1000,1500,2500,3500,4000,4500,5000,5250,5500],dtype=np.float64)
y= np.array([10.5,49.2,72.1,85.4,113,121,112,80.2,61.1,13.8],dtype=np.float64)





#Da die Messpunkte einen Parabel bilden, ist der Grad 2 am naheliegendsten
#Also P(x) = a*x + b*x**2 + c*x**3+ d*x**4 + e

n = len(x) #Anz. Gleichungen
m = 5 # a,b,c,d,e Anzahl Unbekannten

def f1(x):
    return x**4
def f2(x):
    return x**3
def f3(x):
    return x**2
def f4(x):
    return x
def f5(x):
    return 1

def f(x, lam):
    return lam[0]*f1(x) + lam[1]*f2(x) + lam[2]*f3(x) + lam[3]*f4(x) + lam[4]*f5(x)

# def f(x,a):
#     return a[0]*x + a[1]*x**2 + a[3]*x**3+ a[4]*x**4 + 1

A = np.zeros([n,m]) # Zeilen = anzahl x, Spalten = anzahl koeff.
A[:,0] = f1(x)
A[:,1] = f2(x)
A[:,2] = f3(x)
A[:,3] = f4(x)
A[:,4] = f5(x)

#Normalengleichungssystem = A_TA*lambda = AT_y

[Q,R] = np.linalg.qr(A)
lam_qr = np.linalg.solve(R, Q.T @ y)

poly = np.polyfit(x-x[0],y,len(x)-1)
val = np.polyval(poly, x-x[0])

x_neu = np.linspace(500,5500,5001)
plt.figure(1)
plt.grid()
plt.title('Ausgleich')
plt.scatter(x,y,marker='X',label= 'Messpunkte')
plt.plot(x,f(x,lam_qr),color='r',label='Ausgleich')
plt.plot(x,val,color='orange', label='Polyfit')
plt.xlabel('d')
plt.ylabel('P')
plt.legend()
plt.show()



#%% aufgabae 5c)

x = sy.symbols('x')

"""==================== INPUT ===================="""
f = lam_qr[0]*x**4 + lam_qr[1]*x**3 + lam_qr[2]*x**2 + lam_qr[3]*x + lam_qr[4]
x0 = 20000

max_error = 1e-6
"""==============================================="""

df = sy.diff(f, x)
d2f = sy.diff(df, x)
fl = sy.lambdify(x, f)
dfl = sy.lambdify(x, df)
d2fl = sy.lambdify(x, d2f)

print("f'(x) = " + str(df))

print("Konvergenzbedingung für x0 prüfen:")
d = abs((fl(x0) * d2fl(x0)) / ((dfl(x0)) ** 2))

if d < 1:
    print("Konvergenzbedingung erfüllt!")
else:
    print("Konvergenzbedingung NICHT erfüllt!")

if d < 1:
    xn = [x0]
    print("n = 0: x0 = " + str(x0))

    n = 0

    while n < 1 or abs(xn[n] - xn[n - 1]) > max_error:
        xn.append(xn[n] - dfl(xn[n]) / d2fl(xn[n]))

        n += 1

        print("n = " + str(n) + ": x" + str(n) + " = " + str(xn[n]) + ", Δ = " + str(abs(xn[n] - xn[n - 1])))


    





