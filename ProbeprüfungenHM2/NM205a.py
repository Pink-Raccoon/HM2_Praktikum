# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:09:27 2022

@author: ashas
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.array([500,1000,1500,2500,3500,4000,4500,5000,5250,5500])
y= np.array([10.5,49.2,72.1,85.4,113,121,112,80.2,61.1,13.8])





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
plt.plot(x,f(x,lam_qr),color='r')
plt.plot(x,val,color='orange', label='Polyfit')
plt.xlabel('d')
plt.ylabel('P')
plt.legend()
plt.show()



#%%

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

lam0 = np.array([1, 2, 2, 1,2],dtype=np.float64)
tol = 1e-5

p = sp.symbols('p0 p1 p2 p3 p4')
# oder eleganter: 
# p = sp.symbols('p:{n:d}'.format(n=lam.size))
p



def f(x, p):
    return p[0]*f1(x) + p[1]*f2(x) + p[2]*f3(x) + p[3]*f4(x) + p[4]*f5(x)

g = sp.Matrix([y[k]-f(x[k],p) for k in range(len(x))])

Dg = g.jacobian(p)

g = sp.lambdify([p], g, 'numpy')
Dg = sp.lambdify([p], Dg, 'numpy')
g(lam0)
Dg(lam0)
k=0
lam=np.copy(lam0)
[Q,R] = np.linalg.qr(Dg(lam))
delta = np.linalg.solve(R,-Q.T @ g(lam)).flatten()  # Achtung: flatten() braucht es, um aus dem Spaltenvektor delta wieder eine Liste zu machen, da g hier nicht mit Spaltenvektoren als Input umgehen kann
lam = lam+delta
increment = np.linalg.norm(delta)

err_func0 = np.linalg.norm(g(lam0))**2
err_func = np.linalg.norm(g(lam))**2


def gauss_newton(g, Dg, lam0, tol, max_iter):
    k=0
    lam=np.copy(lam0)
    increment = tol+1
    err_func = np.linalg.norm(g(lam))**2
    
    while tol <= err_func and max_iter<k : #Hier kommt Ihre Abbruchbedingung, die tol und max_iter berücksichtigen muss# 

        # QR-Zerlegung von Dg(lam) und delta als Lösung des lin. Gleichungssystems
        [Q,R] = np.linalg.qr(Dg(lam).flatten())
        [delta] = np.linalg.solve(R,-Q.T @ g(lam)).flatten()   
        
       
        # Update des Vektors Lambda        
        lam += delta.flatten()
        err_func = np.linalg.norm(g(lam)).flatten()**2
        increment = np.linalg.norm(delta)
        print('Iteration: ',k)
        print('lambda = ',lam)
        print('Inkrement = ',increment)
        print('Fehlerfunktional =', err_func)
    return(lam,k)

tol = 1e-5
max_iter = 30
[lam_without,n] = gauss_newton(g, Dg, lam0, tol, max_iter)

t = sp.symbols('t')
F = f(t,lam_without)
F = sp.lambdify([t],F,'numpy')
t = np.linspace(x.min(),x.max())

def f_max(x):
    return lam[0] + 2*lam[1]*x + 3*lam[2]*x**2+ 4*lam[3]*x**3 


plt.figure(2)
plt.title('Newton-Gauss')
plt.plot(x,y,'o')
plt.plot(t,F(t))
plt.xlabel('x')
plt.ylabel('y')
plt.show()  


# print('maximum =',g(lam)+Dg(lam0)*(lam-lam0) )
print('maximum =',np.linalg.solve(f_max(x.T),0) )