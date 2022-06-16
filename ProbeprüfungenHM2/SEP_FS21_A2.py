# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 10:18:56 2022

@author: ashas
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.array([0,1,2,3,4,5],dtype=np.float64)
y = np.array([0.54,0.44,0.28,0.18,0.12,0.08],dtype=np.float64)

n = len(x) #Anz. Gleichungen
m = 5 # a,b,c,d,e Anzahl Unbekannten

f1= x**4
f2=x**3
f3=x**2
f4= x
f5 = [1 for xi in x]

def f(x, lam):
    return lam[0]*f1(x) + lam[1]*f2(x) + lam[2]*f3(x) + lam[3]*f4(x) + lam[4]*f5(x)


print('Ansatz: f(x) = a*x^2 + b*x + c')
print('\tf1(x) = x^2\n\tf2(x) = x\n\tf3(x) = 1')



print('\nKonstruiere die Matrix A, so dass in Spalte 1 die Werte der Funktion f1(x) ausgewertet für alle xi stehen, usw. für die weiteren Spalten mit f2(x) und f3(x)')

A = np.array([f1, f2, f3, f4, f5]).T
print('A = \n{}'.format(A))

print('\nFühre für A eine QR-Zerlegung durch, Q * R = A.')
Q, R = np.linalg.qr(A)
print('Q = \n{}\nR = {}'.format(Q, R))

print('\nLöse das LGS R * λ = QT * y')

coeff_direct = np.linalg.solve(A.T @ A, A.T @ y)
coeff_qr = np.linalg.solve(R, Q.T @ y)
coeff_polyfit = np.polyfit(x, y, A.shape[1] - 1)

error_direct = np.linalg.norm(y - A @ coeff_direct, 2) ** 2
error_qr = np.linalg.norm(y - A @ coeff_qr, 2) ** 2
error_polyfit = np.linalg.norm(y - A @ coeff_polyfit, 2) ** 2

print('\n\nMit direktem Lösen von AT * A * λ = AT * y: λ = ' + str(coeff_direct))
print('Mit QR-Zerlegung: λ = ' + str(coeff_qr))
print('Koeffizienten wenn mit numpy polyfit Grad 4 gelöst: λ = ' + str(coeff_polyfit))

print('\nKonditionszahl von AT * A = ' + str(np.linalg.cond(A.T @ A, np.inf)))
print('Konditionszahl von R = ' + str(np.linalg.cond(R, np.inf)))

print('\nFehler mit direktem Lösen = ' + str(error_direct))
print('Fehler mit QR = ' + str(error_qr))
print('Fehler mit numpy polyfit = ' + str(error_polyfit))

print('Ansatz für lineare Regression: f(x) = ax + b = a * f1(x) + b * f2(x) mit f1(x) = x, f2(x) = 1')
print('Minimiere das Fehlerfunktional E(f)(a, b) = ∑[i = 1 .. n](yi - (a*xi + b))^2   (Quadrierte Differenz zwischen den Messwerten yi und den Schätzwerten von f(x))')
print('Die partiellen Ableitungen des Fehlerfunktionals nach a und nach b liefern zwei Gleichungen, als LGS Ax = r')
print('⎡ ∑xi^2   ∑xi ⎤   ⎡ a ⎤   ⎡ ∑xi*yi ⎤\n' +
      '⎢              ⎥ * ⎢   ⎥ = ⎢        ⎥\n' +
      '⎣ ∑xi     n   ⎦    ⎣ b ⎦   ⎣ ∑yi    ⎦\n')

A = np.array([
    [np.sum(x ** 2), np.sum(x)],
    [np.sum(x), x.shape[0]]
])

r = np.array([np.sum(x * y), np.sum(y)])

print('A = \n{}'.format(A))
print('r = {}'.format(r))

print('LGS wird gelöst...\n')

ab = np.linalg.solve(A, r)
a = ab[0]
b = ab[1]

xx = np.arange(x[0], x[-1], (x[-1] - x[0]) / 10000)  # Plot-X-Werte
yy = a * xx + b
print('a = {}, b = {}'.format(a, b))
print('Die gesuchte Ausgleichsgerade ist also f(x) = {}x + {}'.format(a, b))






plt.figure(1)
plt.grid()
plt.title('Ausgleich')
plt.scatter(x,y,marker='X',label= 'Messpunkte')
plt.plot(xx, np.polyval(coeff_direct, xx), zorder=0, label='direct solve')
plt.plot(xx,yy,label='lineare Regression')
plt.show()
plt.legend(['Messpunkte','direct solve','lineare Regression'])
plt.show()