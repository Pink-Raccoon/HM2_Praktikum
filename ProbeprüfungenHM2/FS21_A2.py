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
    return np.array([lam[4]*f1(x) + lam[3]*f2(x) + lam[2]*f3(x) + lam[1]*f4(x) + lam[0]*f5(x)])


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

f1_neu = x**2
f2_neu = [1 for xi in x]

def f_neu(x,lam_neu):
    return (lam_neu[0]*f2_neu+lam_neu[1]*f1_neu)**-1

A_neu = np.array([f1_neu,f2_neu]).T
print('A = \n{}'.format(A_neu))
Q, R = np.linalg.qr(A_neu)
print('Q = \n{}\nR = {}'.format(Q, R))
coeff_lin =np.linalg.solve(A_neu.T @ A_neu, A_neu.T @ y)
print('\n\nMit direktem Lösen von AT * A * λ = AT * y: λ = ' + str(coeff_lin))


error_lin = np.linalg.norm(y - A_neu @ coeff_lin, 2) ** 2
print('Fehler mit lin = ' + str(error_lin))

xx = np.arange(0,5 ,50, dtype=np.float64)  # Plot-X-Werte








plt.figure(1)
plt.grid()
plt.title('Ausgleich')
plt.scatter(x,y,marker='X',label= 'Messpunkte')
plt.plot(xx, f(x,coeff_direct), color='g',zorder=0, label='direct solve')
plt.plot(np.polyval(coeff_lin,xx), label='lin')
plt.legend(['Messpunkte','direct solve','lineare Regression'])
plt.show()