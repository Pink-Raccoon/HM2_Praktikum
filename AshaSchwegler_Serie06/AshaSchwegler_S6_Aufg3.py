# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 21:29:02 2022

@author: Asha
"""
import numpy as np
import matplotlib.pyplot as plt
data=np.array([
    [1971, 2250.],
    [1972, 2500.],
    [1974, 5000.],
    [1978, 29000.],
    [1982, 120000.],
    [1985, 275000.],
    [1989, 1180000.],
    [1989, 1180000.],
    [1993, 3100000.],
    [1997, 7500000.],
    [1999, 24000000.],
    [2000, 42000000.],
    [2002, 220000000.],
    [2003, 410000000.],   
    ])


#3.1

[n,m] = np.shape(data)
y = data[:,m-1]
y_neu= np.log10(y)
x =data[:,m-2]
x_neu = x-1970
print('x: ',x, '\n')
A = np.zeros([n,m])

print('Ansatz für lineare Regression: f(x) = ax + b = a * f1(x) + b * f2(x) mit f1(x) = x, f2(x) = 1')
print('Minimiere das Fehlerfunktional E(f)(a, b) = ∑[i = 1 .. n](yi - (a*xi + b))^2   (Quadrierte Differenz zwischen den Messwerten yi und den Schätzwerten von f(x))')
print('Die partiellen Ableitungen des Fehlerfunktionals nach a und nach b liefern zwei Gleichungen, als LGS Ax = r')
print('⎡ ∑xi^2   ∑xi ⎤   ⎡ a ⎤   ⎡ ∑xi*yi ⎤\n' +
      '⎢              ⎥ * ⎢   ⎥ = ⎢        ⎥\n' +
      '⎣ ∑xi     n   ⎦    ⎣ b ⎦   ⎣ ∑yi    ⎦\n')


A = np.array([
    [np.sum(x_neu ** 2), np.sum(x_neu)],
    [np.sum(x_neu), x_neu.shape[0]]
])

r = np.array([np.sum(x_neu * y_neu), np.sum(y_neu)])

[a,b] = np.linalg.solve(A,r)

print('a = {}, b = {}'.format(a, b))
print('Die gesuchte Ausgleichsgerade ist also f(x) = {}x + {}'.format(a, b))

def Ausgleichgerade(x):
    return b + (x)*a
#3.2
print('Im Jahre 2015 werden ',Ausgleichgerade(2015-1970)**10,' Stücke von Z13 erscheinen')

plt.figure(1)
plt.grid()
plt.plot(x,Ausgleichgerade(x_neu),zorder=0, label='Regression')
plt.xlim(1971,2020)
plt.ylim(3,13)
plt.scatter(x, y_neu, marker='x', color='r',  label='measured')
plt.scatter(2015, Ausgleichgerade(2015-1970), marker='X', color='fuchsia', label='Extrapolated')

plt.legend()

plt.show()


#3.3

'''
theta_1 = b = höhe und theta_b = a = Steigung_ hier 3.13 
'''