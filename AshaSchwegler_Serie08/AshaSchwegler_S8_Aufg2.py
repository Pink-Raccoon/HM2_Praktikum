# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 19:06:18 2022

@author: Asha
"""


import math

m = 10
v = 5
v0 = 20



def integrate(v):
    foo = (v**(3/2))
    to_int = -10*v**(-(3/2)+1)
    integ = (1/(-(3/2)+1)) * to_int
    return integ

Exakter_Wert = abs(integrate(v) - integrate(v0))

def f(x,m):
    return m/(-x * math.sqrt(x))


#a
def sumRechteck(a,b,n,m):
    h = float((b-a)/n)
    result = 0
    for i in range (n):
       # print("range sumRechteck: ", i, '\n')
        result += f(a+i*h+(h/2),m)
    result *= h
    return result
    
    
    
print("Summierte Rechtecksregel: t= ", sumRechteck(20,5,5,10), '\n')
print('Absolute Fehler = ', abs(sumRechteck(20,5,5,10)-Exakter_Wert),'\n')

#b

def sumTrapez(a,b,n,m):
    h = float((b-a)/n)
    result = 0
    for i in range (1,n):
       # print("range sumTrapez: ", i, '\n')
        result += f(a+i*h,m)
    result += (f(a,m)+f(b,m))/2
    result *= h
    return result


print("Summierte Trapezregel: t= ", sumTrapez(20,5,5,10), '\n')
print('Absolute Fehler = ', abs(sumTrapez(20,5,5,10)-Exakter_Wert),'\n')


#c

def sumSimpson(a,b,n,m):
    h = float((b-a)/n)
    result1 = 0
    result2 = 0
    for i in range(1,n):
       # print("range sumSimpson1: ", i, '\n')
        result1 += f(a+i*h,m)
    result1 += (1/2)*f(a,m)
        
    for i in range(1,n+1):
       # print("range sumSimpson2: ", i, '\n')
        result2 += f((a+(i-1)*h+a+i*h)/2+f(b,m),m)
    result2 *= 2
    return (h/3)*(result1+result2)

print("Summierte Simpsonregel: t= ", sumSimpson(20,5,5,10), '\n')    
print('Absolute Fehler = ', abs(sumSimpson(20,5,5,10)-Exakter_Wert),'\n')

print('Exakter Wert=', Exakter_Wert)
