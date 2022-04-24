# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 19:06:18 2022

@author: Asha
"""


import math




def f(x,m):
    return m/(-x * math.sqrt(x))


#a
def sumRechteck(a,b,n):
    h = float((b-a)/n)
    result = 0
    for i in range (n-1):
        result += f(a+i*h+(h/2),10)
    result *= h
    return result
    
    
    
print("Summierte Rechtecksregel: t= ", sumRechteck(20,5,5), '\n')


#b

def sumTrapez(a,b,n):
    h = float((b-a)/n)
    result = 0
    for i in range (1,n-1):
        result += f(a+i*h,10)
    result += (f(a,10)+f(b,10))/2
    result *= h
    return result

# print("f(a):", f(20,10), '\n')
# print("f(b):", f(5,10), '\n')
print("Summierte Trapezregel: t= ", sumTrapez(20,5,5), '\n')


#c

def sumSimpson(a,b,n):
    h = float((b-a)/n)
    result1 = 0
    result2 = 0
    for i in range(1,n-1):
        result1 += f(a+i*h,10)
    result1 += (1/2)*f(a,10)
        
    for i in range(1,n):
        result2 += f((a+(i-1)*h+a+i*h)/2+f(b,10),10)
    result2 *= 2
    return (h/3)*(result1+result2)

print("Summierte Simpsonregel: t= ", sumSimpson(20,5,5), '\n')    



