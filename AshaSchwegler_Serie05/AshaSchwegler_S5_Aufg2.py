# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 20:53:48 2022

@author: Asha
"""


import numpy as np
import matplotlib.pyplot as plt

def AshaSchwegler_S5_Aufg2(x,y,xx):
    n = len(x)-1  # n = 3 
    yy = []
    a = y
    h = [] 
    c = []
    z = []
    b = []
    d = []
          
    for i in range(n):
        h.append(x[i+1]-x[i])
    #print('h: ', h)
    A_len = len(h)-1 # 2
    A = np.zeros((A_len,A_len))

    for i in range(A_len):
        if (i != 0): # 1 ( Zeile 2)
            A[i][i-1] = h[i]            
        if (i != A_len-1): # 0 (Zeile 1)
            A[i][i+1] = h[i+1]
        A[i][i] = 2*(h[i]+h[i+1])
            
    
   # print('A:', A)
    for i in range(n-1):
        z.append(3*((y[i+2]-y[i+1])/h[i+1]-(y[i+1]-y[i])/h[i])) 
      
   #print('z:',z)
    tmp = np.linalg.solve(A,z)
    print('tmp: ', tmp)
    c.append(0)

    for i in range(len(tmp)):
        c.append(tmp[i])
    c.append(0)
    for i in range(n):
        b.append((y[i+1]-y[i])/h[i]-h[i]/3*(c[i+1]+2*c[i]))
        d.append(1/(3*h[i])*(c[i+1]-c[i]))
    j = 0
    for i in range(n):
        while j < len(xx) and xx[j] <= x[i+1]:
            yy.append(d[i]*(xx[j]-x[i])**3+c[i]*(xx[j]-x[i])**2+b[i]*(xx[j]-x[i])+a[i])
            
            j+=1
    print("n:",n)        
    print("a:",a)   
    print("b:",b)
    print("c:",c)  
    print("d:",d)        
    plt.plot(xx,yy)
    plt.show()
            
x = [4,6,8,10]
y = [6,3,9,0]   

xx = np.arange(min(x),max(x),0.1)   
AshaSchwegler_S5_Aufg2(x,y,xx)



    