# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 11:13:19 2022

@author: ashas
"""
import numpy as np
import matplotlib.pyplot as plt


a = 1
b= 6
h= 0.01
n= 500
y0 = 5

x = np.linspace(a,b,n+1)

def f_exakt(t):
    return (t/2) + (9/2*t)

def f(t,y):
    return 1 -(y/t)

def AshaSchwegler_S11_Aufg1(f,xmin,xmax,ymin,ymax,hx,hy):
    x = np.linspace(xmin, xmax, int((xmax - xmin)/hx))
    y = np.linspace(ymin, ymax, int((ymax - ymin)/hy))
    x, y = np.meshgrid(x, y)
    vx = np.ones_like(x)
    vy = f(x,y)
    # normieren
    v = np.sqrt(vx**2+vy**2)
    vx = vx / v
    vy = vy / v
    plt.quiver(x,y,vx,vy,width=0.003,color='pink')
    


def AshaSchwegler_S12_Aufg2(f,a,b,n,y0):
    y = np.full(x.shape[0],0,dtype= np.float64)
    y[0] = y0
    h_halb = h/2
    
    for i in range(n):
        k1 = f(x[i],y[i])
        k2 = f(x[i]+h_halb, y[i]+h_halb * k1)
        k3 = f(x[i]+h_halb, y[i]+h_halb * k2)
        k4 = f(x[i]+h_halb, y[i]+h * k3)
        
        y[i+1] = y[i] + h * (1/6)*(k1 + 2*k2 + 2*k3 + k4)
        print('y{} = '.format(i),y[i],'\n')
        
    return y


#b
c = np.array([0,0.25,0.5,0.75], dtype=np.float64)
a_kut = np.array([[0,0,0,0],
                  [0.5,0,0,0],
                  [0,0,0.75,0],
                  [0,0,1,0]
                  ], dtype=np.float64)
b_kut = np.array([1/10,2/10,3/10,4/10], dtype=np.float64)
s = 4

def kutta_allgemein(f,x,a_kut,b_kut,c,h,y0,s):
    y = np.full(x.shape[0],0,dtype= np.float64)
    y[0] = y0
    
    for i in range(x.shape[0]-1):
        k = np.full(s,0,dtype=np.float64)
        for n in range(s):
            k[n] = f(x[i]+c[n]*h,y[i]+h*np.sum([a_kut[n][m]*k[m] for m in range(1,n-1)]))
            y[i+1] = y[i] + h*np.sum([b_kut[n]*k[n] for n in range(1,s)])
            
    return y













y_kutta_vierstufig = AshaSchwegler_S12_Aufg2(f, a, b, n, y0)
y_kutta_custom = kutta_allgemein(f,x,a_kut,b_kut,c,h,y0,s)


plt.figure(1)
plt.grid()
plt.plot(x,f_exakt(x), label = 'Exakt')
plt.plot(x,y_kutta_vierstufig,label = 'Kutta')
plt.plot(x,y_kutta_custom, label= 'Kutta custom')

AshaSchwegler_S11_Aufg1(f,0,6, 2, 30, 0.5, 2)

plt.legend()
plt.show()