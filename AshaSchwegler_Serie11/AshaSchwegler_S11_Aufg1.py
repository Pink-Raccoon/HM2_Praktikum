# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:49:25 2022

@author: ashas
"""

#Richtungsvektoren normieren statt richtung 1 oben etc.
import numpy as np
import matplotlib.pyplot as plt


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


def f(x, y):
    return ((x**2) * 1.0) / y

AshaSchwegler_S11_Aufg1(f, -1, 4, 0.5, 8, 0.5, 0.5)