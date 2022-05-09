# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:49:25 2022

@author: ashas
"""

#Richtungsvektoren normieren statt richtung 1 oben etc.
import numpy as np
import matplotlib.pyplot as plt


def AshaSchwegler_S11_Aufg1(f,xmin,xmax,ymin,ymax,hx,hy):
    x = np.arange(xmin, xmax, step=hx, dtype=np.float64)
    y = np.arange(ymin, ymax, step=hy, dtype=np.float64)
    [x_grid, y_grid] = np.meshgrid(x, y)

    dy = f(x_grid, y_grid)
    dx = np.full((dy.shape[0], dy.shape[1]), 1, dtype=np.float64)
    richtungsVektor = np.array([1,f(x_grid, y_grid)], dtype= 'float')
    vektor_normieren = np.linalg.norm(richtungsVektor)
    vektor_laenge_1 = 1/vektor_normieren
    plt.quiver(x_grid, y_grid, dx, dy,*vektor_normieren)
    plt.show()

