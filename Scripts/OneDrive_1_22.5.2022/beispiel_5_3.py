#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 08:14:14 2022

@author: miec
"""

# Darstellungsformen
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x,y):
    return x**2 + y**2

def g(x,y):
    return 2*x + 4*y - 5

x = np.linspace(-1,3)
y = np.linspace(0,4)
X, Y = np.meshgrid(x,y)

u = np.full(np.shape(y),1)
v = np.full(np.shape(x),2)



fig = plt.figure(0)
plt.plot(x, f(x,2))
plt.plot(x, g(x,2))
plt.ylim([0,10])
plt.show()

fig = plt.figure(1)
plt.plot(y, f(1,y))
plt.plot(y, g(1,y))
plt.ylim([0,10])
plt.show()

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.plot(x,v,f(x,v),color='blue', linewidth=2)
ax.plot(u,y,f(u,y),color='blue', linewidth=2)
ax.plot_surface(X,Y,f(X,Y), color='cyan', alpha=0.5)
ax.set_zlim([0,10])
plt.show()

fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
ax.plot(x,v,f(x,v),color='blue', linewidth=2)
ax.plot(u,y,f(u,y),color='blue', linewidth=2)
ax.plot_surface(X,Y,f(X,Y), color='cyan', alpha=0.5)
ax.plot(x,v,g(x,v),color='red', linewidth=2)
ax.plot(u,y,g(u,y),color='red', linewidth=2)
ax.plot_surface(X,Y,g(X,Y), color='orange', alpha=0.5)
ax.set_zlim([0,10])
plt.show()