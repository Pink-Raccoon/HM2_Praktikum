# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 14:00:31 2022

@author: ashas
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

c = 1


def w(x,t):
    return np.sin(x+c*t)

def v(x,t):
    return np.sin(x+c*t)+np.cos(2*x+2*c*t)

[x,t] = np.meshgrid(np.linspace (0,5), np.linspace(0,5))



z = w(x,t)
z1 = v(x,t)



fig = plt.figure(0)
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x,t,z, rstride=10, cstride=10)


plt.title('Gitter')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('z')


fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x,t,z1, rstride=10, cstride=10)


plt.title('Gitter')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('z1')

plt.show() 