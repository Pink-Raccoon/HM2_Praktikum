# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 13:10:29 2022

@author: ashas
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


g = 9.81


def W(v0,alpha):
    return (v0**2*np.sin(2*alpha))/g


[v0,alpha] = np.meshgrid(np.linspace (0,100), np.linspace(0,180))
z = W(v0,alpha)
fig = plt.figure(0)
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(v0,alpha,z, rstride=5, cstride=5)


plt.title('Gitter')
ax.set_xlabel('v0')
ax.set_ylabel('alpha')
ax.set_zlabel('z')

plt.show()    


fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(v0,alpha,z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.title('Fläche')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()



fig = plt.figure(2)
cont = plt.contour(v0,alpha,z,cmap=cm.coolwarm)
fig.colorbar(cont, shrink=0.5, aspect=5)

plt.title('Höhenlinien')
plt.xlabel('x')
plt.ylabel('y')

plt.show()