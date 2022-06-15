# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 08:43:04 2022

@author: ashas
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy

mol = 6.022 * 10**23

R = 8.31

[V_p,T_p] = np.meshgrid(np.linspace(0,0.2), np.linspace(0,1*np.exp(4)))

def p(V_p,T_p):
    p = (R*T_p)/V_p
    return p

[p_v,T_v] = np.meshgrid(np.linspace(1*np.exp(4),1*np.exp(5)), np.linspace(0,1*np.exp(4)))   

def V(p_v,T_v):
    V = (R*T_v)/p_v
    return V

[p_t, V_t] = np.meshgrid(np.linspace(1*np.exp(4),1*np.exp(6)), np.linspace(0,10))

def T(p_t, V_t):
    T = (p_t * V_t) / R
    return T


z = p(V_p,T_p)

z2 = T(p_t, V_t)

fig = plt.figure(0)
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(V_p,T_p,z, rstride=5, cstride=5)


plt.title('Gitter')
ax.set_xlabel('V_p')
ax.set_ylabel('T_p')
ax.set_zlabel('z')

plt.show()  

z1 = V(p_v,T_v)
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(p_v,T_v,z1, rstride=5, cstride=5)


plt.title('Gitter')
ax.set_xlabel('p_v')
ax.set_ylabel('T_v')
ax.set_zlabel('z1')

plt.show() 
