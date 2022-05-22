#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 09:01:56 2021

@author: miec
"""

import numpy as np
import matplotlib.pyplot as plt

def f(x,y): return x**2 + 0.1*y**2

x, y = np.meshgrid(np.linspace(-2,2,9), np.linspace(-1,2,7))
n = np.sqrt(1 + f(x,y)**2)
u = 1/n
v = f(x,y)/n
plt.quiver(x,y,u,v, color='blue', width=0.004)

def sol(t): return -10*t**2 - 200*t - 2000 + 1722.5*np.exp(0.05*(2*t+3))
x_ = np.linspace(-1.5,2)
plt.plot(x_, sol(x_), color='red')
plt.ylim([-1.25,2.25])

plt.show()