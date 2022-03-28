# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:25:32 2022

@author: Asha
"""
import numpy as np
import matplotlib.pyplot as plt
from AshaSchwegler_S5_Aufg2 import AshaSchwegler_S5_Aufg2
from scipy import interpolate

x = np.array([1900,1910,1920,1930,1940,1950,1960,1970,1980,1990,2000,2010])
y = np.array([75.995,91.972,105.711,123.203,131.669,150.697,179.323,203.212,226.505,249.633,281.422,308.745])

xx = np.arange(min(x),max(x),0.1)


#a
AshaSchwegler_S5_Aufg2(x,y,xx)


#b
nat_cubSpline = interpolate.CubicSpline(x,y)
plt.plot(xx,nat_cubSpline(xx), "--")


#c
p = np.polyfit(x-min(x),y,len(x)-1)
val = np.polyval(p,xx-min(x))
plt.plot(xx,val,"--")
plt.legend(["spline","cubicspline"])