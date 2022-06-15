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
yy = AshaSchwegler_S5_Aufg2(x,y,xx)
plt.figure(1)
plt.grid()
plt.plot(xx, yy)
plt.title('kubischen Splinefunktion')
plt.scatter(x,y, marker='x', color='r', zorder=1, label='measured')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#b
plt.figure(2)
plt.grid()
nat_cubSpline = interpolate.CubicSpline(x,y)
plt.plot(xx,nat_cubSpline(xx), "--")
plt.plot(xx,yy,"--")
plt.scatter(x,y, marker='x', color='r', zorder=1, label='measured')
plt.legend(["nat.spline","cubicspline"])
plt.show()


#c
plt.figure(3)
plt.grid()
poly = np.polyfit(x-min(x),y,len(x)-1)
val = np.polyval(poly,xx-min(x))
plt.plot(xx,val,"polyfit")
plt.plot(xx,nat_cubSpline(xx), "nat.spline")

plt.legend(["spline","cubicspline"])
plt.show()
