# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 13:08:11 2022

@author: Asha
"""

import numpy as np
import matplotlib.pyplot as plt 


x = [1981.0,1984,1989,1993,1997,2000,2001,2003,2004,2010]
y = [0.5,8.2,15,22.9,36.6,51,56.3,61.8,65,76.6]

X = np.shape(x)
Y = np.shape(y)  
    

z = np.polyfit(X,Y,9)
#print(z)


foo= np.polyval(X,Y)
print(foo)

plt.plot(X,foo)

plt.ylim(-100,250) 
plt.xlim(1975,2020)
plt.scatter(x,y)    

plt.show() 