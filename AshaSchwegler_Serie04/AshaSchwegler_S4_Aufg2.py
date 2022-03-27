# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 19:59:50 2022

@author: Asha
"""

import numpy as np


def lagrange(x,y,x_int):
    y_int = 0
    i_int = 1
    i = 0
    j = 1
    for i in range(len(y)):
        for j in range(len(x)):
            if (x[i] != x[j]):
                i_int *= (x_int-x[j])/(x[i]-x[j])
        print("l",i,"=",i_int)
        y_int += i_int * y[i]
        i_int = 1
    return y_int



x = [0,2500,5000,10000]
y = [1013,747,540,226]
x_int = 3750

y_int = lagrange(x,y,x_int)
print(y_int)


