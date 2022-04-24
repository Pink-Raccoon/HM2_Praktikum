# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 12:26:13 2022

@author: Asha
"""


def AshaSchwegler_S8_Aufg3a(x,y):
    n = len(x)
    sumArray_x= []
    sumArray_y = 0
    Tf_neq = 0
    for i in range(n): 
        if i < n-1:
            sumArray_x.append(x[i+1]-x[i])
    for j in range(n): 
        if j < n-1:        
            sumArray_y=(y[j]+y[j+1])/2
            Tf_neq += sumArray_y * sumArray_x[j]        
    return Tf_neq





        
