# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:31:49 2022

@author: Asha
"""
import numpy as np
import math

from AshaSchwegler_S8_Aufg3a import AshaSchwegler_S8_Aufg3a

x = np.array([0,800,1200,1400,2000,3000,3400,3600,4000,5000,5500,6370], dtype= 'float')
y = np.array([13000,12900,12700,12000,11650,10600,9900,5500,5600,4750,4500,3300],dtype= 'float')

x_ArrayInMeter = (x*1000)
x_ArrayVolumen = (x_ArrayInMeter**3*4*math.pi)/3




#b
print("Masse der Erde in Kg: ", AshaSchwegler_S8_Aufg3a(x_ArrayVolumen,y), '\n')


# Gemäss Wikipedia Erdmasse = 5,9722 · 10^24 kg

AbsoluterFehler = abs(5.9722*10**24 - AshaSchwegler_S8_Aufg3a(x_ArrayVolumen,y))

print("Absoluter Fehler = ", AbsoluterFehler, '\n')

RelativerFehler = AbsoluterFehler /(5.9722*10**24)

print("Relativer Fehler = ", RelativerFehler, '\n')









