# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 11:54:05 2022

@author: Asha
"""

import time

def fac(n):
    print(n)
    return 1 if (n < 1) else n * fac(n-1)
  


print(fac(5))