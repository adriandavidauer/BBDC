# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 16:42:35 2018

@author: TSB6EK
"""
import time
from datetime import datetime
import calendar
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



from numpy import genfromtxt
df=genfromtxt('train.csv', delimiter=',',dtype=None)
x=np.array(df[:,0])
x=np.delete(x,[0])
wind_v=df[:,1]
wind_v=np.delete(wind_v,[0])
counter = len(x)
for i in range(counter):
    x_t= time.strptime(x[i], '%Y-%m-%d %H:%M:%S')
    x_tp= time.mktime(x_t)
    x[i]=x_tp
print x
t=np.arange(counter)
print t
print wind_v

plt.plot(t,wind_v)
plt.show()
print "done"