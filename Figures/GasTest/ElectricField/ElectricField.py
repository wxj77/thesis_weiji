#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 20:56:51 2018

@author: weiji
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 09:58:08 2018

@author: wxj
"""


import os, sys
wd='/Users/weiji/Google Drive/gastest/' #'/home/wxj/gastest/'
workdir=wd + '/ElectricField/' #'test'#
try:
    os.makedirs(workdir)
except:
    i=1
#sys.path.insert(0, "/home/wxj/.2go")
#sys.path.insert(0, "/home/wxj/.x2go")
#sys.path.insert(0, "/home/wxj/.2go/PythonScripts/RunSetups")
#sys.path.insert(0, "/home/wxj/.2go/PythonScripts")
#sys.path.insert(0, "/home/wxj/gtest_result/")
#sys.path.insert(0, "/home/wxj/.2go")
#sys.path.insert(0, "/home/wxj/.x2go")
#sys.path.insert(0, "/home/wxj/.2go/PythonScripts/RunSetups")
#sys.path.insert(0, "/home/wxj/.2go/PythonScripts")
#sys.path.insert(0, "/home/wxj/gtest_result/")

sys.path.insert(0, wd+"/obsolete/PythonScripts/RunSetups/")
sys.path.insert(0, wd+"/obsolete/PythonScripts/")
sys.path.insert(0, wd+"/obsolete/PythonScript/")
sys.path.insert(0, wd+"/xenonProperties/")


import numpy as np

import pickle
import matplotlib.pyplot as plt

from multiprocessing import Pool
from matplotlib.ticker import Formatter    # to convert xaxis label to dates
from matplotlib.colors import LogNorm
from matplotlib import dates

import cPickle as cp
import matplotlib.colors as colors
import math

from scipy import interpolate
import scipy.optimize as opt
from matplotlib import colors as mcolors
from cycler import cycler
from matplotlib.patches import Ellipse
# For DB access
#import MySQLdb
#import ignition_db_interface as idbi

from matplotlib import dates
import matplotlib.gridspec as gridspec
import scipy.stats as stats
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from color import *
from plotStyle import *


'''

config1 (top surface, bottom surface, drift)
0.57,3.47,6.37,9.27,12.17,15.07,17.97,20.88,23.78,26.68,29.58,32.48,35.38,38.28,41.19,44.09,46.99,
-1.35,6.33,14.02,21.70,29.38,37.07,44.75,52.44,60.12,67.80,75.49,83.17,90.85,98.54,106.22,113.91,121.59,
0.01,0.69,1.38,2.07,2.75,3.44,4.13,4.81,5.50,6.18,6.87,7.56,8.24,8.93,9.62,10.30,10.99,

config2 (top surface, bottom surface, drift)
0.73,4.46,8.19,11.92,15.65,19.38,23.11,26.84,30.57,34.30,38.03,41.76,45.49,49.22,52.95,56.68,60.41,
-1.40,6.04,13.48,20.92,28.36,35.80,43.24,50.68,58.12,65.56,73.00,80.44,87.88,95.32,102.77,110.21,117.65,
0.00,0.67,1.33,1.99,2.66,3.32,3.98,4.64,5.31,5.97,6.63,7.30,7.96,8.62,9.29,9.95,10.61, (edited)


'''
dv_l = np.arange(0,17,1)
t_config1 = [
0.57,3.47,6.37,9.27,12.17,15.07,17.97,20.88,23.78,26.68,29.58,32.48,35.38,38.28,41.19,44.09,46.99,
]
b_config1=[
-1.35,6.33,14.02,21.70,29.38,37.07,44.75,52.44,60.12,67.80,75.49,83.17,90.85,98.54,106.22,113.91,121.59,

        ]
drift_config1 =[
     0.01,0.69,1.38,2.07,2.75,3.44,4.13,4.81,5.50,6.18,6.87,7.56,8.24,8.93,9.62,10.30,10.99,   
        ]
t_config2 =[
0.73,4.46,8.19,11.92,15.65,19.38,23.11,26.84,30.57,34.30,38.03,41.76,45.49,49.22,52.95,56.68,60.41,
]
b_config2=[
 -1.40,6.04,13.48,20.92,28.36,35.80,43.24,50.68,58.12,65.56,73.00,80.44,87.88,95.32,102.77,110.21,117.65,
]

drift_config2 =[
        0.00,0.67,1.33,1.99,2.66,3.32,3.98,4.64,5.31,5.97,6.63,7.30,7.96,8.62,9.29,9.95,10.61,
        ]

t_configl =[
1.01,4.76,8.51,12.26,16.02,19.77,23.52,27.27,31.02,34.77,38.53,42.28,46.03,49.78,53.53,57.28,61.04,
]
b_configl=[
-0.05,5.20,10.44,15.68,20.93,26.17,31.41,36.66,41.90,47.14,52.39,57.63,62.87,68.12,73.36,78.61,83.85,
]
drift_configl =[
-0.01,0.85,1.71,2.58,3.44,4.30,5.16,6.02,6.88,7.74,8.60,9.46,10.33,11.19,12.05,12.91,13.77,
       ]


fignum=100

## surface electric field
fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)
ax.plot(dv_l, np.abs(t_config1), color='r', ls='--', label='top grid avg. (grid config. 1)')
ax.plot(dv_l, np.abs(b_config1), color='b', ls='--', label='bottom grid avg. (grid config. 1)')
ax.plot(dv_l, np.abs(t_config2), color='r', ls='-.', label='top grid avg. (grid config. 2)')
ax.plot(dv_l, np.abs(b_config2), color='b', ls='-.', label='bottom grid avg. (grid config. 2)')
#ax.plot(dv_l, np.abs(t_configl), color='darkred', ls='-', label='anode grid avg. (LZ)')
#ax.plot(dv_l, np.abs(b_configl), color='darkblue', ls='-', label='gate grid avg. (LZ)')

ax.set_xlabel('|'+r"$\Delta V_{T-B}$"+'| [kV]')
ax.set_ylabel('surface field [kV cm'+r'$^{-1}$'+']')
ax.set_xlim(0, 16)
ax.set_ylim(0, )
#ax.set_xscale('log')
#ax.set_yscale('log')
ax.grid('on')
box = ax.get_position()
ax.set_position([box.x0+box.width*.0, box.y0+box.height*0.0, box.width*1., box.height*0.7])
ax.legend(loc=3, bbox_to_anchor=(0., 1.1, 1., .102), ncol=1, mode="expand", borderaxespad=0.,fontsize=17)

plt.savefig(workdir+'/SurfaceElectricFieldGas.png')
#plt.close('all')


## drift field
fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)
ax.plot(dv_l, np.abs(drift_config1), color='m', ls='--', label='EL region avg. (grid config. 1)')
ax.plot(dv_l, np.abs(drift_config2), color='purple', ls='-.', label='EL region avg. (grid config. 2)')
#ax.plot(dv_l, np.abs(drift_configl), color='k', ls='-', label='EL region avg. (LZ)')


ax.set_xlabel('|'+r"$\Delta V_{T-B}$"+'| [kV]')
ax.set_ylabel('drift field [kV cm'+r'$^{-1}$'+']')
ax.set_xlim(0, 16)
ax.set_ylim(0, )
#ax.set_xscale('log')
#ax.set_yscale('log')
ax.grid('on')
box = ax.get_position()
ax.set_position([box.x0+box.width*.0, box.y0+box.height*0.0, box.width*1., box.height*0.7])
ax.legend(loc=3, bbox_to_anchor=(0., 1.1, 1., .102), ncol=1, mode="expand", borderaxespad=0.,fontsize=17)

plt.savefig(workdir+'/DriftElectricFieldGas.png')