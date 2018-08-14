# -*- coding: utf-8 -*-
"""
Created on Thu May 31 12:13:27 2018

@author: wxj
"""


from __future__ import print_function
import os, sys, re, glob
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
#import Image

#https://matplotlib.org/1.5.1/users/customizing.html
label_size = 24
text_size =24
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.size'] = 24
mpl.rcParams['legend.fontsize'] = 24
mpl.rcParams['figure.figsize']=8,6
mpl.rcParams['figure.dpi']=150
mpl.rcParams['savefig.dpi']=150
mpl.rcParams['savefig.jpeg_quality']=95
mpl.rcParams['savefig.format']='png'

mpl.rcParams['figure.autolayout'] = False  # When True, automatically adjust subplot
                            # parameters to make the plot fit the figure

mpl.rcParams['figure.subplot.left']  = 0.18  # the left side of the subplots of the figure
mpl.rcParams['figure.subplot.right']  = 0.85    # the right side of the subplots of the figure
mpl.rcParams['figure.subplot.bottom']  = 0.15   # the bottom of the subplots of the figure
mpl.rcParams['figure.subplot.top']  = 0.85     # the top of the subplots of the figure
mpl.rcParams['figure.subplot.wspace']  = 0.2    # the amount of width reserved for blank space between subplots
mpl.rcParams['figure.subplot.hspace']  = 0.2    # the amount of height reserved for white space between subplots

