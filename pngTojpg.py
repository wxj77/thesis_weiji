# -*- coding: utf-8 -*-
"""
Created on Wed May 30 18:10:20 2018

@author: wei

this create convert all the figures in a folder from XXX.png to XXX.jpg. 

how to run:
    python pngTojpg.py [datadir] 

"""
from __future__ import print_function
import os, sys, re, glob
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
try:
    import Image
except:
    from PIL import Image

#https://matplotlib.org/1.5.1/users/customizing.html
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.size'] = 15
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['figure.figsize']=8,6
mpl.rcParams['figure.dpi']=150
mpl.rcParams['savefig.dpi']=150
mpl.rcParams['savefig.jpeg_quality']=95
mpl.rcParams['savefig.format']='png'

datadir= (sys.argv[1]) if len(sys.argv) > 1 else "./"
figure_list=[]
figure_list+=glob.glob(datadir+'./*.png')
figure_list+=glob.glob(datadir+'./*/*.png')
figure_list+=glob.glob(datadir+'./*/*/*.png')
figure_list+=glob.glob(datadir+'./*/*/*/*.png')
figure_list+=glob.glob(datadir+'./*/*/*/*/*.png')
#
for ff in figure_list:
    gg=ff[:-4]+".jpg"
    print (ff,gg)
    try:
    	Image.open(ff).save(gg,'JPEG')
    except:
        Image.open(ff).convert('RGB').save(gg,'JPEG')
