#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 23:01:57 2018

@author: weiji
"""
#import ROOT as root


import os, sys
wd='/Users/weiji/Google Drive/gastest/' # '/home/wxj/gastest' # '/home/wei/'#
workdir='/Users/weiji/Google Drive/Dropbox/thesis_weiji/Figures/GasTest/GarfieldResults/' #'test'#
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
import matplotlib as mpl

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
#import scipy.stats as stats
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

#from root_numpy.testdata import get_filepath
#from root_numpy import root2array
#from ROOT import TLorentzVector 

from color import *
from plotStyle import *
'''
f = root.TFile(workdir + 'test.root')

myH2 = f.Get("h4")

for entry in myTree[0]:         
     # Now you have acess to the leaves/branches of each entry in the tree, e.g.
     events = entry.events
     break;
     
for event in f.tree:
      print event.TT




nx = myH2.GetNbinsX()
ny = myH2.GetNbinsY()

n = np.zeros([nx,ny])
for ii in range(nx):
    for jj in range(ny):
        n[ii][jj]=myH2.GetBinContent(ii+1,jj+1)

'''


from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 100

mpl.rcParams['font.size'] = 20

fignum=100
A = np.loadtxt(workdir+"OneEvent"+str(fignum)+".txt", skiprows=3)

XX = A[:,2]*10
YY = A[:,3]*10
TY = A[:,6]

XXi= XX[TY==1]
YYi= YY[TY==1]

XXe= XX[TY==4]
YYe= YY[TY==4]


fig=plt.figure(fignum,[8,6])
ax = fig.add_subplot(1,1,1)
ax.plot(YYe, XXe, color = 'k', ls='', marker='*', label= "excitation")
ax.plot(YYi, XXi, color = 'r', ls='', marker='o', label= "ionization")

circle2 = plt.Circle((0., 0.), 0.0375, color='blue')
circle1 = plt.Circle((0., 0.), 1., color='blue', fill=False, linewidth=5)
ax.add_artist(circle2)
ax.add_artist(circle1)
ax.set_xlabel("y [mm]")
ax.set_ylabel("x [mm]")



axins = zoomed_inset_axes(ax, 11., loc=2) # zoom-factor: 2.5, location: upper-left

axins.plot(YYe, XXe, color = 'k', ls='', marker='*', label= "excitation")
axins.plot(YYi, XXi, color = 'r', ls='', marker='o', label= "ionization")
circle2ins = plt.Circle((0., 0.), 0.0375, color='blue')
circle1ins = plt.Circle((0., 0.), 1., color='blue', fill=False, linewidth=5)
axins.add_artist(circle2ins)
axins.add_artist(circle1ins)
#axins.set_xlabel("")
#axins.set_ylabel("")
plt.yticks(visible=False)
plt.xticks(visible=False)

axins.set_xlim(.03,.1)
axins.set_ylim(-.01,.01)
#plt.savefig(workdir+"/GarOneEvent"+str(fignum)+"s.png")
mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")
ax.set_xlim(-.09,1.01)
ax.set_ylim(-.45,.45)



ax.legend(loc=3, fontsize=20)
#ax.grid(True)
plt.savefig(workdir+"/GarOneEvent"+str(fignum)+".png")

fignum=101
A = np.loadtxt(workdir+"OneEvent"+str(fignum)+".txt", skiprows=3)

XX = A[:,2]*10
YY = A[:,3]*10
TY = A[:,6]

XXi= XX[TY==1]
YYi= YY[TY==1]

XXe= XX[TY==4]
YYe= YY[TY==4]

fig=plt.figure(fignum,[8,6])
ax = fig.add_subplot(1,1,1)
ax.plot(YYe, XXe, color = 'k', ls='', marker='*', label= "excitation")
ax.plot(YYi, XXi, color = 'r', ls='', marker='o', label= "ionization")

circle2 = plt.Circle((0., 0.), 0.0375, color='blue')
circle1 = plt.Circle((0., 0.), 1., color='blue', fill=False, linewidth=5)
ax.add_artist(circle2)
ax.add_artist(circle1)
ax.set_xlabel("y [mm]")
ax.set_ylabel("x [mm]")

axins = zoomed_inset_axes(ax, 11., loc=2) # zoom-factor: 2.5, location: upper-left

axins.plot(YYe, XXe, color = 'k', ls='', marker='*', label= "excitation")
axins.plot(YYi, XXi, color = 'r', ls='', marker='o', label= "ionization")
circle2ins = plt.Circle((0., 0.), 0.0375, color='blue')
circle1ins = plt.Circle((0., 0.), 1., color='blue', fill=False, linewidth=5)
axins.add_artist(circle2ins)
axins.add_artist(circle1ins)
#axins.set_xlabel("")
#axins.set_ylabel("")
plt.yticks(visible=False)
plt.xticks(visible=False)

axins.set_xlim(.03,.07)
axins.set_ylim(-.005,.015)
#plt.savefig(workdir+"/GarOneEvent"+str(fignum)+"s.png")
mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")

#ax.grid(True)
ax.set_xlim(-.09,1.01)
ax.set_ylim(-.45,.45)
ax.legend(loc=3, fontsize=20)
plt.savefig(workdir+"/GarOneEvent"+str(fignum)+".png")


