# -*- coding: utf-8 -*-
"""
Created on Thu May 17 02:24:26 2018

@author: wxj
"""

#Wei JI 2017.09.04
#Finish it as soon as possible.
import os, sys, re, glob
sys.path.insert(0, "/home/wxj/.2go")
sys.path.insert(0, "/home/wxj/.x2go")
sys.path.insert(0, "/home/wxj/.2go/PythonScripts/RunSetups")
sys.path.insert(0, "/home/wxj/.2go/PythonScripts")
sys.path.insert(0, "/home/wxj/gtest_result/")

import numpy as np

import sys
import numpy as np
import scipy as sp
import scipy 
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.interpolate as interp
import scipy.integrate as integrate
import scipy.io as sio
import scipy.optimize as opt
from scipy.stats import binom
from scipy.stats import norm
#import UsefulFunctions as uf
from color import *
from plotStyle import *

'''
label_size=12.
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.size'] = 15
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.autolayout'] = False  # When True, automatically adjust subplot
                            # parameters to make the plot fit the figure

mpl.rcParams['figure.subplot.left']  = 0.18  # the left side of the subplots of the figure
mpl.rcParams['figure.subplot.right']  = 0.85    # the right side of the subplots of the figure
mpl.rcParams['figure.subplot.bottom']  = 0.15   # the bottom of the subplots of the figure
mpl.rcParams['figure.subplot.top']  = 0.85     # the top of the subplots of the figure
mpl.rcParams['figure.subplot.wspace']  = 0.2    # the amount of width reserved for blank space between subplots
mpl.rcParams['figure.subplot.hspace']  = 0.2    # the amount of height reserved for white space between subplots
'''

procids=[]
data=[]

workdir = "/home/wxj/gastest/xenonProperties/"
savedir =workdir

if not os.path.exists(savedir):
    os.makedirs(savedir)


lightcollection_tot=1.7e-2
lightcollection_top=.85e-2
lightcollection_bot=.85e-2


# drift time in ns vs dv in integer for wire and plate drifting.
WireDriftTimeMean=[30000, 10000., 8097, 7566, 6845., 5852, 4379, 4052, #0-7
3461., 3038, 2696, 2470, 2221., #8-12
2081, #13-16
1900, 1800, 1700, ]# not exact

WireDriftTimeErr=[30000, 2000., 600, 500, 290.1, 246.2, 201.6, 167.8, #0-4 6: 401.6,
151., 108.4, 78.63, 109.7, 107.4,  #5-9
100, #13-16
100, 100, 100,]# not exact

PlateDriftTimeMean=[30000, 30000, 30000, 4141,3100,
2303, 1925, 1640, 1435,1277,
1150.,1048, 962.7, 891.4,830.7,
777.8, 731.9,
]

PlateDriftTimeErr=[30000,  30000,  30000,  137.2, 100.,
100., 139.7,100.4, 55.59, 40.66,
30.03,24.77,21.82, 18.45,16.08,
14.63, 13.32,
]#16

WireDriftTimeMean=np.array(WireDriftTimeMean)
WireDriftTimeErr=np.array(WireDriftTimeErr)
PlateDriftTimeMean=np.array(PlateDriftTimeMean)
PlateDriftTimeErr=np.array(PlateDriftTimeErr)

SE_r = np.array([  0.,   5.,  10.,  15.,  20.,  25.,  30.,  35.,  40.,  45.,  50.,  55.,  60.,
        65.,])#  70.,  75.,  80.,  85.,  90.,  95., 100.])
SE_photon=np.array([      2.0005,    1.9836,    1.9648,    1.9515,    1.9289,    1.9068,    1.8637,    1.8139,    1.7833,    1.7211,    1.6634,
    1.5664,    1.4478,    1.2717,])/100#     0.4088,    0.2090,    0.1034,    0.0622,    0.0435,    0.0330,    0.0246,])/100.

a = 0.5*(SE_r[1:]+SE_r[:-1])
a=np.concatenate((np.array([0]),a))
x= np.arange(0,100,1)
y= np.zeros_like(x)
n=1000
for jj in range(len(a)-1):
    for xx in range(len(x)):
        y[x[xx]]+=(SE_r[jj+1]**2-SE_r[jj]**2)* binom.pmf(x[xx], n, SE_photon[jj])

plt.figure(2)
plt.plot(x,y/float(y.sum()), label='revised profile')
plt.plot(x, binom.pmf(x, n, 0.017), label='binomial profile')
#plt.plot(x, norm.pdf(x, loc=1000*0.017, scale=np.sqrt(1000*0.017)), label='gaussian profile')
plt.xlabel(r'# Photon created [phe/e]')
plt.ylabel(r'Probability [phe$^{-1}$]')
plt.xlim(0,50)
plt.grid('on')
plt.legend(loc=1)
#plt.savefig("/home/wxj/gtest_result/"+"SE_profile.png")
plt.savefig(savedir+"SE_profile.png")
#plt.close('all')

#########Start naive photon creation.

ma= 0.137 #ph/e/V
mb= -4.7e-18 #ph/e *cm^2/atom
mb=mb*1.e-4

NA = 6.022e23 #Avoga
n1=0.020795e3; label1='0.5 bara, 290 K'
n2=0.041709e3; label2='1.0 bara, 290 K'
n3=0.062744e3; label3='1.5 bara, 290 K'
n4=0.083901e3; label4='2.0 bara, 290 K'
n5=0.10518e3; label5='2.5 bara, 290 K'
n6=0.12659e3; label6='3.0 bara, 290 K'
n7=0.13950e3; label7='3.3 bara, 290 K'
n8=0.14812e3; label8='3.5 bara, 290 K'

n_list=[n1,n2,n3,n4,n5,n6,n7,n8]
label_list=[label1,label2,label3,label4,label5,label6,label7,label8]

dv_list= np.arange(0,17,1)
e_list=[]
for dv in dv_list:
    e_list.append(dv*1.e3/13e-3)

fignum=100
fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)
#for jj in range(len(n_list)):
for jj in [1, 3, 6]:
    nn=n_list[jj]
    lab = label_list[jj]    
    xlist=[]
    ylist=[]
    for kk in range(len(e_list)):
        dv=dv_list[kk]
        ee=e_list[kk]
        xlist.append(dv)
        y=((ma*ee)/(nn*NA)+mb)*(nn*NA)*13.e-3
        y=np.max([0,y])
        ylist.append(y)
    plt.plot(xlist, ylist, label=lab)
    

plt.xlabel('operation voltage dV [kV]')
plt.ylabel('# Photon created [phe/e]')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height*0.7])
ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.grid('on')
plt.savefig(savedir+"PhotonCreationNaive.png")
#########End naive photon creation.


fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)
#for jj in range(len(n_list)):
for jj in [1, 3, 6]:
    nn=n_list[jj]
    lab = label_list[jj]    
    xlist=[]
    ylist_se=[]
    for kk in range(len(e_list)):
        dv=dv_list[kk]
        ee=e_list[kk]
        xlist.append(dv)
        y=((ma*ee)/(nn*NA)+mb)*(nn*NA)*13.e-3
        y=np.max([0,y])*lightcollection_tot
        ylist_se.append(y)
    plt.plot(xlist, ylist_se, label=lab)
    

plt.xlabel('operation voltage dV [kV]')
plt.ylabel('# Photon collected [phe/e]')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height*0.7])
ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.grid('on')
plt.savefig(savedir+"PhotonColletion_Naive.png")
#########End naive photon colletion.


T1090_69=np.load("/home/wxj/gtest_result/PhotonNum_T1090_69_SelEff.npy")
T1090_79=np.load("/home/wxj/gtest_result/PhotonNum_T1090_79_SelEff.npy")
  
  ##############################
fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)  
MaxX=50
for jj in [6,8,10,12,14,16]:
    x = scipy.linspace(0,MaxX,MaxX+1)
    pmf = scipy.stats.binom.pmf(x,ylist[jj],lightcollection_tot)
    ax.plot(x,pmf, label="dV: %d kV "%(xlist[jj]) )

text="3.3 bara"
ax.annotate(text, xy=(0.8, 0.8), xycoords='axes fraction', fontsize=10,
    bbox=dict(facecolor='white', alpha=0.8),
    horizontalalignment='left', verticalalignment='bottom')

plt.ylabel(r'Probability [phe$^{-1}$]')
plt.xlabel('# Photon collected [phe/e]')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height*0.7])
ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
#ax.set_xlim(0,16)
ax.set_ylim(0,0.25)
plt.grid('on')
plt.savefig(savedir+"PhotonColletionNaiveProfile3300mbar1.png")
#########End naive profile of photon colletion.


  
##############################
fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)  
MaxX=100
xlist_2=[]
ymedlist_2=[]
yminlist_2=[]
ymaxlist_2=[]
for jj in [6,8,10,12,14,16]:
    #x = scipy.linspace(0,MaxX,MaxX+1)
    #pmf = scipy.stats.binom.pmf(x,ylist[jj],lightcollection_tot)
    pmfmed = binom.ppf(0.5,ylist[jj],lightcollection_tot)
    pmfmin = binom.ppf(0.5-.34,ylist[jj],lightcollection_tot)
    pmfmax = binom.ppf(0.5+.34,ylist[jj],lightcollection_tot)
    xlist_2.append(xlist[jj])
    ymedlist_2.append(pmfmed)
    yminlist_2.append(pmfmin)
    ymaxlist_2.append(pmfmax)

ax.plot(xlist_2, ymedlist_2,ls='-', label='mean', color=color_list[0])
ax.plot(xlist_2, yminlist_2,ls='-.', label='1 sigma', color=color_list[0])
ax.plot(xlist_2, ymaxlist_2,ls='-.', color=color_list[0])
text="3.3 bara"
ax.annotate(text, xy=(0.2, 0.8), xycoords='axes fraction', fontsize=10,
    bbox=dict(facecolor='white', alpha=0.8),
    horizontalalignment='left', verticalalignment='bottom')

plt.xlabel('operation voltage dV [kV]')
plt.ylabel('# Photon collected [phe/e]')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height*0.7])
ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.grid('on')
plt.savefig(savedir+"PhotonColletionNaiveProfileMeanSigma3300mbar.png")
#########End naive profile of photon colletion.