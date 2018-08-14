#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 12:57:52 2018

@author: weiji

This code is to separte PTFE fluorescence analysis (which requires waveform) 
from other analysis that does not require waveform in Baseline.py
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 09:58:08 2018

@author: wxj
"""


import os, sys
wd='/Users/weiji/Google Drive/gastest/' #'/home/wxj/gastest/'
workdir=wd + '/DatasetQuality/' #'test'#
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
sys.path.insert(0, wd+"/obsolete/PythonScript/")

sys.path.insert(0, wd)
sys.path.insert(0, wd+"/xenonProperties/")
sys.path.insert(0, "/Users/weiji/Google Drive/python-lux/")
sys.path.insert(0, "/Users/weiji/Google Drive/python-utils/")
sys.path.insert(0, "/Users/weiji/Google Drive/lzrd-python/lzrd")
sys.path.insert(0, "/Users/weiji/Google Drive/python-utils/utils")

from DriftVoltage import *

option='dv'
pre_length=120
#########################################################
## choose one Setup below and uncomment to use it.

#from Run2XeSetup import *  #useful
#from Run2XeSetup_ProcessAfterTomaszmod import *  #useful
#from Run2VacSetup import *
#from Run3XeSetup import *
#from Run3XeSetupNewProcess import *

#from Run5Xe3300Setup1 import *
#from Run5Xe3300Setup2 import * #useful
#from Run6XeSetup2000mbar_ch4n_ch7p_set0 import *
#from Run6XeSetupBeforeCirApGn import *

#from Run6XeSetup3300mbar_befcir_apgn_set1 import * #useful
#from Run6XeSetup3300mbar_befcir_angp_set1 import * #useful
#from Run6XeSetup3300mbar_aftcir_apgn_set1 import * #useful
#from Run6XeSetup3300mbar_aftcir_apgn_set1_short import * #useful
#from Run6XeSetup3300mbar_aftcir_angp_set1 import * #useful
####################################################################
########before this do not need to run execfile( "/home/wxj/.2go/PythonScripts/RunSetups/PreSetup.py")



#from Run7XeSetup3300mbar_First import * #useful
#from Run7XeSetup3300mbar_Fresh1stTime1dayAfter import * #useful
#from Run7XeSetup3300mbar_Fresh2ndTime import * #useful
#from Run7XeSetup3300mbar_Fresh2ndTime1dayAfter import * #useful
#from Run7XeSetup3300mbar_Fresh2ndTime2dayAfter import * #useful
#from Run7XeSetup3300mbar_Fresh3rdTime import * #useful
#from Run7XeSetup3300mbar_FridayTest import * #useful
#from Run7XeSetup3300mbar_SecondTest0to6kV import * #useful
##################################################

#from Run7XeSetup3300mbar_FullSweepPart1 import * #useful

#from Run7XeSetup3300mbar_FullSweepPartReverse1 import * #useful


#from Run7XeSetup3300mbar_FullSweepPart3 import * #useful
#from Run7XeSetup3300mbar_FullSweepPart4 import * #useful
#from Run7XeSetup3300mbar_A6d5Gn6d5Short import *
#from Run7XeSetup3300mbar_A6d5Gn6d5Long import *
#from Run7XeSetup3300mbar_A4Gn4Short import *
#option='time_min'
#from Run7XeSetup3300mbar_dV8 import *
#from Run7XeSetup3300mbar_dV8_rev import *

#from Run8XeSetup3300mbar_FullSweepPart1 import * #useful
#from Run9XeSetup3300mbar_FullSweepPart1 import * #useful
#from Run9XeSetup_cal_0500mbara import * #useful
#from Run9XeSetup_cal_1000mbara import * #useful
#from Run9XeSetup_cal_1500mbara import * #useful
#from Run9XeSetup_cal_2000mbara import * #useful
#from Run9XeSetup_cal_2500mbara import * #useful
#from Run9XeSetup_cal_3000mbara import * #useful
#from Run9XeSetup_cal_3500mbara import * #useful
#from Run5Xe3300mbar_firstRamp import *
from Param import *
#print FileStart,FileEnd 
#if len(data)==0:
#    execfile( "/home/wxj/.2go/PythonScripts/RunSetups/PreSetup.py")

#print data
## import all module that I need.
#from PostSetup import * 
#### import modules. 
#import numpy as n
import numpy as np
#import numpy as numpy
#import pandas as pd
import random

import time, datetime,calendar
import bindata
import gc
import hist2d
import glob
import re

import lzrdReaderUtils_PyMod as lzrutils
import plot_format_utils as pltutils
import plt2dhist as plt2d
import plotfuncs

import pickle
import matplotlib.pyplot as plt

from multiprocessing import Pool
from matplotlib.ticker import Formatter    # to convert xaxis label to dates
from matplotlib.colors import LogNorm
from matplotlib import dates

import extrap1d

import lzrdReader_PyMod as lzread
#import dict_to_xmlstring_PyMod as dicttoxml

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

color_list=[#'pink',
#'r','g','b','m','k','c','y', 'orange', 'pink','darkgrey','lightblue','darkgreen', 
#'tan', 'firebrick',  'grey', 
#'lightgreen','violet','lightpink','slategrey',
#'darksalmon', 'purple'
'#1f77b4',
'#aec7e8',
'#ff7f0e',
'#ffbb78',
'#2ca02c',
'#98df8a',
'#d62728',
'#ff9896',
'#9467bd',
'#c5b0d5',
'#8c564b',
'#c49c94',
'#e377c2',
'#f7b6d2',
'#7f7f7f',
'#c7c7c7',
'#bcbd22',
'#dbdb8d',
'#17becf',
] 
marker_list=['o',]*40
marker_list=['o','^','v','<','>','+','x','s','*',]
import matplotlib as mpl
import UsefulFunctions as uf
from color import *
from plotStyle import *
procids=[[64767]]
## Compute basic 
porcidnum = len(procids)
#### Selected keys.####
## If want to run faster, load only keys that is necessary for the analysis.
keys = [
'aft_t05', 'aft_t95', 
'aft_t1', 'aft_t2', 'aft_t25', 'aft_t75', 'aft_t0', 
'baselines', 'channels', 'prompt_frac', 
'duration', 
'evtnum', ##'firstvals', 'hft_t1', 
'posareas',
'trigvals', 'rmsratio', 
'neg_area_fraction', 
'skimfactor', 
'times', 
'waveamplitudes', 'waveareas','wavelens',
'pre_baseline',
'post_baseline',
'pos_area_above_threshold',
'pos_len_above_threshold',
#'pos_len_above_threshold_total',
'pos_len_above_threshold_percentile_05',
#'pos_len_above_threshold_percentile_50',
'pos_len_above_threshold_percentile_95',
#'pos_area_begin_pulse',
'pos_area_pulse1',
'pos_area_pulse2',
#'pos_area_begin_pulse',
#'waveforms',
'prompt_frac_250ns', 
'prompt_frac_500ns', 
'prompt_frac_750ns', 
'prompt_frac_1000ns',
'wtime', 
'wtime2', 
'wtime3', 
'wtime4',
]


# variables that can't be written as numpy arrays
picklekeys = [
'waveforms'
]

nonnpykeys = [
'waveforms'
]

## End of Selected keys.

newrqlist=[
    'coin_pulse_ids',
    'coin_pulse_chs',
    'coin_pulse_times',
    'coin_pulse_lens',
    'coin_pulse_areas',
    'coin_pulse_amplitudes',
    'coin_pulse_areas_neg',
    'coin_pulse_amplitudes_neg',
    
    'coin_pulse_lastpulse_ids',
    'coin_pulse_lastpulse_times',
    'coin_pulse_lastpulse_lens',
    'coin_pulse_lastpulse_areas',
    
    'coin_pulse_areas_norm',
    'coin_pulse_areas_sum',
    'coin_pulse_areas_t01',
    'coin_pulse_areas_t05',
    'coin_pulse_areas_t10',
    'coin_pulse_areas_t15',
    'coin_pulse_areas_t25',
    'coin_pulse_areas_t50',
    'coin_pulse_areas_t75',
    'coin_pulse_areas_t85',
    'coin_pulse_areas_t90',
    'coin_pulse_areas_t95',
    'coin_pulse_areas_t99',
    
#    'coin_pulse_waveforms',
#    'coin_pulse_waveforms_norm',
#    'coin_pulse_waveforms_sum',
    
    'coin_pulse_areas_pre_100us',
    'coin_pulse_areas_pre_50us',
    'coin_pulse_areas_pre_20us',
    'coin_pulse_areas_pre_10us',
    
    'coin_pulse_areas_post_10000us',
    'coin_pulse_areas_post_5000us',
    'coin_pulse_areas_post_2000us',    
    'coin_pulse_areas_post_1000us',
    'coin_pulse_areas_post_500us',
    'coin_pulse_areas_post_200us',
    'coin_pulse_areas_post_100us',
    'coin_pulse_areas_post_50us',
    'coin_pulse_areas_post_20us',
    'coin_pulse_areas_post_10us',
    
    'random_pulse_times',
    'random_pulse_areas_pre_100us',
    'random_pulse_areas_pre_50us',
    'random_pulse_areas_pre_20us',
    'random_pulse_areas_pre_10us',
    'random_pulse_areas_post_100us',
    'random_pulse_areas_post_50us',
    'random_pulse_areas_post_20us',
    'random_pulse_areas_post_10us',
    
    'sphe_size',
    'suppress_last_NSamples',
    'arearq',
    'sample_size',
    'usechannels',
    'window_width',
    'post_pulse_length',
    'pre_pulse_length',
    'number_of_channels',

    'coin_pulse_wtime_t1585', 
    'coin_pulse_wtime2_t1585', 
    'coin_pulse_wtime3_t1585', 
    'coin_pulse_wtime4_t1585',    
    
    'coin_pulse_wtime_t0595', 
    'coin_pulse_wtime2_t0595', 
    'coin_pulse_wtime3_t0595', 
    'coin_pulse_wtime4_t0595', 
    
    'in_coin_pulse',
    'waveareas_trim_end',
    'va',#kV
    'vg',#kV
    'dv',#kV
    'disp',#'a:?;g:?'
    'procid',
    'pulse1_start', 'pulse1_stop',# first 300 ns, return integrated area in this window.
'pulse2_start', 'pulse2_stop', # first 800 ns, return integrated area in this window.
    
    'AmpThreshold',
    
    'coin_pulse_areas_section1',
    'coin_pulse_areas_section2',
    'coin_pulse_amplitudes_peaktime',
    'coin_pulse_amplitudes_peaktime_smooth',
    'pos_len_above_threshold_trim_end',
]

waveform_list= [
#    'coin_pulse_waveforms',
    'coin_pulse_waveforms_norm',
    'coin_pulse_waveforms_sum'
]
# end of new rq list


## Load rqs from npy dicts.
ddicts = []
#for jj in range(porcidnum):#range(15,20):#
procids = []
datadir="/Users/weiji/Google Drive/"
procids.append([64767,]) #13001,65901,65831,10201, 33001
for jj in [0]:#range(15,20):#12 kV
    print "Load dict:", jj, '\t', procids[jj]
    try:
#        ddict = uf.LoadDict(procids[jj], keys + newrqlist, nonnpykeys, datadir)
        ddict = uf.LoadCoinPulseRecord(procids[jj], keys+newrqlist, {}, datadir)
        ddicts.append(ddict)
        ddicts[0]['procid'] = procids[0]
    except:
        print "cannot load dict:", jj, '\t', procids[jj]
    del ddict

'''
         <PostDelay0>0x1F4</PostDelay0>
         <PostDelay1>0x1E</PostDelay1>
         <PostDelay2>0x1F4</PostDelay2>
         <PostDelay3>0x1E</PostDelay3>
         <PostThreshold0>0x7D61</PostThreshold0>
         <PostThreshold1>0xFFFF</PostThreshold1>
         <PostThreshold2>0x7F5B</PostThreshold2>
         <PostThreshold3>0xFFFF</PostThreshold3>
         <PreDelay0>0x1E</PreDelay0>
         <PreDelay1>0x1E</PreDelay1>
         <PreDelay2>0x1E</PreDelay2>
         <PreDelay3>0x1E</PreDelay3>
         <PreThreshold0>0x7D80</PreThreshold0>
         <PreThreshold1>0xFFFF</PreThreshold1>
         <PreThreshold2>0x7F62</PreThreshold2>
         <PreThreshold3>0xFFFF</PreThreshold3>
'''
plt.close('all')
fignum=100
jj=0
topADC=0x7D80
botADC=0x7F62
topmV = topADC/(2.**16)*2500
botmV = botADC/(2.**16)*2500
topPMT=(ddicts[jj]['channels']==0) #& (ddicts[jj]['waveareas']>0) & (ddicts[jj]['waveareas']<1000) 
botPMT=(ddicts[jj]['channels']==2)

toptrig = topmV - ddicts[jj]['baselines'][topPMT]
toptrig_med = np.median(toptrig)
topamp = ddicts[jj]['waveamplitudes'][topPMT]

bottrig = botmV - ddicts[jj]['baselines'][botPMT]
bottrig_med = np.median(bottrig)
botamp = ddicts[jj]['waveamplitudes'][botPMT]

print ('top pmt trig value [mV]')
print (toptrig_med)
print ('bot pmt trig value [mV]')
print (bottrig_med)


#######################
##start t2575 vs t85

#######################

XMIN,XMAX=0,60
WBIN = 0.1
NBINS = int((XMAX-XMIN)/WBIN)
bins = np.linspace(XMIN,XMAX,NBINS, endpoint=False)  
YMIN, YMAX=1, 10000

##############################
# top pmt triggerefficiency

fignum=100
fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)
MIN,MAX=12,28
NBINS = int((MAX-MIN)/WBIN)
#plt.hist(toptrig, bins = bins, histtype='step', label= 'top PMT trigger', color='k')
ax.axvline(toptrig_med, label= 'trigger voltage', color='k')
#plt.hist(topamp, bins = bins, histtype='step', label= 'all pulse', color='r')

H, BINS = np.histogram(topamp,range=[0,60],bins=600)
x = uf.MiddleValue(BINS)
ax.step(x, H, where='mid',  label= 'all pulse', color='r', lw=2 )
H, BINS = np.histogram(topamp,range=[MIN,MAX],bins=NBINS)
x = uf.MiddleValue(BINS)

#popt,pcov = opt.curve_fit(uf.Gauss, x, H, p0=[max(H), x.mean(), 100])
popt,pcov = opt.curve_fit(uf.Gauss, x, H, p0=[1.,20,10])
#l = ax.step(x, H, where='mid', color='r', label='top PMT max amplitude')
#l = ax.plot(x,uf.Gauss(x,*popt),color='k',label='Gaussian fit: mean: %.0f, sigma: %.0f'%(popt[1],np.abs(popt[2])))
x= bins
y= uf.Gauss(x,*popt)
l = ax.plot(x, y,color='darkred',ls=":", label='all SPHE')   


print popt[1]-np.abs(popt[2])*2, popt[1]+np.abs(popt[2])*2
text=r'$\mu$'+': %.1f [mV]\n'%(popt[1])+r'$\sigma$'+': %.1f [mV]'%(np.abs(popt[2]))
ax.annotate(text, xy=(0.6, 0.7), xycoords='axes fraction', fontsize=text_size,
    bbox=dict(facecolor='white', alpha=0.8),
    horizontalalignment='left', verticalalignment='bottom', color='darkred')
sel = (x>toptrig_med)
ax.fill_between(x[sel], YMIN*np.ones_like(x[sel]), y[sel], color='darkred', alpha=0.3, label='triggered SPHE')
print ("top trig eff:%.3f")%(1.-stats.norm.cdf((toptrig_med-popt[1])/(np.abs(popt[2]))))
ax.grid('on')
ax.set_xlabel('pulse amplitude [mV]')
ax.set_ylabel(r'counts [bin$^{-1}$]')
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(YMIN, YMAX)
ax.set_yscale('log')
box = ax.get_position()
ax.set_position([box.x0, box.y0+box.height*0.0, box.width, box.height*0.7])

handles, labels = ax.get_legend_handles_labels()
handles[0], handles[1] ,handles[2]= handles[1] ,handles[2] ,handles[0]
labels[0], labels[1],labels[2] = labels[1] ,labels[2],labels[0]
ax.legend(handles, labels, loc=3, bbox_to_anchor=(0., 1.1, 1., .102), ncol=2, mode="expand", borderaxespad=0.,fontsize=20)


plt.savefig(workdir+'/topPMTTriggerEfficiency'+str(ddicts[0]['procid'][0])+'.png')
#plt.close('all')botampsel

#bottom pmt trigger efficiency
fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)
MIN,MAX=22,38
NBINS = int((MAX-MIN)/WBIN)
#plt.hist(bottrig, bins = bins, histtype='step', label= 'bot PMT trigger', color='k')
#ax.hist(botamp, bins = bins, histtype='step', label= 'all pulse', color='b')

ax.axvline(bottrig_med, label= 'trigger voltage', color='k')

H, BINS = np.histogram(botamp,range=[0,60],bins=600)
x = uf.MiddleValue(BINS)
ax.step(x, H, where='mid',  label= 'all pulse', color='b', lw=2 )
H, BINS = np.histogram(botamp,range=[MIN,MAX],bins=NBINS)
x = uf.MiddleValue(BINS)
#popt,pcov = opt.curve_fit(uf.Gauss, x, H, p0=[max(H), x.mean(), 100])
popt,pcov = opt.curve_fit(uf.Gauss, x, H, p0=[1.,20,10])
#l = ax.step(x, H, where='mid', color='r', label='bot PMT max amplitude')
#l = ax.plot(x,uf.Gauss(x,*popt),color='k',label='Gaussian fit: mean: %.0f, sigma: %.0f'%(popt[1],np.abs(popt[2])))
x= bins
y= uf.Gauss(x,*popt)
l = ax.plot(x, y,color='darkblue',ls=":", label='all SPHE')   

print popt[1]-np.abs(popt[2])*2, popt[1]+np.abs(popt[2])*2
text=r'$\mu$'+': %.1f [mV]\n'%(popt[1])+r'$\sigma$'+': %.1f [mV]'%(np.abs(popt[2]))
ax.annotate(text, xy=(0.6, 0.7), xycoords='axes fraction', fontsize=text_size,
    bbox=dict(facecolor='white', alpha=0.8),
    horizontalalignment='left', verticalalignment='bottom', color='darkblue')
sel = (x>bottrig_med)
ax.fill_between(x[sel], YMIN*np.ones_like(x[sel]), y[sel], color='darkblue', alpha=0.3, label='triggered SPHE')
print ("vot trig eff:%.3f")%(1.-stats.norm.cdf((toptrig_med-popt[1])/(np.abs(popt[2]))))
ax.grid('on')
ax.set_xlabel('pulse amplitude [mV]')
ax.set_ylabel(r'counts [bin$^{-1}$]')
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(YMIN, YMAX)
ax.set_yscale('log')
box = ax.get_position()
ax.set_position([box.x0, box.y0+box.height*0.0, box.width, box.height*0.7])

handles, labels = ax.get_legend_handles_labels()
handles[0], handles[1] ,handles[2]= handles[1] ,handles[2] ,handles[0]
labels[0], labels[1],labels[2] = labels[1] ,labels[2],labels[0]
ax.legend(handles, labels, loc=3, bbox_to_anchor=(0., 1.1, 1., .102), ncol=2, mode="expand", borderaxespad=0.,fontsize=20)

plt.savefig(workdir+'/botPMTTriggerEfficiency'+str(ddicts[0]['procid'][0])+'.png')
#plt.close('all')




















################################################
def twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset=0):
    offset=0    
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2.)/(2.*sigma_x**2.) + (np.sin(theta)**2.)/(2.*sigma_y**2.)
    b = -(np.sin(2.*theta))/(4.*sigma_x**2.) + (np.sin(2*theta))/(4.*sigma_y**2.)
    c = (np.sin(theta)**2.)/(2.*sigma_x**2.) + (np.cos(theta)**2)/(2.*sigma_y**2.)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2.) + 2.*b*(x-xo)*(y-yo) + c*((y-yo)**2.)))
    return g.ravel()

XMIN,XMAX=-200,800
XWBIN = 10
XNBINS = int((XMAX-XMIN)/XWBIN)
binsx = np.linspace(XMIN,XMAX,XNBINS+1, endpoint=True) 
x = uf.MiddleValue(binsx)
YMIN, YMAX=12, 28
YWBIN = .5
YNBINS = int((YMAX-YMIN)/YWBIN)
binsy = np.linspace(YMIN,YMAX,YNBINS+1, endpoint=True) 
y = uf.MiddleValue(binsy)

topampsel = (ddicts[jj]['waveamplitudes']>YMIN) & (ddicts[jj]['waveamplitudes']<YMAX)
try:
    toparea = ddicts[jj]['waveareas_trim_end'][topPMT & topampsel ]
except:
    toparea = ddicts[jj]['waveareas'][topPMT & topampsel ]
topamp2 = ddicts[jj]['waveamplitudes'][topPMT & topampsel ]

x, y = np.meshgrid(x, y)
##############################
# top pmt area
fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)

#plt.hist(toptrig, bins = bins, histtype='step', label= 'top PMT trigger', color='k')
datad, binsx, binsy = np.histogram2d(toparea, topamp2, bins = (binsx, binsy) )
xx=x.flatten()
yy=y.flatten()
datad =datad.T
dataf = datad.flatten()
initial_guess = (100,  539.63027286,   23.31065411,  122.62491461,
         11.48272811,    2.98534865,    0)
popt, pcov = opt.curve_fit(twoD_Gaussian, np.array([xx,yy]), dataf, p0 = initial_guess,maxfev=160000)
data_fitted = twoD_Gaussian((xx, yy), *popt)

data_ini = twoD_Gaussian((xx, yy), *initial_guess)

ax.hold(True)
cax = plt.pcolormesh(x, y, datad.reshape(YNBINS,XNBINS), norm = mpl.colors.LogNorm(vmin=.5), cmap=plt.cm.jet)
#ax.contour(x, y, data_fitted.reshape(XNBINS-1,YNBINS-1).T, 8, colors='w')
cbar = plt.colorbar()
#cbar.ax.set_yticklabels(['0','1','2','>3'])


fac1=1.068
e1 = Ellipse((popt[1], popt[2]), np.abs(popt[3])*fac1, np.abs(popt[4])*fac1, popt[5], ls=":", lw=3, fill=False, zorder=2)
ax.add_patch(e1)
fac2=1.731
e2 = Ellipse((popt[1], popt[2]), np.abs(popt[3])*fac2, np.abs(popt[4])*fac2, popt[5], ls=":", lw=3, fill=False, zorder=2)
ax.add_patch(e2)
ax.plot(popt[1], popt[2], markersize=4, marker='o', color="k")

text=r'$\mu_{x}$'+': %.1f\n'%(popt[1])+r'$\sigma_{x}$'+': %.1f\n'%(np.abs(popt[3]))+r'$\mu_{y}$'+': %.1f\n'%(popt[2])+r'$\sigma_{y}$'+': %.1f\n'%(np.abs(popt[4]))+r'$\theta$'+': %.1f'%(popt[5]/np.pi*180)+r'$^{\circ}$'        
ax.annotate(text, xy=(0.015, 0.52), xycoords='axes fraction', fontsize=text_size,
    bbox=dict(facecolor='white', alpha=0.8),
    horizontalalignment='left', verticalalignment='bottom', color='darkred')

cbar.set_label(r'counts [bin$^{-1}$]', rotation=90)
ax.set_xlabel('pulse area [mV ns]')
ax.set_ylabel('pulse amplitude [mV]')
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(YMIN, YMAX)
#ax.set_yscale('log')
#box = ax.get_position()
#ax.set_position([box.x0, box.y0+box.height*0.0, box.width, box.height*0.7])
#ax.legend(loc=3, bbox_to_anchor=(0., 1.1, 1., .102), ncol=2, mode="expand", borderaxespad=0.,fontsize=12)

plt.savefig(workdir+'/topPMTArea'+str(ddicts[0]['procid'][0])+'.png')
#plt.close('all')


print ('top pmt area [mV ns], amp [mV] error')
print (popt[1], np.abs(popt[3])*np.cos(popt[5]) - np.abs(popt[4])*np.sin(popt[5])  )
print (popt[2], np.abs(popt[3])*np.sin(popt[5]) + np.abs(popt[4])*np.cos(popt[5])  )


#####################################################################

XMIN,XMAX=0,1000
XWBIN = 10
XNBINS = int((XMAX-XMIN)/XWBIN)
binsx = np.linspace(XMIN,XMAX,XNBINS+1, endpoint=True) 
x = uf.MiddleValue(binsx)
YMIN, YMAX=22, 38
YWBIN = .5
YNBINS = int((YMAX-YMIN)/YWBIN)
binsy = np.linspace(YMIN,YMAX,YNBINS+1, endpoint=True) 
y = uf.MiddleValue(binsy)

botampsel = (ddicts[jj]['waveamplitudes']>YMIN) & (ddicts[jj]['waveamplitudes']<YMAX)
try:
    botarea = ddicts[jj]['waveareas_trim_end'][botPMT & botampsel ]
except: 
    botarea = ddicts[jj]['waveareas'][botPMT & botampsel ]
botamp2 = ddicts[jj]['waveamplitudes'][botPMT & botampsel ]

x, y = np.meshgrid(x, y)
##############################
# bot pmt area
fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)

#plt.hist(bottrig, bins = bins, histtype='step', label= 'bot PMT trigger', color='k')
datad, binsx, binsy = np.histogram2d(botarea, botamp2, bins = (binsx, binsy) )
xx=x.flatten()
yy=y.flatten()
datad =datad.T
dataf = datad.flatten()
initial_guess = (100,  539.63027286,   23.31065411,  122.62491461,
         11.48272811,    2.98534865,    0)
popt, pcov = opt.curve_fit(twoD_Gaussian, np.array([xx,yy]), dataf, p0 = initial_guess,maxfev=160000)
data_fitted = twoD_Gaussian((xx, yy), *popt)

data_ini = twoD_Gaussian((xx, yy), *initial_guess)

ax.hold(True)
cax = plt.pcolormesh(x, y, datad.reshape(YNBINS,XNBINS), norm = mpl.colors.LogNorm(vmin=.5), cmap=plt.cm.jet)
#ax.contour(x, y, data_fitted.reshape(XNBINS-1,YNBINS-1).T, 8, colors='w')
cbar = plt.colorbar()
#cbar.ax.set_yticklabels(['0','1','2','>3'])

fac1=1.068
e1 = Ellipse((popt[1], popt[2]), np.abs(popt[3])*fac1, np.abs(popt[4])*fac1, popt[5], ls=":", lw=3, fill=False, zorder=2)
ax.add_patch(e1)
fac2=1.731
e2 = Ellipse((popt[1], popt[2]), np.abs(popt[3])*fac2, np.abs(popt[4])*fac2, popt[5], ls=":", lw=3, fill=False, zorder=2)
ax.add_patch(e2)
ax.plot(popt[1], popt[2], markersize=4, marker='o', color="k")

text=r'$\mu_{x}$'+': %.1f\n'%(popt[1])+r'$\sigma_{x}$'+': %.1f\n'%(np.abs(popt[3]))+r'$\mu_{y}$'+': %.1f\n'%(popt[2])+r'$\sigma_{y}$'+': %.1f\n'%(np.abs(popt[4]))+r'$\theta$'+': %.1f'%(popt[5]/np.pi*180)+r'$^{\circ}$'        
ax.annotate(text, xy=(0.015, 0.52), xycoords='axes fraction', fontsize=text_size,
    bbox=dict(facecolor='white', alpha=0.8),
    horizontalalignment='left', verticalalignment='bottom', color='darkblue')

cbar.set_label(r'counts [bin$^{-1}$]', rotation=90)
ax.set_xlabel('pulse area [mV ns]')
ax.set_ylabel('pulse amplitude [mV]')
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(YMIN, YMAX)
#ax.set_yscale('log')
#box = ax.get_position()
#ax.set_position([box.x0, box.y0+box.height*0.0, box.width, box.height*0.7])
#ax.legend(loc=3, bbox_to_anchor=(0., 1.1, 1., .102), ncol=2, mode="expand", borderaxespad=0.,fontsize=12)

plt.savefig(workdir+'/botPMTArea'+str(ddicts[0]['procid'][0])+'.png')
#plt.close('all')

print ('bottom pmt area [mV ns], amp [mV] error')
print (popt[1], np.abs(popt[3])*np.cos(popt[5]) - np.abs(popt[4])*np.sin(popt[5])  )
print (popt[2], np.abs(popt[3])*np.sin(popt[5]) + np.abs(popt[4])*np.cos(popt[5])  )



##################### PMT single pulse quality t2575 vs t95


XMIN,XMAX=0,200
XWBIN = 4
XNBINS = int((XMAX-XMIN)/XWBIN)
binsx = np.linspace(XMIN,XMAX,XNBINS+1, endpoint=True) 
x = uf.MiddleValue(binsx)
YMIN, YMAX=0, 2500
YWBIN = 4
YNBINS = int((YMAX-YMIN)/YWBIN)
binsy = np.linspace(YMIN,YMAX,YNBINS+1, endpoint=True) 
y = uf.MiddleValue(binsy)

topt2575=(ddicts[jj]['aft_t75']-ddicts[jj]['aft_t25'])[topPMT & topampsel ]
topt95=(ddicts[jj]['aft_t95'])[topPMT & topampsel ]
x, y = np.meshgrid(x, y)
##############################
# top pmt short pulse cut
fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)

xcut = 35
ycut = 150
ax.plot([xcut, xcut], [ycut, YMAX], label= '', color='k', ls="-")
ax.plot([xcut, XMAX], [ycut, ycut], label= '', color='k', ls="-")

#plt.hist(toptrig, bins = bins, histtype='step', label= 'top PMT trigger', color='k')
datad, binsx, binsy = np.histogram2d(topt2575, topt95, bins = (binsx, binsy) )
xx=x.flatten()
yy=y.flatten()

ax.hold(True)
cax = plt.pcolormesh(x, y, datad.T, norm = mpl.colors.LogNorm(vmin=.5), cmap=plt.cm.jet)
#ax.contour(x, y, data_fitted.reshape(XNBINS-1,YNBINS-1).T, 8, colors='w')
cbar = plt.colorbar()


cbar.set_label(r'counts [bin$^{-1}$]', rotation=90)
ax.set_xlabel('t2575 [ns]')
ax.set_ylabel('t95 [ns]')
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(.5e2, YMAX)
ax.set_yscale('log')
#box = ax.get_position()
#ax.set_position([box.x0, box.y0+box.height*0.0, box.width, box.height*0.7])
#ax.legend(loc=3, bbox_to_anchor=(0., 1.1, 1., .102), ncol=2, mode="expand", borderaxespad=0.,fontsize=12)

plt.savefig(workdir+'/topPMTt2575t95'+str(ddicts[0]['procid'][0])+'.png')
#plt.close('all')


bott2575=(ddicts[jj]['aft_t75']-ddicts[jj]['aft_t25'])[botPMT & botampsel ]
bott95=(ddicts[jj]['aft_t95'])[botPMT & botampsel ]
x, y = np.meshgrid(x, y)
##############################
# bot pmt short pulse cut
fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)

xcut = 35
ycut = 230
ax.plot([xcut, xcut], [ycut, YMAX], label= '', color='k', ls="-")
ax.plot([xcut, XMAX], [ycut, ycut], label= '', color='k', ls="-")

#plt.hist(bottrig, bins = bins, histtype='step', label= 'bot PMT trigger', color='k')
datad, binsx, binsy = np.histogram2d(bott2575, bott95, bins = (binsx, binsy) )
xx=x.flatten()
yy=y.flatten()

ax.hold(True)
cax = plt.pcolormesh(x, y, datad.T, norm = mpl.colors.LogNorm(vmin=.5), cmap=plt.cm.jet)
#ax.contour(x, y, data_fitted.reshape(XNBINS-1,YNBINS-1).T, 8, colors='w')
cbar = plt.colorbar()


cbar.set_label(r'counts [bin$^{-1}$]', rotation=90)
ax.set_xlabel('t2575 [ns]')
ax.set_ylabel('t95 [ns]')
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(YMIN, YMAX)
#ax.set_yscale('log')
#box = ax.get_position()
#ax.set_position([box.x0, box.y0+box.height*0.0, box.width, box.height*0.7])
#ax.legend(loc=3, bbox_to_anchor=(0., 1.1, 1., .102), ncol=2, mode="expand", borderaxespad=0.,fontsize=12)

plt.savefig(workdir+'/botPMTt2575t95'+str(ddicts[0]['procid'][0])+'.png')
#plt.close('all')






###############################
#dead time evaluation
# top pmt
topdt = np.diff(ddicts[jj]['times'][topPMT])
toplens = ddicts[jj]['wavelens'][topPMT]
toplensp= toplens[:-1]
topti=topdt-toplensp
fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)

BINS = 10**np.linspace(0,7,211, endpoint=True)
normto = 180 #I want to norm to 1000e3 ns (1000 us, 1ms), which is bin number 150 

topsel = (toplensp >=0)
topti = topti[topsel]
H, BINS = np.histogram(topti,bins=BINS)
x = uf.MiddleValue(BINS)
Hp = H/(np.diff(BINS))
Hpr = Hp
Hpp =Hpr/Hpr[normto]
ax.step(x, Hpp, where='mid',  label= 'all pulse', color='r', lw=2 )

topsel1 = (toplensp >=0) & (toplensp <3.e3)
topti1 = topti[topsel1]
H, BINS = np.histogram(topti1,bins=BINS)
x = uf.MiddleValue(BINS)
Hp = H/(np.diff(BINS))
Hpr = Hp
Hpp =Hpr/Hpr[normto]
ax.step(x, Hpp, where='mid',  label= 'pulse length'+r'$\in$'+'[0, 3) [us]', color='m', lw=2 )


topsel2 = (toplensp >=3.e3) & (toplensp <10.e3)
topti2 = topti[topsel2]
H, BINS = np.histogram(topti2,bins=BINS)
x = uf.MiddleValue(BINS)
Hp = H/(np.diff(BINS))
Hpr = Hp
Hpp =Hpr/Hpr[normto]
ax.step(x, Hpp, where='mid',  label= 'pulse length'+r'$\in$'+'[3, 10) [us]', color='orange', lw=2 )


topsel3 = (toplensp >=10.e3) & (toplensp <30.e3)
topti3 = topti[topsel3]
H, BINS = np.histogram(topti3,bins=BINS)
x = uf.MiddleValue(BINS)
Hp = H/(np.diff(BINS))
Hpr = Hp
Hpp =Hpr/Hpr[normto]
ax.step(x, Hpp, where='mid',  label= 'pulse length'+r'$\in$'+'[10, '+'30'+') [us]', color='darkgreen', lw=2 )

topsel4 = (toplensp >=30.e3) #& (toplensp <100.e3)
topti4 = topti[topsel4]
H, BINS = np.histogram(topti4,bins=BINS)
x = uf.MiddleValue(BINS)
Hp = H/(np.diff(BINS))
Hpr = Hp
Hpp =Hpr/Hpr[normto]
ax.step(x, Hpp, where='mid',  label= 'pulse length'+r'$\in$'+'[30, '+r'$\infty$'+') [us]', color='darkblue', lw=2 )


#ax.axvline(20.e3, ls= ':', color='k', label='20e3 [ns]')

ax.set_xlabel('time interval [ns]')
ax.set_ylabel('counts (scaled)')
#ax.set_xlim(XMIN, XMAX)
ax.set_ylim(1.e-1, 1.e3)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid('on')
box = ax.get_position()
ax.set_position([box.x0, box.y0+box.height*0.0, box.width, box.height*0.7])
ax.legend(loc=3, bbox_to_anchor=(0., 1.1, 1., .102), ncol=2, mode="expand", borderaxespad=0.,fontsize=12)

plt.savefig(workdir+'/topPMTdeadtime'+str(ddicts[0]['procid'][0])+'.png')
#plt.close('all')


###############################
#dead time evaluation
# bottom pmt
botdt = np.diff(ddicts[jj]['times'][botPMT])
botlens = ddicts[jj]['wavelens'][botPMT]
botlensp= botlens[:-1]
botti=botdt-botlensp
fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)

BINS = 10**np.linspace(0,7,211, endpoint=True)
normto = 180 #I want to norm to 1000e3 ns (1000 us, 1ms), which is bin number 150 

 
H, BINS = np.histogram(botti,bins=BINS)
x = uf.MiddleValue(BINS)
Hp = H/(np.diff(BINS))
Hpp =Hp/Hp[normto]
ax.step(x, Hpp, where='mid',  label= 'all pulse', color='r', lw=2 )

botsel1 = (botlensp >=0) & (botlensp <3.e3)
botti1 = botti[botsel1]
H, BINS = np.histogram(botti1,bins=BINS)
x = uf.MiddleValue(BINS)
Hp = H/(np.diff(BINS))
Hpp =Hp/Hp[normto]
ax.step(x, Hpp, where='mid',  label= 'pulse length'+r'$\in$'+'[0, 3) [us]', color='m', lw=2 )


botsel2 = (botlensp >=3.e3) & (botlensp <10.e3)
botti2 = botti[botsel2]
H, BINS = np.histogram(botti2,bins=BINS)
x = uf.MiddleValue(BINS)
Hp = H/(np.diff(BINS))
Hpp =Hp/Hp[normto]
ax.step(x, Hpp, where='mid',  label= 'pulse length'+r'$\in$'+'[3, 10) [us]', color='orange', lw=2 )


botsel3 = (botlensp >=10.e3) & (botlensp <30.e3)
botti3 = botti[botsel3]
H, BINS = np.histogram(botti3,bins=BINS)
x = uf.MiddleValue(BINS)
Hp = H/(np.diff(BINS))
Hpp =Hp/Hp[normto]
ax.step(x, Hpp, where='mid',  label= 'pulse length'+r'$\in$'+'[10, '+'30'+') [us]', color='darkgreen', lw=2 )

botsel4 = (botlensp >=30.e3) #& (botlensp <100.e3)
botti4 = botti[botsel4]
H, BINS = np.histogram(botti4,bins=BINS)
x = uf.MiddleValue(BINS)
Hp = H/(np.diff(BINS))
Hpp =Hp/Hp[normto]
ax.step(x, Hpp, where='mid',  label= 'pulse length'+r'$\in$'+'[30, '+r'$\infty$'+') [us]', color='darkblue', lw=2 )


#ax.axvline(20.e3, ls= ':', color='k', label='20e3 [ns]')

ax.set_xlabel('time interval [ns]')
ax.set_ylabel('counts (scaled)')
#ax.set_xlim(XMIN, XMAX)
ax.set_ylim(1.e-1, 1.e3)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid('on')
box = ax.get_position()
ax.set_position([box.x0, box.y0+box.height*0.0, box.width, box.height*0.7])
ax.legend(loc=3, bbox_to_anchor=(0., 1.1, 1., .102), ncol=2, mode="expand", borderaxespad=0.,fontsize=12)

plt.savefig(workdir+'/botPMTdeadtime'+str(ddicts[0]['procid'][0])+'.png')
#plt.close('all')








######################
# PMT PTFE fluorescence evaluation
# top pmt
coinsel = np.ones_like(ddicts[jj]['coin_pulse_times'], dtype=bool)
coinsel1 = (1<=ddicts[jj]['coin_pulse_areas_sum']) & (ddicts[jj]['coin_pulse_areas_sum']<3)
coinsel2 = (3<=ddicts[jj]['coin_pulse_areas_sum']) & (ddicts[jj]['coin_pulse_areas_sum']<10)
coinsel3 = (10<=ddicts[jj]['coin_pulse_areas_sum']) & (ddicts[jj]['coin_pulse_areas_sum']<30)
coinsel4 = (30<=ddicts[jj]['coin_pulse_areas_sum']) & (ddicts[jj]['coin_pulse_areas_sum']<100)
coinsel5 = (100<=ddicts[jj]['coin_pulse_areas_sum']) & (ddicts[jj]['coin_pulse_areas_sum']<300)
coinsel6 = (300<=ddicts[jj]['coin_pulse_areas_sum']) & (ddicts[jj]['coin_pulse_areas_sum']<1000)
coinsel7 = (1000<=ddicts[jj]['coin_pulse_areas_sum']) & (ddicts[jj]['coin_pulse_areas_sum']<3000)
coinsel8 = (3000<=ddicts[jj]['coin_pulse_areas_sum']) & (ddicts[jj]['coin_pulse_areas_sum']<10000)
coinsel9 = (10000<=ddicts[jj]['coin_pulse_areas_sum']) & (ddicts[jj]['coin_pulse_areas_sum']<30000)
coinsel10 = (30000<=ddicts[jj]['coin_pulse_areas_sum']) & (ddicts[jj]['coin_pulse_areas_sum']<100000)
coinsel11 = (100000<=ddicts[jj]['coin_pulse_areas_sum']) 

coins = [
coinsel, coinsel1| coinsel2, coinsel3| coinsel4, 
coinsel5| coinsel6, coinsel7| coinsel8, coinsel9|coinsel10|coinsel11, 
]


labellist = [
        'all',
        '[10'+r'$^{0}$'+' ,10'+r'$^{1}$'+')',
        '[10'+r'$^{1}$'+' ,10'+r'$^{2}$'+')',
        '[10'+r'$^{2}$'+' ,10'+r'$^{3}$'+')',
        '[10'+r'$^{3}$'+' ,10'+r'$^{4}$'+')',
#        '[10'+r'$^{4}$'+' ,10'+r'$^{5}$'+')',
        '[10'+r'$^{4}$'+' ,'+r'$\infty$'+')',
]



areaslist = [
#np.zeros_like(ddicts[jj]['coin_pulse_areas_post_10us']),
#ddicts[jj]['coin_pulse_areas_post_10us'],
ddicts[jj]['coin_pulse_areas_post_100us'] - ddicts[jj]['coin_pulse_areas_post_10us'],
ddicts[jj]['coin_pulse_areas_post_1000us'] - ddicts[jj]['coin_pulse_areas_post_100us'],
ddicts[jj]['coin_pulse_areas_post_10000us'] - ddicts[jj]['coin_pulse_areas_post_1000us'],      

]

normlist=[ # duration of the post area in second.
 #       10.e-6,
        90.e-6,
        900.e-6,
        9000.e-6,
        ]

legendlist = [
#        '[0'+' ,10'+r'$^{1}$'+')' + ' [us]',
        '[10'+r'$^{1}$'+' ,10'+r'$^{2}$'+')' + ' [us]',
        '[10'+r'$^{2}$'+' ,10'+r'$^{3}$'+')' + ' [us]',
        '[10'+r'$^{3}$'+' ,10'+r'$^{4}$'+')' + ' [us]',
#        '[10'+r'$^{4}$'+' ,10'+r'$^{5}$'+')' + ' [us]',
]




yrange = np.arange(len(coins))

kk=0
topaf = []
for ll in range(len(coins)):
    topaf.append( areaslist[kk][:,0][coins[ll]]/normlist[kk])

topmed =[]
topmean =[]
topquar1 =[]
topquar3 =[]
botmed =[]
botmean =[]
botquar1 =[]
botquar3 =[]

for ll in range(len(coins)):
    print (labellist[ll], (coins[ll]).sum())
    topmed.append( np.median(areaslist[kk][:,0][coins[ll]])/normlist[kk])
    topmean.append( np.mean(areaslist[kk][:,0][coins[ll]])/normlist[kk])
    topquar1.append(np.percentile(areaslist[kk][:,0][coins[ll]],14)/normlist[kk])
    topquar3.append(np.percentile(areaslist[kk][:,0][coins[ll]],86)/normlist[kk])
    botmed.append( np.median(areaslist[kk][:,2][coins[ll]])/normlist[kk])
    botmean.append( np.mean(areaslist[kk][:,2][coins[ll]])/normlist[kk])
    botquar1.append(np.percentile(areaslist[kk][:,2][coins[ll]],14)/normlist[kk])
    botquar3.append(np.percentile(areaslist[kk][:,2][coins[ll]],86)/normlist[kk])

'''
#ax.boxplot(topaf, 0 , '', vert=False, labels=labellist)
#ax.plot(topmean, yrange, ls = ':', color='r', label = 'mean')
ax.plot(topmed, yrange, ls = '-', color='r', label = 'top (median + 1 '+r'$\sigma$'+')')
ax.plot(topquar1, yrange, ls = ':', color='r')
ax.plot(topquar3, yrange, ls = ':', color='r')
#ax.plot(botmean, yrange, ls = ':', color='b', label = 'mean')
ax.plot(botmed, yrange, ls = '-', color='b', label = 'bottom (median + 1 '+r'$\sigma$'+')')
ax.plot(botquar1, yrange, ls = ':', color='b')
ax.plot(botquar3, yrange, ls = ':', color='b')
'''


fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)
for nn in range(len(areaslist)):
    topmed =[]
    botmed =[]
    topquar1 =[]
    topquar3 =[]
    botquar1 =[]
    botquar3 =[]
    for ll in range(len(coins)):
        topmed.append( np.median(areaslist[nn][:,0][coins[ll]])/normlist[nn])
#        botmed.append( np.median(areaslist[nn][:,2][coins[ll]])/normlist[nn])
        topquar1.append(np.percentile(areaslist[nn][:,0][coins[ll]],14)/normlist[nn])
        topquar3.append(np.percentile(areaslist[nn][:,0][coins[ll]],86)/normlist[nn])
        botquar1.append(np.percentile(areaslist[nn][:,2][coins[ll]],14)/normlist[nn])
        botquar3.append(np.percentile(areaslist[nn][:,2][coins[ll]],86)/normlist[nn])

    ax.plot(topmed, yrange, ls = '-', color=color_list[2*nn+2], label = 'post ' + legendlist[nn])
#    ax.plot(topquar1, yrange, ls = ':', color=color_list[2*nn+2])
#    ax.plot(topquar3, yrange, ls = ':', color=color_list[2*nn+2])
#    ax.plot(botmed, yrange, ls = '-', color=color_list[2*nn+2], label = 'post ' + legendlist[nn])
#    ax.plot(botquar1, yrange, ls = ':', color=color_list[2*nn+2])
#    ax.plot(botquar3, yrange, ls = ':', color=color_list[2*nn+2])
#

#ax.axvline(np.mean(ddicts[jj]['random_pulse_areas_post_100us'][:,0])/100.e-6, ls = ':', color='k', label='random sample 100 [us]')
ax.set_yticks(yrange)
ax.set_yticklabels(labellist) 
ax.set_xlabel('photon rate [phe s'+r'$^{-1}$'+']')
ax.set_ylabel('signal area [phe]')
ax.set_xticks(np.array([1.e2,1.e3,1.e4,1.e5,1.e6]))
ax.set_xlim(1.e2, 1.e5)
#ax.set_ylim(1.e2, 1.e7)
ax.set_xscale('log')
#ax.set_yscale('log')
ax.grid('on')
box = ax.get_position()
ax.set_position([box.x0+box.width*.25, box.y0+box.height*0.0, box.width*.75, box.height*0.7])
ax.legend(loc=3, bbox_to_anchor=(0., 1.1, 1., .102), ncol=1, mode="expand", borderaxespad=0.,fontsize=18)

plt.savefig(workdir+'/topPMTPTFEFluoEva'+str(ddicts[0]['procid'][0])+'.png')


fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)

for nn in range(len(areaslist)):
    topmed =[]
    botmed =[]
    for ll in range(len(coins)):
#        topmed.append( np.median(areaslist[nn][:,0][coins[ll]])/normlist[nn])
        botmed.append( np.median(areaslist[nn][:,2][coins[ll]])/normlist[nn])
#    ax.plot(topmed, yrange, ls = '-', color=color_list[2*nn+2], label = 'post ' + legendlist[nn])
    ax.plot(botmed, yrange, ls = '-', color=color_list[2*nn+2], label = 'post ' + legendlist[nn])

#ax.axvline(np.mean(ddicts[jj]['random_pulse_areas_post_100us'][:,2])/100.e-6, ls = ':', color='k', label='random sample 100 [us]')
ax.set_yticks(yrange)
ax.set_yticklabels(labellist) 
ax.set_xlabel('photon rate [phe s'+r'$^{-1}$'+']')
ax.set_ylabel('signal area [phe]')
ax.set_xticks(np.array([1.e2,1.e3,1.e4,1.e5,1.e6]), minor=False)
ax.set_xlim(1.e2, 1.e5)
#ax.tick_params(axis='x',which='minor',bottom='off')
#ax.locator_params(axis='x', nbins=4)
#ax.set_ylim(1.e2, 1.e7)
ax.set_xscale('log')
#ax.set_yscale('log')
ax.grid('on')
box = ax.get_position()
ax.set_position([box.x0+box.width*.25, box.y0+box.height*0.0, box.width*.75, box.height*0.7])
ax.legend(loc=3, bbox_to_anchor=(0., 1.1, 1., .102), ncol=1, mode="expand", borderaxespad=0.,fontsize=18)

plt.savefig(workdir+'/botPMTPTFEFluoEva'+str(ddicts[0]['procid'][0])+'.png')
#plt.close('all')


####PTFE fluo rate
uf.LoadDictWaveform(procids[0], ddicts[0], [np.arange(len(ddicts[0]['times']))] , datadir)#load waveforms from process files.
csel=coinsel9|coinsel10|coinsel11#|coinsel7|coinsel8
t0=(ddicts[0]['coin_pulse_times']+ddicts[0]['coin_pulse_lens'])[csel]
a0=(ddicts[0]['coin_pulse_areas_sum'])[csel]


nptotal=len(t0) #100#
ntotal=10
totalsam=250000
avnsam = 1000
avsam=totalsam/avnsam 
waveforms_post_tops=np.zeros(totalsam)
waveforms_post_bots=np.zeros(totalsam)
waveforms_times = np.arange(totalsam)*4/1.e3
waveforms_times_av = np.zeros(avsam)
waveforms_post_tops_av  = np.zeros(avsam)
waveforms_post_bots_av  = np.zeros(avsam)


fitbeg,fitend=50,150
for qq in range(250):
    waveforms_times_av[qq] = waveforms_times[qq*1000:(qq+1)*1000].mean()
for hh in range(nptotal):
    if hh%100==0:
        print hh
    t00=t0[hh]
    waveforms_post = uf.GetWaveformInTimeRange(t00, t00+1.e6, ddicts[0])
    waveforms_post_topn = waveforms_post[0]/waveforms_post[0,].sum()/4.
    waveforms_post_botn = waveforms_post[2]/waveforms_post[2,].sum()/4.
#    waveforms_post_topn = waveforms_post[0]/a0[hh]
#    waveforms_post_botn = waveforms_post[2]/a0[hh]
    if (waveforms_post[0,].sum()<=0):
        waveforms_post_topn = np.zeros_like(waveforms_post_topn)
    if (waveforms_post[2,].sum()<=0):
        waveforms_post_botn = np.zeros_like(waveforms_post_botn)        
    waveforms_post_tops = waveforms_post_tops +  waveforms_post_topn
    waveforms_post_bots = waveforms_post_bots +  waveforms_post_botn
#    if (waveforms_post[0].sum()>0) & (waveforms_post[2].sum()>0) & (ntotal>0):
#        for qq in range(avsam):
#            waveforms_post_tops_av[qq]= waveforms_post_topn[qq*avnsam:(qq+1)*avnsam].mean()
#            waveforms_post_bots_av[qq]= waveforms_post_botn[qq*avnsam:(qq+1)*avnsam].mean() 
#        ax.plot(waveforms_times_av, waveforms_post_tops_av)
        
waveforms_post_topa = waveforms_post_tops /nptotal
waveforms_post_bota = waveforms_post_bots /nptotal

#waveforms_post_tops_av  = np.zeros(avsam)
#waveforms_post_bots_av  = np.zeros(avsam)

for qq in range(avsam):
    waveforms_post_tops_av[qq]= waveforms_post_topa[qq*avnsam:(qq+1)*avnsam].mean()
    waveforms_post_bots_av[qq]= waveforms_post_bota[qq*avnsam:(qq+1)*avnsam].mean()

fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)   
ax.plot(waveforms_times_av, waveforms_post_tops_av, color='r', ls=':', label='top (average)') 
ax.plot(waveforms_times_av, waveforms_post_bots_av, color='b', ls=':', label='bottom (average)')


tfit = waveforms_times_av[fitbeg:fitend] 
p, V = np.polyfit(tfit ,np.log(waveforms_post_tops_av[fitbeg:fitend]) , 1, cov=True)
ax.plot(tfit, np.exp(p[1]+p[0]*tfit), color='r', ls='-', label='top fit ('+r'$\tau$'+' = %.0f '%(-1./p[0])+r'$\pm$'+' %.0f [us])'%(np.sqrt(V[0][0])/p[0]**2))
p, V = np.polyfit(tfit ,np.log(waveforms_post_bots_av[fitbeg:fitend]) , 1, cov=True)
ax.plot(tfit, np.exp(p[1]+p[0]*tfit), color='b', ls='-', label='bottom fit ('+r'$\tau$'+' = %.0f '%(-1./p[0])+r'$\pm$'+' %.0f [us])'%(np.sqrt(V[0][0])/p[0]**2))

waveforms_post_tots_av =(waveforms_post_tops_av +waveforms_post_bots_av)/2.
p, V = np.polyfit(tfit ,np.log(waveforms_post_tots_av[fitbeg:fitend]) , 1, cov=True)
ax.plot(tfit, np.exp(p[1]+p[0]*tfit), color='k', ls='-', label='total fit ('+r'$\tau$'+' = %.0f '%(-1./p[0])+r'$\pm$'+' %.0f [us])'%(np.sqrt(V[0][0])/p[0]**2))


ax.set_xlabel('post signal time [us]')
ax.set_ylabel('scaled photon rate')
#ax.set_xlim(1.e2, 1.e5)
ax.set_ylim(1.e-7, 1.e-5)
#ax.set_xscale('log')
ax.set_yscale('log')
ax.grid('on')
box = ax.get_position()
ax.set_position([box.x0+box.width*.0, box.y0+box.height*0.0, box.width*1., box.height*0.7])
ax.legend(loc=3, bbox_to_anchor=(0., 1.1, 1., .102), ncol=1, mode="expand", borderaxespad=0.,fontsize=17)

plt.savefig(workdir+'/PMTPTFEFluoTau'+str(ddicts[0]['procid'][0])+'.png')

sphe_size=np.array([413,np.inf,610, np.inf])
sample_size =4
atoppost=np.zeros_like(a0)
abotpost=np.zeros_like(a0)
for hh in range(nptotal):
    if hh%100==0:
        print hh
    t00=t0[hh]
    waveforms_post = uf.GetWaveformInTimeRange(t00+2.e5, t00+6.e5, ddicts[0])
    atoppost[hh] = waveforms_post[0].sum()/sphe_size[0]*sample_size
    abotpost[hh] = waveforms_post[2].sum()/sphe_size[2]*sample_size


#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command

fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)   
#ax.hist(atoppost/a0, range=[1.e-5,1.e-3], bins=99, color='r', histtype='step', lw=3)
#ax.hist(abotpost/a0, range=[1.e-5,1.e-3], bins=99, color='b', histtype='step', lw=3)
#ax.hist((atoppost+abotpost)/a0, range=[1.e-5,1.e-3], bins=99, color='k', histtype='step', lw=3)
boxprops = dict(linestyle='-', linewidth=3, color='k')
medianprops= dict(linestyle='-', linewidth=2.5, color='orange')
ax.boxplot([atoppost/a0,abotpost/a0,(atoppost+abotpost)/a0], 0 , '', vert=False, labels=['top', 'bottom', 'total'], boxprops=boxprops, medianprops=medianprops)#,showmeans=True

#ax.set_xlabel(r'$\frac{\text{area in post 200 to 600 us [phe]}}{\text{signal area [phe]}}$')
ax.set_xlabel(r'$\frac{\mathrm{area\ in\ post\ 200\ to\ 600\ us\ [phe]}}{\mathrm{signal\ area\ [phe]}}$')

ax.set_ylabel('')
#ax.set_xlim(1.e2, 1.e5)
#ax.set_ylim(1.e-7, 1.e-5)
#ax.set_xscale('log')
#ax.set_yscale('log')
ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
ax.grid('on')
box = ax.get_position()
ax.set_position([box.x0+box.width*.0, box.y0+box.height*0.1, box.width*1., box.height*0.8])
#ax.legend(loc=3, bbox_to_anchor=(0., 1.1, 1., .102), ncol=1, mode="expand", borderaxespad=0.,fontsize=17)
ax.xaxis.set_label_coords(.4, -0.12)
plt.savefig(workdir+'/PMTPTFEFluoRatio'+str(ddicts[0]['procid'][0])+'.png')

totrate =(atoppost+abotpost)/a0
totratesel= totrate[~np.isnan(totrate)] 
print np.median(totratesel)

#mpl.rcParams['text.usetex'] = False



