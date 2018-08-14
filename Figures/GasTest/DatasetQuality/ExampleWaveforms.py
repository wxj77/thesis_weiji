#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 14:20:39 2018

@author: weiji
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 09:58:08 2018

@author: wxj
"""


import os, sys
wd='/Users/weiji/Google Drive/gastest/'
workdir=wd + '/exampleWaveforms/'#'/home/wxj/gastest/DatasetQuality/'
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
'''
ddicts = []
#for jj in range(porcidnum):#range(15,20):#
procids = []
datadir="/Users/weiji/Google Drive/"
procids.append([64767,]) #13001,65901,65831,10201, 33001
for jj in [0]:#range(15,20):#12 kV
    print "Load dict:", jj, '\t', procids[jj]
    try:
#        ddict = uf.LoadDict(procids[jj], keys + newrqlist, nonnpykeys, datadir)
        ddict = uf.LoadCoinPulseRecord(procids[jj], keys+newrqlist+waveform_list+['waveforms'], {}, datadir)
        ddicts.append(ddict)
        ddicts[0]['procid'] = procids[0]
    except:
        print "cannot load dict:", jj, '\t', procids[jj]
    del ddict
'''


# plot continuous event separately
ddicts = []
procids = []
datadir="/Users/weiji/Google Drive/"
procids.append([64767,]) #13001,65901,65831,10201, 33001
for jj in [0]:#range(15,20):#12 kV
    print "Load dict:", jj, '\t', procids[jj]
    try:
        ddict = uf.LoadDict(procids[jj], keys + newrqlist+nonnpykeys, nonnpykeys, datadir)
        ddict = uf.LoadCoinPulseRecord(procids[jj], keys+newrqlist+waveform_list+['waveforms'], ddict, datadir)
        ddicts.append(ddict)
        ddicts[0]['procid'] = procids[0]
    except:
        print "cannot load dict:", jj, '\t', procids[jj]
    del ddict

replot=False
plt.close('all')
for ii in range(10000):
    plt.close('all')
    if replot:
        uf.PlotCoinWaveform(ddicts[0],ii,plot_savitzky_golay=False,window_savitzky_golay=51, savename=workdir+'proc'+str(ddicts[0]['procid'][0])+'id%08d'%(ii)+'.png',fignum=ii,sample_size=4, showarea=False, time0=-120)

plt.close('all')

plt.close('all')

# muon event proc64747 coin id 206,207 , seperate id 2474,2475,2476,2477
plotids=[2474,2475,2476,2477]
tt0 = np.min(ddicts[0]['times'][plotids])
uf.PlotWaveformList(ddicts[0],plotids,plot_savitzky_golay=False,window_savitzky_golay=51, savename=workdir+'proc'+str(ddicts[0]['procid'][0])+'AnodeMuon1'+'.png',fignum=fignum,sample_size=4, showarea=False, time0=-120)
plotid=206
uf.PlotCoinWaveform(ddicts[0],plotid,plot_savitzky_golay=False,window_savitzky_golay=51, savename=workdir+'proc'+str(ddicts[0]['procid'][0])+'AnodeMuon1P1'+'.png',fignum=plotid,sample_size=4, showarea=False, time0=ddicts[0]['coin_pulse_times'][plotid]-tt0-120)
plotid=207
uf.PlotCoinWaveform(ddicts[0],plotid,plot_savitzky_golay=False,window_savitzky_golay=51, savename=workdir+'proc'+str(ddicts[0]['procid'][0])+'AnodeMuon1P2'+'.png',fignum=plotid,sample_size=4, showarea=False, time0=ddicts[0]['coin_pulse_times'][plotid]-tt0-120)
plt.close('all')

# anode cone event proc64747 coin id 76,77 , seperate id 863-869
plotids=range(863,867+1)
tt0 = np.min(ddicts[0]['times'][plotids])
uf.PlotWaveformList(ddicts[0],plotids,plot_savitzky_golay=False,window_savitzky_golay=51, savename=workdir+'proc'+str(ddicts[0]['procid'][0])+'AnodeCone1'+'.png',fignum=fignum,sample_size=4, showarea=False, time0=-120)
plotid=76
uf.PlotCoinWaveform(ddicts[0],plotid,plot_savitzky_golay=False,window_savitzky_golay=51, savename=workdir+'proc'+str(ddicts[0]['procid'][0])+'AnodeCone1P1'+'.png',fignum=plotid,sample_size=4, showarea=False, time0=ddicts[0]['coin_pulse_times'][plotid]-tt0-120)
plotid=77
uf.PlotCoinWaveform(ddicts[0],plotid,plot_savitzky_golay=False,window_savitzky_golay=51, savename=workdir+'proc'+str(ddicts[0]['procid'][0])+'AnodeCone1P2'+'.png',fignum=plotid,sample_size=4, showarea=False, time0=ddicts[0]['coin_pulse_times'][plotid]-tt0-120)
plt.close('all')

# PTFE fluorescence event proc64747 coin id 117,118 , seperate id 1276-1285,
plotids=range(1816,1851+1)
tt0 = np.min(ddicts[0]['times'][plotids])
tt1 = np.max(ddicts[0]['times'][plotids] + ddicts[0]['wavelens'][plotids]+8)
uf.PlotWaveformList(ddicts[0],plotids,plot_savitzky_golay=False,window_savitzky_golay=51, savename=workdir+'proc'+str(ddicts[0]['procid'][0])+'PTFEFluo1'+'.png',fignum=fignum,sample_size=4, showarea=False, time0=-120)
plotid=160
uf.PlotCoinWaveform(ddicts[0],plotid,plot_savitzky_golay=False,window_savitzky_golay=51, savename=workdir+'proc'+str(ddicts[0]['procid'][0])+'PTFEFluo1P1'+'.png',fignum=plotid,sample_size=4, showarea=False, time0=ddicts[0]['coin_pulse_times'][plotid]-tt0-120)
plotid=161
uf.PlotCoinWaveform(ddicts[0],plotid,plot_savitzky_golay=False,window_savitzky_golay=51, savename=workdir+'proc'+str(ddicts[0]['procid'][0])+'PTFEFluo1P2'+'.png',fignum=plotid,sample_size=4, showarea=False, time0=ddicts[0]['coin_pulse_times'][plotid]-tt0-120)
plt.close('all')

uf.PlotWaveformList(ddicts[0],plotids,plot_savitzky_golay=False,window_savitzky_golay=51, savename=workdir+'proc'+str(ddicts[0]['procid'][0])+'PTFEFluo1x'+'.png',fignum=fignum,sample_size=4, showarea=False, time0=-120-2.e6, timerange=True, timestart=tt0-2.e6, timestop=tt1-2.e6)
uf.PlotWaveformList(ddicts[0],plotids,plot_savitzky_golay=False,window_savitzky_golay=51, savename=workdir+'proc'+str(ddicts[0]['procid'][0])+'PTFEFluo1long'+'.png',fignum=fignum,sample_size=4, showarea=False, time0=-120-1.e8, timerange=True, timestart=tt0-1.e8, timestop=tt0+1.e8)

# dead time event proc64747 coin id 65, seperate id 718-720,
a = (ddicts[0]['waveareas']>1.e5)[1:-1]
bb = np.diff(ddicts[0]['times'])-ddicts[0]['wavelens'][:-1]
b = (bb[:-1] >5.e3)  & (bb[1:] > 10.e4) #& (bb[:-1] <1.e4)
c = np.where(a&b)[0]+1
#plotids=range(c[9]-2,c[9]+2)
plotids=range(14096,14100-1)
uf.PlotWaveformListSimple(ddicts[0],plotids,plot_savitzky_golay=False,window_savitzky_golay=51, savename=workdir+'proc'+str(ddicts[0]['procid'][0])+'DeadTime1'+'.png',fignum=fignum,sample_size=4, showarea=False, time0=-120)
fignum+=1
plotids=range(51429,51433-1)
uf.PlotWaveformListSimple(ddicts[0],plotids,plot_savitzky_golay=False,window_savitzky_golay=51, savename=workdir+'proc'+str(ddicts[0]['procid'][0])+'DeadTime2'+'.png',fignum=fignum,sample_size=4, showarea=False, time0=-120)
fignum+=1
plt.close('all')


