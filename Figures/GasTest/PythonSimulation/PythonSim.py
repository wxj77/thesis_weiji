# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import random as random
import matplotlib.pyplot as plt
import scipy.stats as stats
savedir='/media/wei/ACA8-1ECD3/PythonSimulation/'


import os, sys
wd= '/media/wei/ACA8-1ECD3/gastest/' #'/Users/weiji/Google Drive/gastest/' #'/home/wxj/gastest/' # '/media/wei/ACA8-1ECD3/gastest/' #
workdir=wd
try:
    os.makedirs(workdir)
except:
    i=1

sys.path.insert(0, wd+"/obsolete/PythonScripts/RunSetups/")
sys.path.insert(0, wd+"/obsolete/PythonScripts/")
sys.path.insert(0, wd+"/obsolete/PythonScript/")
sys.path.insert(0, wd+"/xenonProperties/")
workdir = wd + "/xenonProperties/"
savedir =wd + "/PythonSimulation/"
sys.path.insert(0, workdir)
try:
    os.makedirs(savedir)
except:
    i=1

savedir='/media/wei/ACA8-1ECD3/PythonSimulation/'

from color import *
from plotStyle import *
from matplotlib.patches import Polygon

class A(object):
    def __init__(self):
        self.b = 1
        self.c = 2
    def do_nothing(self):
        pass

import pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
        
# generate a simple simulated waveform for an average phe.
WaveformSimple = np.concatenate((np.ones(30), np.zeros(0)), axis=0) #100ns phe

# generate a list of simple wave for with the average phe wave form and a facor in width and height. the factor follows N(1, 0.2), 
mu=1
sigma=0.4
num=5000
WaveformList=[]
for ii in range(num):
    factor = np.random.normal(mu, sigma, 1)
    Waveform = WaveformSimple*factor
    WaveformList.append(Waveform)

AreaList=[]    
for ii in range(num):
    AreaList.append(np.sum(WaveformList[ii]))    

ddict={}
NumOfSim=500000
PhotonNum_list=[]
for nn in range(2,20):
    PhotonNum_list.append(nn)

for nn in range(20,60,2):
    PhotonNum_list.append(nn)

for nn in range(60,100,4):
    PhotonNum_list.append(nn)

for nn in range(100,200,10):
    PhotonNum_list.append(nn)

for nn in range(200,500,25):
    PhotonNum_list.append(nn)

for nn in range(500,1500,50):
    PhotonNum_list.append(nn)

#for nn in PhotonNum_list:
#    print(nn)

#print(len(PhotonNum_list))

for TotalTime in [10000]: #3000 total drift time
    ddict[TotalTime]={}
    for NumOfPhoton in PhotonNum_list:#range(1,150):ddict[TotalTime][xx]
#        print(NumOfPhoton)
        ddict[TotalTime][NumOfPhoton]={}
        ddict[TotalTime][NumOfPhoton]['t01']=np.zeros(NumOfSim)
        ddict[TotalTime][NumOfPhoton]['t05']=np.zeros(NumOfSim)
        ddict[TotalTime][NumOfPhoton]['t10']=np.zeros(NumOfSim)
        ddict[TotalTime][NumOfPhoton]['t15']=np.zeros(NumOfSim)
        ddict[TotalTime][NumOfPhoton]['t25']=np.zeros(NumOfSim)
        ddict[TotalTime][NumOfPhoton]['t50']=np.zeros(NumOfSim)
        ddict[TotalTime][NumOfPhoton]['t75']=np.zeros(NumOfSim)
        ddict[TotalTime][NumOfPhoton]['t85']=np.zeros(NumOfSim)
        ddict[TotalTime][NumOfPhoton]['t90']=np.zeros(NumOfSim)
        ddict[TotalTime][NumOfPhoton]['t95']=np.zeros(NumOfSim)
        ddict[TotalTime][NumOfPhoton]['t99']=np.zeros(NumOfSim)
        ddict[TotalTime][NumOfPhoton]['mean']=np.zeros(NumOfSim)
        ddict[TotalTime][NumOfPhoton]['std']=np.zeros(NumOfSim)
        ddict[TotalTime][NumOfPhoton]['skew']=np.zeros(NumOfSim)
        ddict[TotalTime][NumOfPhoton]['kurtosis']=np.zeros(NumOfSim)
        ddict[TotalTime][NumOfPhoton]['skew0595']=np.zeros(NumOfSim)
        ddict[TotalTime][NumOfPhoton]['kurtosis0595']=np.zeros(NumOfSim)
        for ii in range(NumOfSim):
            Waveform= np.zeros(TotalTime+1000)
            for jj in range(NumOfPhoton):
                timestart = random.randint(0,TotalTime-1)
                UsePulse= WaveformList[random.randint(0,num-1)]
                Waveform[timestart:timestart+len(UsePulse)]+=UsePulse
            sumWaveform = np.cumsum(Waveform)
            t01 = np.where(sumWaveform>0.01*sumWaveform[-1])[0][0]
            t05 = np.where(sumWaveform>0.05*sumWaveform[-1])[0][0]
            t10 = np.where(sumWaveform>0.10*sumWaveform[-1])[0][0]
            t15 = np.where(sumWaveform>0.15*sumWaveform[-1])[0][0]
            t25 = np.where(sumWaveform>0.25*sumWaveform[-1])[0][0]
            t50 = np.where(sumWaveform>0.50*sumWaveform[-1])[0][0]
            t75 = np.where(sumWaveform>0.75*sumWaveform[-1])[0][0]
            t85 = np.where(sumWaveform>0.85*sumWaveform[-1])[0][0]
            t90 = np.where(sumWaveform>0.90*sumWaveform[-1])[0][0]
            t95 = np.where(sumWaveform>0.95*sumWaveform[-1])[0][0]
            t99 = np.where(sumWaveform>0.99*sumWaveform[-1])[0][0]
            ddict[TotalTime][NumOfPhoton]['t01'][ii]=t01
            ddict[TotalTime][NumOfPhoton]['t05'][ii]=t05
            ddict[TotalTime][NumOfPhoton]['t10'][ii]=t10
            ddict[TotalTime][NumOfPhoton]['t15'][ii]=t15
            ddict[TotalTime][NumOfPhoton]['t25'][ii]=t25
            ddict[TotalTime][NumOfPhoton]['t50'][ii]=t50
            ddict[TotalTime][NumOfPhoton]['t75'][ii]=t75
            ddict[TotalTime][NumOfPhoton]['t85'][ii]=t85
            ddict[TotalTime][NumOfPhoton]['t90'][ii]=t90
            ddict[TotalTime][NumOfPhoton]['t95'][ii]=t95
            ddict[TotalTime][NumOfPhoton]['t99'][ii]=t99
            ddict[TotalTime][NumOfPhoton]['mean'][ii]=np.mean(Waveform)
            ddict[TotalTime][NumOfPhoton]['std'][ii]=np.std(Waveform)
            ddict[TotalTime][NumOfPhoton]['skew'][ii]=stats.skew(Waveform)
            ddict[TotalTime][NumOfPhoton]['kurtosis'][ii]=stats.kurtosis(Waveform)
            ddict[TotalTime][NumOfPhoton]['skew0595'][ii]=stats.skew(Waveform[t05:t95])
            ddict[TotalTime][NumOfPhoton]['kurtosis0595'][ii]=stats.kurtosis(Waveform[t05:t95])




''' 
ddict_gau={}
NumOfSim=1000
for TotalTime in [10000]: #3000 total drift time
    ddict_gau[TotalTime]={}
    for NumOfPhoton in range(10,1500,10):
        ddict_gau[TotalTime][NumOfPhoton]={}
        t2575=[]
        t0595=[]
        t1090=[]
        t50=[]
        for ii in range(NumOfSim):
            Waveform= np.zeros(TotalTime+1000)
            for jj in range(NumOfPhoton):
                timestart = np.random.normal(TotalTime/2., TotalTime/6.,1)
                timestart=int(timestart)
                if (timestart<0 or timestart>TotalTime+900):
                    continue
                UsePulse= WaveformList[random.randint(0,num-1)]
                Waveform[timestart:timestart+len(UsePulse)]+=UsePulse
            sumWaveform = np.cumsum(Waveform)
            t05 = np.where(sumWaveform>0.05*sumWaveform[-1])[0][0]
            t10 = np.where(sumWaveform>0.10*sumWaveform[-1])[0][0]
            t25 = np.where(sumWaveform>0.25*sumWaveform[-1])[0][0]
            T50 = np.where(sumWaveform>0.50*sumWaveform[-1])[0][0]
            t75 = np.where(sumWaveform>0.75*sumWaveform[-1])[0][0]
            t90 = np.where(sumWaveform>0.90*sumWaveform[-1])[0][0]
            t95 = np.where(sumWaveform>0.95*sumWaveform[-1])[0][0]
            t2575.append(t75-t25)
            t0595.append(t95-t05)
            t1090.append(t90-t10)
            t50.append(T50)
        ddict_gau[TotalTime][NumOfPhoton]['t2575']=np.array(t2575)
        ddict_gau[TotalTime][NumOfPhoton]['t0595']=np.array(t0595)
        ddict_gau[TotalTime][NumOfPhoton]['t1090']=np.array(t1090)
        ddict_gau[TotalTime][NumOfPhoton]['t50']=np.array(t50)
'''          
fignum=100
plt.close('all')

'''
for TotalTime in [10000]: #3000 total drift time
    for NumOfPhoton in range(1,150):  
        plt.figure(fignum)
        fignum+=1
        plt.hist2d(ddict[TotalTime][NumOfPhoton]['t2575']/TotalTime,ddict[TotalTime][NumOfPhoton]['t1090']/TotalTime, bins=[25,25], range=[[0.25,0.75],[0.5,1.0]])
        plt.xlabel('t2575 /t_total')
        plt.ylabel('t1090 /t_total')
        plt.colorbar()
        plt.clim(0,10)
        plt.savefig(savedir+'PhotonSim_TotalTime%d_NumOfPhoton_%d.png'%(TotalTime, NumOfPhoton))

plt.close('all')
for TotalTime in [10000]: #3000 total drift time
    for NumOfPhoton in range(1,150):  
        plt.figure(fignum)
        fignum+=1
        plt.hist(ddict[TotalTime][NumOfPhoton]['t1090']/TotalTime, bins=50, range=[0.5,1.0])
        plt.ylabel('Counts')
        plt.xlabel('t1090 /t_total')
#        plt.colorbar()
#        plt.clim(0,10)
        plt.savefig(savedir+'PhotonSim_TotalTime%d_T1090Hist_NumOfPhoton_%d.png'%(TotalTime, NumOfPhoton))

             
for TotalTime in [10000]: #3000 total drift time
    fig = plt.figure(fignum)
    ax = fig.add_subplot(1,1,1)
    xlist=[]
    ylist=[]
    for NumOfPhoton in range(1,100): 
        xlist.append(NumOfPhoton)
        sel = (np.array(ddict[TotalTime][NumOfPhoton]['t2575']) > TotalTime*0.35) &\
            (np.array(ddict[TotalTime][NumOfPhoton]['t2575']) < TotalTime*0.65) &\
            (np.array(ddict[TotalTime][NumOfPhoton]['t0595']) > TotalTime*0.75) &\
            (np.array(ddict[TotalTime][NumOfPhoton]['t0595']) < TotalTime*1.05) 
        ylist.append(float(np.sum(sel))/float(len(sel)))        
    plt.plot(xlist,ylist, label=r't2575/total $\in$ (0.35, 0.65)'+'\n'+r'and t0595/total $\in$ (0.75, 1.05)')
    xlist=[]
    ylist=[]
    for NumOfPhoton in range(1,100): 
        xlist.append(NumOfPhoton)
        sel = (np.array(ddict[TotalTime][NumOfPhoton]['t2575']) > TotalTime*0.35) &\
            (np.array(ddict[TotalTime][NumOfPhoton]['t2575']) < TotalTime*0.65) #&\
            # (np.array(ddict[TotalTime][NumOfPhoton]['t0595']) > TotalTime*0.75) &\
            #(np.array(ddict[TotalTime][NumOfPhoton]['t0595']) < TotalTime*1.05) 
        ylist.append(float(np.sum(sel))/float(len(sel)))        
    plt.plot(xlist,ylist, label=r't2575/total $\in$ (0.35, 0.65)')
    xlist=[]
    ylist=[]
    for NumOfPhoton in range(1,100): 
        xlist.append(NumOfPhoton)
        sel =(np.array(ddict[TotalTime][NumOfPhoton]['t0595']) > TotalTime*0.75) &\
            (np.array(ddict[TotalTime][NumOfPhoton]['t0595']) < TotalTime*1.05) 
        ylist.append(float(np.sum(sel))/float(len(sel)))        
    plt.plot(xlist,ylist, label=r't0595/total $\in$ (0.75, 1.05)')
    xlist=[]
    ylist=[]
    for NumOfPhoton in range(1,100): 
        xlist.append(NumOfPhoton)
        sel =(np.array(ddict[TotalTime][NumOfPhoton]['t1090']) > TotalTime*0.65) &\
            (np.array(ddict[TotalTime][NumOfPhoton]['t1090']) < TotalTime*0.95) 
        ylist.append(float(np.sum(sel))/float(len(sel)))        
    plt.plot(xlist,ylist, label=r't1090/total $\in$ (0.65, 0.95)')
    ax.set_ylim(0.4,1.05)
    ax.set_xlabel('Num of Phtons')
    ax.set_ylabel('selection efficiency')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height*0.6])
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1, mode="expand", borderaxespad=0.)
    plt.grid('on')
    plt.savefig(savedir+'PhotonSim_TotalTime%d_SelectionEfficiency.png'%(TotalTime))
    
             
for TotalTime in [10000]: #3000 total drift time
    fig = plt.figure(fignum)
    ax = fig.add_subplot(1,1,1)
    xlist=[]
    ylist=[]
    for NumOfPhoton in range(1,100): 
        xlist.append(NumOfPhoton)
        sel = (np.array(ddict[TotalTime][NumOfPhoton]['t2575']) > TotalTime*0.10)
        ylist.append(float(np.sum(sel))/float(len(sel)))        
    plt.plot(xlist,ylist, label=r't2575/total $\in$ (0.35, 0.65)'+'\n'+r'and t0595/total $\in$ (0.75, 1.05)')
    
  
plt.close('all')
for TotalTime in [10000]: #3000 total drift time
    xlist=[]
    ylist=[]
    zlist=[]
    fig = plt.figure(fignum)
    fignum+=1
    ax = fig.add_subplot(1,1,1)
    for NumOfPhoton in range(1,150):  
        xlist.append(NumOfPhoton)
        ylist.append(np.mean(ddict[TotalTime][NumOfPhoton]['t1090']/TotalTime))
        zlist.append(np.std(ddict[TotalTime][NumOfPhoton]['t1090']/TotalTime))
    ax.errorbar(xlist, ylist, yerr=zlist, fmt='o')
    ax.set_xlabel('Num of Phtons')
    ax.set_ylabel('t1090/t_total')
    ax.grid('on')
    ax.set_xlim(0,150)
    ax.set_ylim(0.5,0.95)
    plt.savefig(savedir+'PhotonSim_TotalTime%d_T1090_MeanStd.png'%(TotalTime))
    
a = np.array([xlist, ylist, zlist])
np.save(savedir+'PhotonNum_T1090Mean_T1090Std',a)    
   
plt.close('all')
for TotalTime in [10000]: #3000 total drift time
    xlist=[]
    ylist=[]
    fig = plt.figure(fignum)
    fignum+=1
    ax = fig.add_subplot(1,1,1)
    for NumOfPhoton in range(1,150):  
        xlist.append(NumOfPhoton)
        ylist.append(ddict[TotalTime][NumOfPhoton]['t1090']/TotalTime)
    ax.boxplot(ylist, 0,'',positions=xlist)
    ax.set_xlabel('Num of Phtons')
    ax.set_ylabel('t1090/t_total')
    ax.grid('on')
    ax.set_xlim(0,150)
    ax.set_ylim(0.5,0.95)
    plt.savefig(savedir+'PhotonSim_TotalTime%d_T1090_MedianQ1Q3.png'%(TotalTime))  
    
     
plt.close('all')
for TotalTime in [10000]: #3000 total drift time
    fig = plt.figure(fignum)
    fignum+=1
    ax = fig.add_subplot(1,1,1)
    xlist=[]
    ylist=[]
    for NumOfPhoton in range(1,150):  
        xlist.append(NumOfPhoton)
        sel = (np.array(ddict[TotalTime][NumOfPhoton]['t1090']) > TotalTime*0.70) & (np.array(ddict[TotalTime][NumOfPhoton]['t1090']) < TotalTime*0.90)
        ylist.append(float(np.sum(sel))/float(len(sel)))        
    plt.plot(xlist,ylist, label=r't1090/t_total $\in$ (0.7, 0.9)')
    a = np.array([xlist, ylist])
    np.save(savedir+'PhotonNum_T1090_79_SelEff',a)  
    xlist=[]
    ylist=[]
    for NumOfPhoton in range(1,150):  
        xlist.append(NumOfPhoton)
        sel = (np.array(ddict[TotalTime][NumOfPhoton]['t1090']) > TotalTime*0.60) & (np.array(ddict[TotalTime][NumOfPhoton]['t1090']) < TotalTime*0.90)
        ylist.append(float(np.sum(sel))/float(len(sel)))        
    plt.plot(xlist,ylist, label=r't1090/t_total $\in$ (0.6, 0.9)')
    a = np.array([xlist, ylist])
    np.save(savedir+'PhotonNum_T1090_69_SelEff',a)  
    ax.set_xlabel('Num of Phtons')
    ax.set_ylabel('t1090/t_total')
    ax.grid('on')
    ax.set_xlim(0,150)
    ax.set_ylim(0.5,1.05)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height*0.6])
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1, mode="expand", borderaxespad=0.)
    plt.savefig(savedir+'PhotonSim_TotalTime%d_T1090_selectionEff.png'%(TotalTime))    
    
a = np.array([xlist, ylist,])
np.save(savedir+'PhotonNum_T1090_79_SelEff',a)   



 
plt.close('all')
for TotalTime in [10000]: #3000 total drift time
    xlist=[]
    ylist=[]
    zlist=[]
    fig = plt.figure(fignum)
    fignum+=1
    ax = fig.add_subplot(1,1,1)
    for NumOfPhoton in range(10,1500,10):  
        xlist.append(NumOfPhoton)
        ylist.append(np.mean(ddict[TotalTime][NumOfPhoton]['t1090']/ddict[TotalTime][NumOfPhoton]['t2575']))
        zlist.append(np.std(ddict[TotalTime][NumOfPhoton]['t1090']/ddict[TotalTime][NumOfPhoton]['t2575']))
    ax.errorbar(xlist, ylist, yerr=zlist, fmt='o',color='b')
    xlist=[]
    ylist=[]
    zlist=[]
    for NumOfPhoton in range(10,1500,10):  
        xlist.append(NumOfPhoton)
        ylist.append(np.mean(ddict_gau[TotalTime][NumOfPhoton]['t1090']/ddict_gau[TotalTime][NumOfPhoton]['t2575']))
        zlist.append(np.std(ddict_gau[TotalTime][NumOfPhoton]['t1090']/ddict_gau[TotalTime][NumOfPhoton]['t2575']))
    ax.errorbar(xlist, ylist, yerr=zlist, fmt='o',color='r')
    ax.set_xlabel('Num of Phtons')
    ax.set_ylabel('t1090/t2575')
    ax.grid('on')
    ax.set_xlim(0,150)
    ax.set_ylim(1.2,2.0)
    plt.savefig(savedir+'PhotonSim_TotalTime%d_T1090overT2575_MeanStd.png'%(TotalTime))
     

   
plt.close('all')
for TotalTime in [10000]: #3000 total drift time
    xlist=[]
    ylist=[]
    fig = plt.figure(fignum)
    fignum+=1
    ax = fig.add_subplot(1,1,1)
    for NumOfPhoton in range(1,150):  
        xlist.append(NumOfPhoton)
        ylist.append(ddict[TotalTime][NumOfPhoton]['t1090']/ddict[TotalTime][NumOfPhoton]['t2575'])
    ax.boxplot(ylist, 0,'',positions=xlist)
    for NumOfPhoton in range(1,150):  
        xlist.append(NumOfPhoton)
        ylist.append(ddict_gau[TotalTime][NumOfPhoton]['t1090']/ddict_gau[TotalTime][NumOfPhoton]['t2575'])
    ax.boxplot(ylist, 0,'',positions=xlist)
    ax.set_xlabel('Num of Phtons')
    ax.set_ylabel('t1090/t2575')
    ax.grid('on')
    ax.set_xlim(0,150)
    ax.set_ylim(0.6,3.0)
'''

save_obj(ddict, savedir+"/simData")
ddict = load_obj(savedir+"/simData")

plt.close('all')

fignum=100
 #3000 total drift time
def Plot0123Sigma(x_list, data_list, ylabel='',ylim=[0.5,0.8], xlabel='# detected photons', drawhatch=True, sigma0=True, sigma1=True, sigma2=True, sigma3=True, fignum=fignum, savename='test.png', savearr='test.txt'):
    plt.close('all')
    ss = [-4,-3,-2,-1,0,1,2,3,4]
    y_list=[]
    for ll in range(len(ss)):
        y_list.append([])
    fig = plt.figure(fignum)
    ax = fig.add_subplot(1,1,1)
    for kk in range(len(x_list)):
        data = data_list[kk]        
        for ll in range(len(ss)):
            cdf=stats.norm.cdf(ss[ll])
            y_list[ll].append(np.percentile(data, cdf*100.))
    if sigma3:
        ax.plot(x_list, y_list[0], color='r', lw=2, linestyle='-.')
#        ax.plot(x_list, y_list[8], color='r', lw=2, linestyle='-.', label=r'$5 \sigma$')
        ax.fill_between(x_list, y_list[0], y_list[8], color='r',  label=r'$4 \sigma$')
        ww=0
    if sigma2:
#        ax.plot(x_list, y_list[1], color='b', lw=2, linestyle='-.')
#        ax.plot(x_list, y_list[7], color='b', lw=2, linestyle='-.', label=r'$3 \sigma$')
#        ax.fill_between(x_list, y_list[1], y_list[7], color='b',  label=r'$3 \sigma$')
        ww=0
    if sigma1:
        ax.plot(x_list, y_list[2], color='lawngreen', lw=2, linestyle='-.')
#        ax.plot(x_list, y_list[6], color='yellow', lw=2, linestyle='-.', label=r'$1 \sigma$')
        ax.fill_between(x_list, y_list[2], y_list[6], color='lawngreen',  label=r'$2 \sigma$')
    if sigma0:
        ax.plot(x_list, y_list[3], color='yellow', lw=2, linestyle='-.')
#        ax.plot(x_list, y_list[5], color='yellow', lw=2, linestyle='-.', label=r'$1 \sigma$')
        ax.fill_between(x_list, y_list[3], y_list[5], color='yellow',  label=r'$1 \sigma$')   
    if drawhatch:
#        y_listup=np.ones_like(   y_list[0])*ylim[1]
#        y_listdown=np.ones_like(   y_list[0])*ylim[0]
#        ax.fill_between(x_list, y_list[-1], y_listup, facecolor="b",color="r",hatch="+",edgecolor="m",  label=r'long')
#        ax.fill_between(x_list, y_listdown, y_list[0], facecolor="b",color="r",hatch="+",edgecolor="c",  label=r'short')
        y_listuppoly=[[1e4,1],[0,1],]
        y_listdownpoly=[[1e4,0],[0,0],]
        for qq in range(len(x_list)):
            y_listuppoly.append([x_list[qq], y_list[8][qq]])
            y_listdownpoly.append([x_list[qq], y_list[0][qq]])
        mpl.rcParams['hatch.color'] = 'm'
#        ax.add_patch(Polygon(y_listuppoly, hatch='+',  edgecolor='m',lw=0, fill=False))
        mpl.rcParams['hatch.color'] = 'b'
#        ax.add_patch(Polygon(y_listdownpoly, hatch='+', color='m', lw=0, fill=False))
    ax.plot(x_list, y_list[4], color='k', lw=3, linestyle='-', label='Median')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([x_list[0], x_list[-1]])
    ax.set_ylim(ylim)
    ax.set_xscale('log')
    ax.grid('on')
    box=ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height *.7])
    ax.legend(loc=3, bbox_to_anchor=(0., 1.15, 1., .102), ncol=2, mode="expand", borderaxespad=0.,fontsize=24)
    plt.savefig(savename)
    y_list.insert(0,x_list)
    np.savetxt(savearr,np.array(y_list))    
    #plt.close('all')

fignum=100


for TotalTime in [10000]:
    fignum+=1
    x_list=np.sort(list(ddict[TotalTime].keys()))
    data_list=[]
    for xx in x_list:
        data_list.append((ddict[TotalTime][xx]['t90']-ddict[TotalTime][xx]['t10'])/TotalTime)
#    Plot0123Sigma(x_list[1:-1], data_list[1:-1], xlabel='# detected photons', ylabel=r'$t_{10-90}/t_{drfit}$',ylim=[-0.1,1.1], savename=savedir+'t1090.png',savearr=savedir+'t1090.txt', fignum=fignum)
    Plot0123Sigma(x_list[:], data_list[:], xlabel='# detected photons', ylabel=r'$t_{10-90}/t_{drfit}$',ylim=[-0.1,1.1], savename=savedir+'t1090.png',savearr=savedir+'t1090.txt', fignum=fignum)



for TotalTime in [10000]:
    fignum+=1
    x_list=np.sort(list(ddict[TotalTime].keys()))
    data_list=[]
    for xx in x_list:
#        data_list.append((ddict[TotalTime][xx]['t90']-ddict[TotalTime][xx]['t10'])/TotalTime)
        data_list.append((ddict[TotalTime][xx]['t50'])/(ddict[TotalTime][xx]['t10'] + ddict[TotalTime][xx]['t90']))
    Plot0123Sigma(x_list[1:-1], data_list[1:-1],  xlabel='# detected photons', ylabel=r'$t50/(t10+t90)$',ylim=[0.1,0.9], savename=savedir+'t50dT1090.png',savearr=savedir+'t50dT1090.txt', fignum=fignum)


for TotalTime in [10000]:
    fignum+=1
    x_list=np.sort(list(ddict[TotalTime].keys()))
    data_list=[]
    for xx in x_list:
        data_list.append((ddict[TotalTime][xx]['t90']-ddict[TotalTime][xx]['t10'])/(ddict[TotalTime][xx]['t75']-ddict[TotalTime][xx]['t25']))
    Plot0123Sigma(x_list[1:-1], data_list[1:-1], ylabel=r'$t_{10-90}/t2575$',ylim=[1.,4.0], savename=savedir+'t1090dt2575.png',savearr=savedir+'t1090dt2575.txt', fignum=fignum)

for TotalTime in [10000]:
    fignum+=1
    x_list=np.sort(list(ddict[TotalTime].keys()))
    data_list=[]
    for xx in x_list:
        data_list.append((ddict[TotalTime][xx]['t95']-ddict[TotalTime][xx]['t85'])/(ddict[TotalTime][xx]['t15']-ddict[TotalTime][xx]['t05']))
    Plot0123Sigma(x_list[1:-1], data_list[1:-1], ylabel=r'$t8595/t0515$',ylim=[0.,10.0], savename=savedir+'t8595dt0515.png',savearr=savedir+'t8595dt0515.txt', fignum=fignum)

for TotalTime in [10000]:
    for key in ['t01','t05', 't10','t15','t25', 't50', 't75', 't85', 't90','t95', 't99']:
        fignum+=1
        x_list=np.sort(list(ddict[TotalTime].keys()))
        data_list=[]
        for xx in x_list:
            data_list.append((ddict[TotalTime][xx][key])/TotalTime)
        Plot0123Sigma(x_list[1:-1], data_list[1:-1], ylabel=key+r'$/t_{drfit}$',ylim=[0.,1.0], savename=savedir+key+'.png',savearr=savedir+key+'.txt', fignum=fignum)

import scipy
y_list=[]
for TotalTime in [10000]:
    for key in ['t01','t05', 't10','t15','t25', 't50', 't75', 't85', 't90','t95', 't99']:
        y_list.append((ddict[TotalTime][xx][key]))

t_list= [1,5,10,15,25, 50, 75,85,90,95,99]
r_list=[]
p_list=[]    
for TotalTime in [10000]: 
    x_list=np.sort(list(ddict[TotalTime].keys()))
    data_list=[] 
    for xx in x_list:
        r_list=[]
        p_list=[]
        y_list=[]
        for key in ['t01','t05', 't10','t15','t25', 't50', 't75', 't85', 't90','t95', 't99']:
            y_list.append((ddict[TotalTime][xx][key]))
        for nn in range(len(y_list[0])):
            slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(t_list), np.array(y_list)[:,nn])
            r_list.append(r_value)
            p_list.append(p_value)
        data_list.append(r_list)
        mm=len(data_list)
        Plot0123Sigma(x_list[:mm], data_list[:mm], ylabel='r',ylim=[0.8,1.0], savename=savedir+'r.png',savearr=savedir+'r.txt', fignum=fignum)

########################
t_list= [1,5,10,15,25, 50, 75,85,90,95,99]

for nn in range(3, len(t_list)):    
    data_list=[] 
    for jj in x_list:
        l_list=[]
        for rr in range(nn):
            tt=t_list[rr]
            rq = 't%02d'%(tt)
            l_list.append(ddict[TotalTime][jj][rq])
        l_arr=np.array(l_list)
        l_av = l_arr.mean(axis=0)
        l_av_arr = np.array([l_av]).T.repeat(nn,axis=1).T
        t_arr= np.array([t_list[:nn]]).T.repeat(l_arr.shape[1],axis=1)
        t_av_arr = np.full_like(l_arr,np.array([t_list[:nn]]).mean())
        r2 = (((l_arr- l_av_arr)*(t_arr-t_av_arr)).sum(axis=0))**2/(((l_arr-l_av_arr)**2).sum(axis=0) * ((t_arr-t_av_arr)**2).sum(axis=0))
        r=r2**(0.5)
        data_list.append(r)        
    mm=len(data_list)
    Plot0123Sigma(x_list[:mm], data_list[:mm], ylabel='r',ylim=[0.8,1.0], savename=savedir+str(nn)+'r.png',savearr=savedir+str(nn)+'r.txt', fignum=fignum)


    
    sel = (ddicts[jj]['coin_pulse_areas_sum']<100) & (ddicts[jj]['coin_pulse_areas_t1090']>500)
    plt.hist(r[sel], range=[0,1],bins=100)
    plt.yscale('log')

for nn in range(10):
    plt.plot(t_list, l_arr[:,nn], label=str(nn)+': %.3f'%(r[nn]))

plt.legend()




