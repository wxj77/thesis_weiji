# -*- coding: utf-8 -*-
"""
Created on Thu May 17 12:58:19 2018

@author: wxj
"""

import os, sys, re, glob
sys.path.insert(0, "/home/wxj/.2go")
sys.path.insert(0, "/home/wxj/.x2go")
sys.path.insert(0, "/home/wxj/.2go/PythonScripts/RunSetups")
sys.path.insert(0, "/home/wxj/.2go/PythonScripts")
sys.path.insert(0, "/home/wxj/gtest_result/")
sys.path.insert(0, './')

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
import matplotlib 


import os, sys
wd='/Users/weiji/Google Drive/gastest/' #'/home/wxj/gastest/'
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
savedir =workdir
sys.path.insert(0, workdir)


from color import *
from plotStyle import *


import EFieldFactorFromGarFieldSim as efg



lightcollection_tot=1.7e-2
lightcollection_top=.85e-2
lightcollection_bot=.85e-2

def weight_array(ar, weights):
    zipped = zip(ar, weights)
    weighted=[]
    for i in zipped:
        for j in range(int(i[1])):
            weighted.append(i[0])
    return weighted




#ff=1.0 is gate wire surface field operate at dV=8
#pressure might be 3.3 bara, gas density 0.137 mol/l, factor=1.0 is operation voltage dv=10kV, anode wire (plane woven, pitch 2.5mm diameter 100 um), gate wire (plane woven, pitch 5mm diameter 75 um), distance 13mm; and this is calucalating ionization around gate wire. 
for ff in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]:
#    execstr = "ion"+ "%05d"%(ff*1000)+"=np.loadtxt('"+workdir+"xenon3300mbar/Factor_"+"%.3f"%(int(np.abs(ff*10)))+"_Radius_37.5_ion.txt', skiprows=2)"
    #max electric field
    execstr = "ion"+ "%05d"%(ff*1000)+"=np.loadtxt('"+workdir+"xenon3300mbar/result_"+"%.3f"%(efg.EFieldMaxFactor[int(np.abs(ff*10))])+"_ion.txt', skiprows=2)"
    exec(execstr)
    # average electric field
    execstr = "ionA"+ "%05d"%(ff*1000)+"=np.loadtxt('"+workdir+"xenon3300mbar/result_"+"%.3f"%(efg.EFieldAveFactor[int(np.abs(ff*10))])+"_ion.txt', skiprows=2)"
    exec(execstr)

baseSurfaceE = 1000/(np.log(1000)-np.log(36))/37.5/0.1 # iff (factor) is 1 the surface electric  field in kV/cm (81.2kV/cm)
NA = 6.022e23 #Avoga
n7=0.1370e3; label7='3.3 bara, 290 K'
baseDensityN = NA*n7#convert kV/cm to Td at 295 K 3.3 bara
baseRedSurE = baseSurfaceE*1.e5/baseDensityN*1.e21

fignum=100
fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)

#for ff in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]:
for ff in [ 0.6, 0.8,1.0, 1.2, 1.4, 1.6,]:
#    execstr = "ax.step("+"ion"+ "%05d"%(ff*1000)+"[:,0], "+"ion"+ "%05d"%(ff*1000)+"[:,1], "+" color=color_list["+str(np.absolute(int(ff*10))) +"],label='dV: %d kV"%(ff*10)+"', where='mid' "+")"
    execstr = "ax.plot("+"ion"+ "%05d"%(ff*1000)+"[:,0], "+"ion"+ "%05d"%(ff*1000)+"[:,1], "+"c=color_list["+str(int(np.absolute(ff*10))) +"],label='%.1f kV cm"%(efg.EFieldMaxFactor[int(np.abs(ff*10))]*baseSurfaceE)+r"$^{-1}$"+"')"
    exec(execstr)    

text="3.3 bara\n295 K"
ax.annotate(text, xy=(0.7, 0.6), xycoords='axes fraction', fontsize=24,
    bbox=dict(facecolor='white', alpha=0.8),
    horizontalalignment='left', verticalalignment='bottom')

plt.xlabel('cathodic gas gain [electron'+r'$^{-1}$'+']')
plt.ylabel(r'probability')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height*0.7])
ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,fontsize=18
           )
          
ax.set_xlim(0,25)
#ax.set_ylim(0,.25)
ax.set_ylim(5.e-4, 1.)
ax.set_yscale('log')
plt.grid('on')
plt.savefig(savedir+"PhotonMultiplicationNaive.png")

fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)

#for ff in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]:
for ff in [ 0.6, 0.8,1.0, 1.2, 1.4, 1.6,]:
#    execstr = "ax.step("+"ion"+ "%05d"%(ff*1000)+"[:,0], "+"ion"+ "%05d"%(ff*1000)+"[:,1], "+" color=color_list["+str(np.absolute(int(ff*10))) +"],label='dV: %d kV"%(ff*10)+"', where='mid' "+")"
    execstr = "ax.plot("+"ion"+ "%05d"%(ff*1000)+"[:,0], "+"ion"+ "%05d"%(ff*1000)+"[:,1], "+"c=color_list["+str(int(np.absolute(ff*10))) +"],label='%.1f"%(efg.EFieldMaxFactor[int(np.abs(ff*10))]*baseRedSurE)+" [Td]"+"')"
    exec(execstr)    

text="3.3 bara\n295 K"
ax.annotate(text, xy=(0.7, 0.6), xycoords='axes fraction', fontsize=24,
    bbox=dict(facecolor='white', alpha=0.8),
    horizontalalignment='left', verticalalignment='bottom')

plt.xlabel('cathodic gas gain [electron'+r'$^{-1}$'+']')
plt.ylabel(r'probability')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height*0.7])
ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,fontsize=18
           )
          
ax.set_xlim(0,25)
#ax.set_ylim(0,.25)
ax.set_ylim(5.e-4, 1.)
ax.set_yscale('log')
plt.grid('on')
plt.savefig(savedir+"PhotonMultiplicationNaiveReduced.png")
#########End naive ionization creation.

##plot naive #electron median sigma
NumOfSims=1000000
E_2=[]
ymeanlist_2=[]
ymedlist_2=[]
yminlist_2=[]
ymaxlist_2=[] 
ymin2list_2=[]
ymax2list_2=[] 
EA_2=[]
yAmeanlist_2=[]
yAmedlist_2=[]
yAminlist_2=[]
yAmaxlist_2=[] 
yAmin2list_2=[]
yAmax2list_2=[] 

for ff in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]:
#for ff in [ 0.6, 0.8,1.0, 1.2, 1.4, 1.6,]:
#for ff in [1.4,] :
    xxx=0;    
    yyy=np.array([])
    zzz=np.array([])
    execstr = "yyy =weight_array((ion"+ "%05d"%(ff*1000)+"[:,0]),(ion"+ "%05d"%(ff*1000)+"[:,1] * NumOfSims) )"
    exec(execstr)
    execstr = "zzz =weight_array((ionA"+ "%05d"%(ff*1000)+"[:,0]),(ionA"+ "%05d"%(ff*1000)+"[:,1] * NumOfSims) )"
    exec(execstr)      
    pmfmed = np.percentile(yyy, 50)
    pmfmin = np.percentile(yyy, 50-34.1)
    pmfmax = np.percentile(yyy, 50+34.1)    
    pmfAmed = np.percentile(zzz, 50)
    pmfAmin = np.percentile(zzz, 50-34.1)
    pmfAmax = np.percentile(zzz, 50+34.1)
    E_2.append(np.absolute(  efg.EFieldMaxFactor[int(np.abs(ff*10))]) *baseSurfaceE)
    EA_2.append(np.absolute(  efg.EFieldAveFactor[int(np.abs(ff*10))]) *baseSurfaceE)
    ymeanlist_2.append(np.mean(yyy))
    ymedlist_2.append(pmfmed)
    yminlist_2.append(pmfmin)
    ymaxlist_2.append(pmfmax)
    ymin2list_2.append(np.percentile(yyy, 50-47.7))
    ymax2list_2.append(np.percentile(yyy, 50+47.7))
    yAmeanlist_2.append(np.mean(zzz))
    yAmedlist_2.append(pmfAmed)
    yAminlist_2.append(pmfAmin)
    yAmaxlist_2.append(pmfAmax)
    yAmin2list_2.append(np.percentile(zzz, 50-47.7))
    yAmax2list_2.append(np.percentile(zzz, 50+47.7))



E = np.append(E_2, EA_2)
lmean = np.append(ymeanlist_2, yAmeanlist_2)
lmed = np.append(ymedlist_2, yAmedlist_2)
lmin = np.append(yminlist_2, yAminlist_2)
lmax = np.append(ymaxlist_2, yAmaxlist_2)
lmin2 = np.append(ymin2list_2, yAmin2list_2)
lmax2 = np.append(ymax2list_2, yAmax2list_2)


argE= np.argsort(E)
sE = E[argE]
sRE = sE*1.e5/baseDensityN*1.e21
slmean = lmean[argE]
slmed = lmed[argE]
slmin = lmin[argE]
slmax = lmax[argE]
slmin2 = lmin2[argE]
slmax2 = lmax2[argE]

fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)
ax.plot(sE, slmean, ls='--', label='avg.', color=color_list[0])
ax.plot(sE, slmed, ls='-', label='med.', color=color_list[0])
ax.plot(sE, slmin, ls='-.', color=color_list[0], label='15.9%-84.1%')#label='1 '+r'$\sigma$'
ax.plot(sE, slmax, ls='-.', color=color_list[0])
ax.plot(sE, slmin2, ls=':', color=color_list[0], label='2.3%-97.7%')#label='1 '+r'$\sigma$'
ax.plot(sE, slmax2, ls=':', color=color_list[0])
text="3.3 bara\n295 K"
ax.annotate(text, xy=(0.1, 0.6), xycoords='axes fraction', fontsize=24,
    bbox=dict(facecolor='white', alpha=0.8),
    horizontalalignment='left', verticalalignment='bottom')

plt.xlabel('surface field [kV cm'+r'$^{-1}$'+']')
plt.ylabel('cathodic gas gain [electron'+r'$^{-1}$'+']')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height*0.7])
ax.yaxis.set_label_coords(-0.15,.6)
ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
#plt.yscale('log')
plt.grid('on')
plt.savefig(savedir+"MultiplicationMeanSigma3300mbar.png")
##########################

fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)
ax.plot(sRE, slmean, ls='--', label='avg.', color=color_list[0])
ax.plot(sRE, slmed, ls='-', label='med.', color=color_list[0])
ax.plot(sRE, slmin, ls='-.', color=color_list[0], label='15.9%-84.1%')#label='1 '+r'$\sigma$'
ax.plot(sRE, slmax, ls='-.', color=color_list[0])
ax.plot(sRE, slmin2, ls=':', color=color_list[0], label='2.3%-97.7%')#label='1 '+r'$\sigma$'
ax.plot(sRE, slmax2, ls=':', color=color_list[0])
text="3.3 bara\n295 K"
ax.annotate(text, xy=(0.1, 0.6), xycoords='axes fraction', fontsize=24,
    bbox=dict(facecolor='white', alpha=0.8),
    horizontalalignment='left', verticalalignment='bottom')

plt.xlabel('reduced surface field [Td]')
plt.ylabel('cathodic gas gain [electron'+r'$^{-1}$'+']')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height*0.7])
ax.yaxis.set_label_coords(-0.15,.6)
ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
#plt.yscale('log')
plt.grid('on')
plt.savefig(savedir+"MultiplicationMeanSigma3300mbarReduced.png")

########################






ma= 0.137 #ph/e/V
mb= -4.7e-18 #ph/e *cm^2/atom
mb=mb*1.e-4

NA = 6.022e23 #Avoga
n1=0.020795e3; label1='0.5 bara, 295 K'
n2=0.041709e3; label2='1.0 bara, 295 K'
n3=0.062744e3; label3='1.5 bara, 295 K'
n4=0.083901e3; label4='2.0 bara, 295 K'
n5=0.10518e3; label5='2.5 bara, 295 K'
n6=0.12659e3; label6='3.0 bara, 295 K'
n7=0.13950e3; label7='3.3 bara, 295 K'
n8=0.14812e3; label8='3.5 bara, 295 K'

n_list=[n1,n2,n3,n4,n5,n6,n7,n8]
label_list=[label1,label2,label3,label4,label5,label6,label7,label8]

dv_list= np.arange(0,17,1)
e_list=[]
for dv in dv_list:
    e_list.append(dv*1.e3/13e-3)

fignum=1000
fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)

#for jj in range(len(n_list)):
for jj in [6]: # 6 is 3.3 bara
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








  
NumOfSims=100000
MaxX=400

fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)
yyy=np.array([])
xxx=0;    
ymedlist_2=[]
yminlist_2=[]
ymaxlist_2=[] 

yAmedlist_2=[]
yAminlist_2=[]
yAmaxlist_2=[] 


#for ff in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]:
for ff in [ 0.6, 0.8,1.0, 1.2, 1.4, 1.6,]:
#for ff in [1.4,] :
    xxx=0;    
    yyy=np.array([])
    for aa in range(1,100):
        execstr = "xxx =binom.rvs(int(int(ion"+ "%05d"%(ff*1000)+"[aa,0]) * ylist["+"%d"%(ff*10)+"]),lightcollection_tot, size= int(ion"+ "%05d"%(ff*1000)+"[aa,1] * NumOfSims) )"
        exec(execstr)    
        if (len(xxx)>0):    
 #           print xxx
            yyy = np.append(yyy, xxx)
    pmfmed = np.percentile(yyy, 50)
    pmfmin = np.percentile(yyy, 50-34)
    pmfmax = np.percentile(yyy, 50+34)
    bins=np.linspace(0,MaxX, MaxX+1)
    value, bins =np.histogram(yyy, bins=bins, normed=True)
    ax.plot(bins[:-1], value, label="dV: %d kV "%(ff*10) , color=color_list[int(np.abs(ff*10))])


text="3.3 bara"
ax.annotate(text, xy=(0.8, 0.8), xycoords='axes fraction', fontsize=10,
    bbox=dict(facecolor='white', alpha=0.8),
    horizontalalignment='left', verticalalignment='bottom')


plt.ylabel(r'Probability [phe$^{-1}$]')
plt.xlabel('# Photon collected [phe/e]')
box = ax.get_position()
#ax.set_ylim(0,0.04)
ax.set_ylim(5.e-4, .2)
ax.set_yscale('log')
ax.set_position([box.x0, box.y0, box.width, box.height*0.7])
ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.grid('on')
plt.savefig(savedir+"MultiplicationPhotonColletionNaiveProfile3300mbar1.png")




fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)
xlist_2=[]
ymedlist_2=[]
yminlist_2=[]
ymaxlist_2=[] 
yAmedlist_2=[]
yAminlist_2=[]
yAmaxlist_2=[] 
ypmed=[]
ypmin=[]
ypmax=[]
for ff in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]:
#for ff in [ 0.6, 0.8,1.0, 1.2, 1.4, 1.6,]:
#for ff in [1.4,] :
    xxx=0;    
    yyy=np.array([])
    for aa in range(1,100):
        execstr = "xxx =binom.rvs(int(int(ion"+ "%05d"%(ff*1000)+"[aa,0]) * ylist["+"%d"%(ff*10)+"]),lightcollection_tot, size= int(ion"+ "%05d"%(ff*1000)+"[aa,1] * NumOfSims) )"
        exec(execstr)    
        if (len(xxx)>0):    
#            print xxx
            yyy = np.append(yyy, xxx)
    xxx=0;
    zzz=np.array([])
    for aa in range(1,100):
        execstr = "xxx =binom.rvs(int(int(ionA"+ "%05d"%(ff*1000)+"[aa,0]) * ylist["+"%d"%(ff*10)+"]),lightcollection_tot, size= int(ionA"+ "%05d"%(ff*1000)+"[aa,1] * NumOfSims) )"
        exec(execstr)    
        if (len(xxx)>0):    
#            print xxx
            zzz = np.append(zzz, xxx)
    pmfmed = np.percentile(yyy, 50)
    pmfmin = np.percentile(yyy, 50-34.1)
    pmfmax = np.percentile(yyy, 50+34.1)
    pmfAmed = np.percentile(zzz, 50)
    pmfAmin = np.percentile(zzz, 50-34.1)
    pmfAmax = np.percentile(zzz, 50+34.1)
    xlist_2.append(np.absolute(  ff*10) )
    ymedlist_2.append(pmfmed)
    yminlist_2.append(pmfmin)
    ymaxlist_2.append(pmfmax)
    yAmedlist_2.append(pmfAmed)
    yAminlist_2.append(pmfAmin)
    yAmaxlist_2.append(pmfAmax)
    ypmed.append(binom.ppf(0.5,ylist[int(ff*10)],lightcollection_tot,0))
    ypmin.append(binom.ppf(0.159,ylist[int(ff*10)],lightcollection_tot,0))
    ypmax.append(binom.ppf(0.841,ylist[int(ff*10)],lightcollection_tot,0))

ax.plot(xlist_2, ymedlist_2,ls='-', label='max. wire surface field', color=color_list[0])
ax.plot(xlist_2, yAmedlist_2,ls='-', label='avg. wire surface filed', color=color_list[1])
ax.plot(xlist_2, yminlist_2,ls='-.', color=color_list[0])
ax.plot(xlist_2, ymaxlist_2,ls='-.', color=color_list[0])
ax.plot(xlist_2, yAminlist_2,ls='-.', color=color_list[1])
ax.plot(xlist_2, yAmaxlist_2,ls='-.', color=color_list[1])

ax.plot(xlist_2, ypmed,ls='-', label='per drift electron', color=color_list[6])
ax.plot(xlist_2, ypmin,ls='-.', color=color_list[6])
ax.plot(xlist_2, ypmax,ls='-.', color=color_list[6])

text="3.3 bara\n295 K"
ax.annotate(text, xy=(0.1, 0.6), xycoords='axes fraction', fontsize=24,
    bbox=dict(facecolor='white', alpha=0.8),
    horizontalalignment='left', verticalalignment='bottom')

plt.xlabel('operation voltage dV [kV]')
plt.ylabel('signal area [phe electron'+r'$^{-1}$'+']')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height*0.7])
ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1, mode="expand", borderaxespad=0.)
plt.yscale('log')
plt.grid('on')
plt.savefig(savedir+"MultiplicationPhotonCollectionNaiveProfileMeanSigma3300mbar.png")
#########End naive ionization creation.


###per drift electron

    
fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)

for jj in [1,3,6]: # 6 is 3.3 bara
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
    xlist_2=[]
    ypmed=[]
    ypmin=[]
    ypmax=[]
    for ff in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]:
        xlist_2.append(np.absolute(  ff*10) )
        ypmed.append(binom.ppf(0.5,ylist[int(ff*10)],lightcollection_tot,0))
        ypmin.append(binom.ppf(0.159,ylist[int(ff*10)],lightcollection_tot,0))
        ypmax.append(binom.ppf(0.841,ylist[int(ff*10)],lightcollection_tot,0))    
    ax.plot(xlist_2, ypmed,ls='-', label=lab, color=color_list[jj])
    ax.plot(xlist_2, ypmin,ls='-.', color=color_list[jj])
    ax.plot(xlist_2, ypmax,ls='-.', color=color_list[jj])

plt.xlabel('operation voltage dV [kV]')
plt.ylabel('signal area [phe electron'+r'$^{-1}$'+']')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height*0.7])
ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1, mode="expand", borderaxespad=0.)
#plt.yscale('log')
plt.grid('on')
plt.savefig(savedir+"MultiplicationPhotonCollectionNaiveProfileMeanSigma3300mbarPerDriftElecton.png")
#########End naive ionization creation.
