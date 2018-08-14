# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
drawlinear=1

import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.interpolate as interp
import scipy.integrate as integrate
import scipy.io as sio
import scipy.optimize as opt

from mpl_toolkits.mplot3d import Axes3D

from color import *
from plotStyle import *

workdir = "/home/wxj/gastest/xenonProperties/"
'''
#https://matplotlib.org/1.5.1/users/customizing.html
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.size'] = 15
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.figsize']=8,6
mpl.rcParams['figure.dpi']=150
mpl.rcParams['savefig.dpi']=100
mpl.rcParams['savefig.jpeg_quality']=95
mpl.rcParams['savefig.format']='png'
'''
mark='o'

xx = np.loadtxt(workdir+"/xenonDrift.txt", skiprows=2); #from English and Hanna
yy_3= np.loadtxt(workdir+"/xenonDrift2.txt", skiprows=2); # from H_L_Brooks 1982
yy_4= np.loadtxt(workdir+"/xenonDrift3.txt", skiprows=2); # from H_L_Brooks 1982 , original from Pack 1962

BarTorr = 750.0616;
dist = 11.05; #cm distance between anode an pmts
dist_ac = 1.3; #cm distance betwwen anode and gate Wei, change to 1.3 cm for grid wires, 0.9 cm for grid rings
AvogadroConstant=6.022140857e23 
lcm3=1.e3
Pressure = 3.3; #3.3 bara
den = 0.137# mol/l
den_cm3 = den / lcm3*AvogadroConstant 
TdVcm2 =1.e-17
EoverP =xx[:,0]*1.e-3*BarTorr#V/cm/Torr to kV/cm/bar
driftv =xx[:,1]#cm/us
Volt= EoverP* Pressure * dist-1.5 # -1.5 PMT cathode voltage
EoverN_3 = TdVcm2* yy_3[:,0]#Td to Vcm2
driftv_3= yy_3[:, 1] # cm/us
EoverN_4 = TdVcm2* yy_4[:,0]#Td to Vcm2
driftv_4= yy_4[:, 1] # cm/us

a_a = 0.25e-2 #  pitch of anode grid (m)
d_a = 100.e-6 #  wire diameter of anode grid (m)
a_c = 0.5e-2 #  pitch of anode grid (m)
d_c = 75.e-6 #  wire diameter of anode grid (m)

def GridFactor(a, d):
    p0=0.5247
    p1= 0.7488
    return p0/(2.*np.pi) *np.log(p0*a/(np.pi*p1*d) )

fig = plt.figure(1)
ax = fig.add_subplot(1,1,1)
plt.plot(Volt, dist/driftv, label="Englist & Hanna")

ax.set_xlabel("operation voltage on anodic grid [kV]")
ax.set_ylabel("maximum drift time [us]")
#ax.set_yscale('log')
#ax.set_xscale('log')
text="11.05 cm distance between \nthe closest PMT and the anodic grid"
ax.annotate(text, xy=(0.4, 0.5), xycoords='axes fraction', fontsize=label_size,
    bbox=dict(facecolor='white', alpha=0.8),
    horizontalalignment='left', verticalalignment='bottom')
ax.set_xlim(-.5,10)
ax.set_ylim(70,125)
#ax.text(0.5,0.8, r"$log_{10}y=(%.3f)+(%.3f)*x$"%(best_vals[0], best_vals[1]), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
ax.legend()

plt.savefig(workdir+"AboveAnodeEventDriftTime.png")


Volt2= EoverP* Pressure * dist_ac # -1.5 PMT cathode voltage
Efield_3 = EoverN_3* den_cm3 * 1.e-3 #v/cm to kv/cm
Volt_3 = Efield_3*dist_ac
Efield_4 = EoverN_4* den_cm3 * 1.e-3 #v/cm to kv/cm
Volt_4 = Efield_4*dist_ac

fig = plt.figure(2)
ax = fig.add_subplot(1,1,1)



plt.plot(Volt_3, dist_ac/driftv_3, marker=mark,
            markerfacecolor='None',
            linestyle = 'None',label="Brooks et al (3.3 bara)")
plt.plot(Volt_3/3.3*2.0, dist_ac/driftv_3, marker=mark,markerfacecolor='None', linestyle = 'None',label="Brooks et al (2.0 bara)")
plt.plot(Volt_3/3.3*1.0, dist_ac/driftv_3, marker=mark,markerfacecolor='None',linestyle = 'None',label="Brooks et al (1.0 bara)")
#plt.plot(Volt_3- Efield_3* 1.e2* (a_a *GridFactor(a_a, d_a)-a_c *GridFactor(a_c, d_c)), dist_ac/driftv_3, label="Brooks et al, cor for wire plane") # cm to m
#plt.plot(Volt_4, dist_ac/driftv_4, marker=mark,markerfacecolor='None',linestyle = 'None',label="Pack el al (3.3 bara)")
#plt.plot(Volt2, dist_ac/driftv, marker=mark,markerfacecolor='None',linestyle = 'None',label="Englist & Hanna (3.3 bara)")
ax.set_xlabel("operation voltage dV [kV]")
ax.set_ylabel("maximum duration [us]")
#ax.set_yscale('log')
#ax.set_xscale('log')
text="%.1f cm distance between \nthe anode grid and the cathodic grid"%(dist_ac)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height*0.7])
ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
ax.annotate(text, xy=(0.4, 0.8), xycoords='axes fraction', fontsize=label_size,
    bbox=dict(facecolor='white', alpha=0.8),
    horizontalalignment='left', verticalalignment='bottom')
ax.set_xlim(0,16)
ax.set_ylim(0,6)
ax.grid('on')
#ax.text(0.5,0.8, r"$log_{10}y=(%.3f)+(%.3f)*x$"%(best_vals[0], best_vals[1]), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
#ax.legend()

plt.savefig(workdir+"BetweenGridsEventDriftTime%02dmm.png"%(dist_ac*10))