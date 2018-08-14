drawlinear=1

import os, sys

sys.path.insert(0, "/home/wxj/.2go")
sys.path.insert(0, "/home/wxj/.x2go")
sys.path.insert(0, "/home/wxj/.2go/PythonScripts/RunSetups")
sys.path.insert(0, "/home/wxj/gastest/obsolete/PythonScripts")
sys.path.insert(0, "/home/wxj/gtest_result/")

import numpy as np
from DriftVoltage import *
import matplotlib.pyplot as plt

workdir='/home/wxj/gastest/xenonProperties/'


from color import *
from plotStyle import *



'''
#https://matplotlib.org/1.5.1/users/customizing.html
label_size = 18
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.size'] = 15
mpl.rcParams['legend.fontsize'] = 15
mpl.rcParams['figure.autolayout'] = False  # When True, automatically adjust subplot
                            # parameters to make the plot fit the figure

mpl.rcParams['figure.subplot.left']  = 0.18  # the left side of the subplots of the figure
mpl.rcParams['figure.subplot.right']  = 0.85    # the right side of the subplots of the figure
mpl.rcParams['figure.subplot.bottom']  = 0.15   # the bottom of the subplots of the figure
mpl.rcParams['figure.subplot.top']  = 0.85     # the top of the subplots of the figure
mpl.rcParams['figure.subplot.wspace']  = 0.2    # the amount of width reserved for blank space between subplots
mpl.rcParams['figure.subplot.hspace']  = 0.2    # the amount of height reserved for white space between subplots
'''

NA = 6.022e23 #Avoga
n1=0.020795e3; label1='0.5 bara, 290 K'
n2=0.041709e3; label2='1.0 bara, 290 K'
n3=0.062744e3; label3='1.5 bara, 290 K'
n4=0.083901e3; label4='2.0 bara, 290 K'
n5=0.10518e3; label5='2.5 bara, 290 K'
n6=0.12659e3; label6='3.0 bara, 290 K'
n7=0.14812e3; label7='3.5 bara, 290 K'
n8=0.13950e3; label8='3.3 bara, 290 K'


n_list=[n1,n2,n3,n4,n5,n6,n7,n8]
label_list=[label1,label2,label3,label4,label5,label6,label7,label8]

'''
color_list=[
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
'''
Filenames=[
'cal_0500mbara',
'cal_1000mbara',
'cal_1500mbara',
'cal_2000mbara',
'cal_2500mbara',
'cal_3000mbara',
'cal_3500mbara',
]

Td = 1.e-21

fignum=100
fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)

for jj in range(len(Filenames)):
    xlist, ylist, zlist, yylist, yyq1list, yyq3list=np.loadtxt('/home/wxj/gtest_result/Run9WithKaptonCover/'+Filenames[jj]+'/drift_time_cal_030040.txt') 
    px=[]
    py=[]
    perry=[]
    for kk in range(len(xlist)):
        vv = np.abs(xlist[kk])
        px.append(np.abs(drift_voltage_list[int(vv)]/(np.array(n_list[jj])*NA))/Td)
#        px.append(np.abs(drift_voltage_naive_list[int(vv)]/(np.array(n_list[jj])*NA))/Td)
        yy = ylist[kk]/.78
        py.append(13/yy*1.e3)
        erry= np.sqrt(zlist[kk]**2-(yy*.07)**2)
        perry.append(13*erry/yy**2*1.e3)
    ax.errorbar(px, py, yerr=perry, fmt='o', label=label_list[jj], color= color_list[2*jj+2], markersize='4', capsize=3)

yy_3= np.loadtxt(workdir+"/xenonDrift2.txt", skiprows=2); # from H_L_Brooks 1982
EoverN_3 = yy_3[:,0]#Td to Vcm2
driftv_3= yy_3[:, 1]*10 # cm/us to mm/us
plt.plot(EoverN_3, driftv_3, marker='^', color='k', linestyle='None', label="Brooks 1982", markersize='6')

ax.grid('on')
ax.set_xlabel('E/N [Td]')
ax.set_ylabel('Drift velocity(mm/us)')
ax.set_xlim(0,30)
ax.set_ylim(0,15)
if drawlinear:
    xx=np.linspace(0,30)
    yy = 15./27.*xx
    plt.plot(xx, yy, label= "y = %.3f x" %(15./27.), color ="k", ls=':')
    

box = ax.get_position()
ax.set_position([box.x0, box.y0+box.height*0.0, box.width, box.height*0.7])
ax.legend(loc=3, bbox_to_anchor=(0., 1.1, 1., .102), ncol=2, mode="expand", borderaxespad=0.,fontsize=12)

plt.savefig(workdir+'/drift_vel_cal_sum.png')
#plt.close('all')




