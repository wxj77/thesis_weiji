# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 12:52:44 2018

@author: wei
"""

import numpy as np
import matplotlib.pyplot as plt
savedir='/media/wei/ACA8-1ECD1/PythonSimulation/'


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

a= 0.137 #ph/e/V
b= -4.7e-18 #ph/e *cm^2/atom
b=b*1.e-4

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

dv_list= np.arange(0,17,1)
e_list=[]
for dv in dv_list:
    e_list.append(dv*1.e3/13e-3)

fignum=100
fig = plt.figure(fignum)
fignum+=1
ax = fig.add_subplot(1,1,1)
for jj in range(len(n_list)):
    nn=n_list[jj]
    lab = label_list[jj]    
    xlist=[]
    ylist=[]
    for kk in range(len(e_list)):
        dv=dv_list[kk]
        ee=e_list[kk]
        xlist.append(dv)
        y=((a*ee)/(nn*NA)+b)*(nn*NA)*13.e-3
        y=np.max([0,y])
        ylist.append(y)
    plt.plot(xlist, ylist, label=lab, color=color_list[2*jj])
    

plt.xlabel('dV (kV)')
plt.ylabel('Num of Photon/e')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height*0.6])
ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.grid('on')
plt.savefig(savedir+'PhotonCreation_Naive.png')
    
