# -*- coding: utf-8 -*-
"""
Created on Thu May 31 12:40:12 2018

@author: wxj
"""



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