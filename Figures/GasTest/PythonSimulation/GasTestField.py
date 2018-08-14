# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:57:36 2018

@author: wei
"""
X=10000

printless = False
#printless = True
workdir ="/media/wei/ACA8-1ECD1/ComsolGeoFile/"
import sys
sys.path.insert(0, "/media/wei/ACA8-1ECD/SystemTest//XenonProperty/")
sys.path.insert(0, "/media/wei/ACA8-1ECD1/SystemTest//XenonProperty/")
sys.path.insert(0, workdir)
import unit
import CalculateField as cf
import numpy as np
###########################################################
####System Test Current.
###########################################################

#LengthUnit="mm", ForceUnit="N", StressUnit="MPa",
GridList=[]
TopPlate = cf.LZ_Grid(GridName = "Top Plate",WireDiameter = 1465.*unit.m, WirePitch = 0.*unit.m, GridType = "P", z_Location = (65+8+X)*unit.m, Tension = 0.,GridDiameter = 130.*unit.m , MaterialMaxStress = 290.*unit.M)
#TopPlate.ElectricFieldUpInf = 0. # in V/mm
#TopPlate.ElectricFieldDownInf = .114e2  #in V/mm
#TopPlate.Print()
GridList.append(TopPlate)

AnodeGrid = cf.LZ_Grid(GridName = "Anode Grid",WireDiameter = 100.e-3*unit.m, WirePitch = 2.5*unit.m, GridType = "WW", z_Location = (8+X)*unit.m, Tension = 2.5,GridDiameter = 130.*unit.m , MaterialMaxStress = 290.*unit.M)
#AnodeGrid.ElectricFieldUpInf = .114e2 # in V/mm
#AnodeGrid.ElectricFieldDownInf = -10.2e2  #in V/mm
#AnodeGrid.Print()
GridList.append(AnodeGrid)

GateGrid = cf.LZ_Grid(GridName = "Gate Grid",WireDiameter = 75.e-3*unit.m, WirePitch = 5.*unit.m, GridType = "WW", z_Location = (-5+X)*unit.m, Tension = 3.3,GridDiameter = 130.*unit.m , MaterialMaxStress = 290.*unit.M)
#GateGrid.ElectricFieldUpInf = -6.1e2 # in V/mm
#GateGrid.ElectricFieldDownInf = -.366e2  #in V/mm
#GateGrid.Print()
GridList.append(GateGrid)

BottomPlate = cf.LZ_Grid(GridName = "Bottom Plate",WireDiameter = 1465.*unit.m, WirePitch = 0.*unit.m, GridType = "P", z_Location = (-5-110.5+X)*unit.m, Tension = 0.,GridDiameter = 130.*unit.m , MaterialMaxStress = 290.*unit.M)
#BottomPlate.ElectricFieldUpInf = 2.90e2 # in V/mm
#BottomPlate.ElectricFieldDownInf = .25e2  #in V/mm
#BottomPlate.Print()
GridList.append(BottomPlate)

print "############################################################"
print "############################################################"
print "############################################################"
print "############################################################"
print "System Test Current. 50 kV Cathode. "

##################################"

SYS_Gas= cf.LZ_detector(GridList)
VList=[-1.5e3,4e3,-4e3,-1.5e3,]
SYS_Gas.UpdateVoltage(VList)
SYS_Gas.Print(printless)


drift_voltage_list=[]
for a in np.arange(0,8.5,0.5):
    VList=[-1.5e3,a*1e3,-a*1.e3,-1.5e3,]
    SYS_Gas.UpdateVoltage(VList)
    drift_voltage_list.append(SYS_Gas.GridList[1].ElectricFieldDownInf/1.e5)
    print SYS_Gas.GridList[1].ElectricFieldDownInf/1.e5
#    print SYS_Gas.GridList[2].SurfaceFieldNaive()



surface_voltage_list=[]
for a in range(7):
    VList=[-1.5e3,a*1e3,-a*1.e3,-1.5e3,]
    SYS_Gas.UpdateVoltage(VList)
    surface_voltage_list.append(SYS_Gas.GridList[1].SurfaceFieldNaive()/1.e5)
    print 'dv = %.0f kV'%(a*2)
    print 'Anode: %.1f, Gate: %.1f'%(SYS_Gas.GridList[1].SurfaceFieldNaive()[0]/1.e5, SYS_Gas.GridList[2].SurfaceFieldNaive()[0]/1.e5)

VList=[-1.5e3,4*1e3,-4*1.e3,-1.5e3,]
SYS_Gas.UpdateVoltage(VList)
SYS_Gas.GridList[1].SurfaceFieldNaive()
SYS_Gas.GridList[2].SurfaceFieldNaive()