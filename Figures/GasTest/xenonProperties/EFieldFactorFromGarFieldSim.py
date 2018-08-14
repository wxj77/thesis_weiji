# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:28:40 2018

@author: wxj
"""
# the garfield simulation was done withe a 75 um diameter wire in a 1000 mm OD cylinder. this means the surface field on the wire is 102.9kV/cm
# So to get the surface electric field on the wire on each operation voltage, I put a factor to sclae the electric field down, this document is the list of factors
#EFieldAveFactor[ii] is if Va-Vg=ii kV, what is the average surface field on the gate wire.
#EFieldMaxFactor[ii] is if Va-Vg=ii kV, what is the maximum surface field on the gate wire.

EFieldMaxFactor=[0, 0.083, 0.167, 0.250, 0.333, 0.417, 0.500, 0.583, 0.667, 0.750, 0.833, 0.917, 1.000, 1.083, 1.167, 1.250, 1.333];
EFieldAveFactor=[0, 0.071, 0.143, 0.214, 0.286, 0.358, 0.430, 0.501, 0.573, 0.645, 0.716, 0.788, 0.860, 0.932, 1.003, 1.075, 1.147]; 