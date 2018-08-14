# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


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

workdir = "/home/wxj/gastest/xenonProperties/"

xx = np.loadtxt(workdir+"/xenonDrift.txt");
