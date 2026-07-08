#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 11:43:50 2022

@author: yasaman
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

with np.load(os.path.join("/Users/yasaman/AMS02/data/He/results.npz")) as result_file:
   var1_binning = result_file["var1_binning"]
   Events = result_file["Events"]
   
figure = plt.figure(figsize=(12, 10))
plot = figure.subplots(1, 1)
plot.set_xlabel("He estimator",fontsize = 26,fontweight = 'bold')
plot.set_ylabel("Events", fontsize = 26,fontweight = 'bold')
plot.set_yscale("log")
plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
# plot.set_xlim(0.72,1.35)
# plot.set_ylim(10**-4,0.2)
for axis in ['top','bottom','left','right']:
    plot.spines[axis].set_linewidth(2)   
    
plot.errorbar((var1_binning[1:] + var1_binning[:-1]) / 2, Events, np.sqrt(Events), fmt=".")    
plot.step(np.concatenate(([var1_binning[0]], var1_binning)), np.concatenate(([0], Events, [0])), where="post",linewidth=2)
plt.show()
figure.savefig("/Users/yasaman/AMS02/plots/He_histogram.pdf" , dpi=250)    