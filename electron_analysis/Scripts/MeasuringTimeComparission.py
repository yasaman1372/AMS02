#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 13:11:42 2022

@author: yasaman
"""

import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors



def lafferty_whyatt(edges, gamma):
     ex = 1 - gamma
     rmin = edges[:-1]
     rmax = edges[1:]
     return ((rmax - rmin) * ex / (rmax**ex - rmin**ex))**(1 / gamma)


with uproot.open('/Users/yasaman/AMS02/data/Niko/LeptonFluxes_Zimmermann_06_08_2019_2D.root') as file:
    myfile = file['SxFluxTools_Electron_RWTH']
    ref_time_values = myfile['grElectron_RWTH_MeasurementTime'].values()
    
with uproot.open('/Users/yasaman/AMS02/data/time/MeasuringTime_00000.root') as time_file:
    time_values , time_edge = time_file['MeasuringTime/fIntegratedMeasuringTimeOverCutOff'].to_numpy()    
    
print(max(time_values))
figure = plt.figure(figsize=(12, 10))
plot = figure.subplots(1, 1)
plot.set_xlabel("Energy/Gev",fontsize = 26,fontweight = 'bold')
plot.set_ylabel("Measuring time/days", fontsize = 26,fontweight = 'bold')
#plot.set_yscale("log")
plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
# plot.set_xlim(0.72,1.35)
#plot.set_ylim(bottom=0)
for axis in ['top','bottom','left','right']:
    plot.spines[axis].set_linewidth(2)
plot.set_xscale("log")  
Time_values= time_values/86400    
plot.step(np.concatenate(([time_edge[0]], time_edge)), np.concatenate(([0], time_values/86400 , [0])), where="post",linewidth=3.5, color='m', label="Yasaman")
plot.step(np.concatenate(([time_edge[0]], time_edge)), np.concatenate(([0], ref_time_values[1]/86400 , [0])), where="post",linewidth=3.5, color='k', label = "Niko")
plt.legend(fontsize=25)
plt.show() 
figure.savefig("/Users/yasaman/AMS02/plots/MTimenew.pdf" , dpi=250)   
print(ref_time_values)





