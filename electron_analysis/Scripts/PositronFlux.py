#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:22:34 2023

@author: yasaman
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from scipy import interpolate

DM = genfromtxt("/Users/yasaman/Downloads/Default Dataset (9).csv", delimiter=',')
PL = genfromtxt("/Users/yasaman/Downloads/Default Dataset (7).csv", delimiter=',')
PositronData = np.loadtxt("/Users/yasaman/Downloads/ssdc_cosmicrays_2023-03-18/e+_AMS_PRL2019_ekin_000.txt",comments="#") 

pl=PL[PL[:, 0].argsort()]
PLf = interpolate.UnivariateSpline(pl[:,0], pl[:,1], s=0.3)
PLx= np.logspace(-0.25,3.5,100000)
PLy= PLf(PLx)

dm=DM[DM[:, 0].argsort()]
PLf = interpolate.UnivariateSpline(dm[:,0], dm[:,1], s=0.4)
DMx= np.logspace(-0.25,2.95,100000)
DMy= PLf(DMx)


figure = plt.figure(figsize=(12, 10))
plot = figure.subplots(1, 1)

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


plot.errorbar(PositronData[:,0], PositronData[:,3]*(PositronData[:,0])**3,yerr=PositronData[:,4]*(PositronData[:,0])**3, ms=15,color="r",fmt='.', label="AMS-02 data")

plot.plot(DMx, DMy,color="m", linewidth=3,label="Dark Matter Model")
plot.plot(PLx, PLy,color="k",linewidth=3,label="Pulsar Model")
#plot.plot(PL[0], PL[1], label ="Pulsar Model")
plot.set_xlabel("Energy [GeV]",fontsize = 26,fontweight = 'bold')
plot.set_ylabel(r"$ \rm E^3 \; \Phi_{\rm e^+}/( GeV^{2} \; m^{-2} \; sr^{-1} \; s^{-1})$", fontsize = 20,fontweight = 'bold')
plot.set_xscale("log")
plot.set_xlim(right=5000)
plot.legend(fontsize=25)
plt.show()
figure.savefig("/Users/yasaman/AMS02/positronflux.pdf" , dpi=250)