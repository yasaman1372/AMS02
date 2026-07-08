#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:15:27 2023

@author: yasaman
"""

import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
from numpy import genfromtxt


def lafferty_whyatt(edges, gamma):
     ex = 1 - gamma
     rmin = edges[:-1]
     rmax = edges[1:]
     return ((rmax - rmin) * ex / (rmax**ex - rmin**ex))**(1 / gamma)
 
E = np.array([0.5,0.65,0.82,1.01,1.22,1.46,1.72,2.0,2.31,2.65,3.0,3.36,3.73,
             4.12,4.54,5.0,5.49,6.0,6.54,7.1,7.69,8.3,8.95,9.62,10.32,11.04,
             11.8,12.59,13.41,14.25,15.14,16.05,17.0,17.98,18.99,20.04,21.13,
             22.25,23.42,24.62,25.9,27.25,28.68,30.21,31.82,33.53,35.36,37.31,
             39.39,41.61,44.0,46.57,49.33,52.33,55.58,59.13,63.02,67.3,72.05,
             77.37,83.36,90.19,98.08,107.3,118.4,132.1,148.8,169.9,197.7,237.2,
             290.0,370.0,500.0,700.0,1000.0])

deltaE = E[1:]-E[:-1]
Energy_bins= lafferty_whyatt(E, 3)    
 
Trigger_file_pass8 = "/Users/yasaman/AMS02/data/pass8/Triggercount.root"
with uproot.open(Trigger_file_pass8) as file:
    trigger_values , trigger_edge = file['TriggerCounts'].to_numpy() 


Time_file_pass8 = "/Users/yasaman/AMS02/data/pass8/MeasuringTime_00000.root"    
with uproot.open(Time_file_pass8) as file:
    time_values , time_edge = file['MeasuringTime/fIntegratedMeasuringTimeOverCutOff'].to_numpy() 
    
    
MC_momentum_file_pass8 = "/Users/yasaman/AMS02/data/pass8/Electron_Mc_Momentum_Edependent_result_Test.npz"
with np.load(MC_momentum_file_pass8) as file:
    MC_binning = file["var1_binning"]
    MC_momentum_values = file["Events"] 
    


template_fit_file = np.load("/Users/yasaman/AMS02/YasamanAnalysis/Scripts/template_fit/results/parameters_all.npz") 
el_number = template_fit_file["nel"]  
pos_number = template_fit_file["npos"]



template_fit_error_file = np.load("/Users/yasaman/AMS02/YasamanAnalysis/Scripts/template_fit/results/errors_all.npz") 
el_error = template_fit_error_file["nel"]
pos_error = template_fit_error_file["npos"]


MC_momentum = MC_momentum_values.sum(axis=0)    
acceptance = (3.9)**2 *np.pi*(MC_momentum/trigger_values)

el_flux = el_number/(acceptance*time_values*deltaE)    
el_flux_err = el_error/(acceptance*time_values*deltaE)

pos_flux = pos_number/(acceptance*time_values*deltaE)    
pos_flux_err = pos_error/(acceptance*time_values*deltaE)

############################################################################### Electron Flux Plot

figure = plt.figure(figsize=(12, 10))
plot = figure.subplots(1, 1)
plot.set_xlabel("Energy/Gev",fontsize = 26,fontweight = 'bold')
plot.set_ylabel(r"$\rm \Phi_{e^-} /( GeV^{-1} \; m^{-2} \; sr^{-1} \; s^{-1})$", fontsize = 26,fontweight = 'bold')
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

plot.errorbar(Energy_bins[1:],el_flux[1:],el_flux_err[1:],fmt='.',markersize=16,color='m')
 
plot.set_xscale("log")       
plt.show()
figure.savefig("/Users/yasaman/AMS02/plots/flux/electronflux.pdf" , dpi=250)

############################################################################### Positron Flux plot

figure = plt.figure(figsize=(12, 10))
plot = figure.subplots(1, 1)
plot.set_xlabel("Energy/Gev",fontsize = 26,fontweight = 'bold')
plot.set_ylabel(r"$\rm \Phi_{e^+} /( GeV^{-1} \; m^{-2} \; sr^{-1} \; s^{-1})$", fontsize = 26,fontweight = 'bold')
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

plot.errorbar(Energy_bins[1:],pos_flux[1:],pos_flux_err[1:],fmt='.',markersize=16,color='m')
 
plot.set_xscale("log")       
plt.show()
figure.savefig("/Users/yasaman/AMS02/plots/flux/electronflux.pdf" , dpi=250)

###############################################################################Electron Scaled Flux
figure = plt.figure(figsize=(12, 10))
ax1, ax2 = figure.subplots(2,1,gridspec_kw={'height_ratios': [3,1]})

ax1.set_ylabel(r"$ \rm E^3 \; \Phi_{\rm e^-}/( GeV^{2} \; m^{-2} \; sr^{-1} \; s^{-1})$", fontsize = 16,fontweight = 'bold')
ax1.errorbar(Energy_bins,(el_flux*Energy_bins**3),(el_flux_err*Energy_bins**3),fmt='.',markersize=16,color='m',label='YN')

ref_flux_file_published = '/Users/yasaman/AMS02/data/electron_published_data/electron flux/ssdc_cosmicrays_2023-10-05-4/T2_AMS_rev_000.txt'
file = np.loadtxt(ref_flux_file_published)
published_flux_x = file[:,0]
published_flux_y = file[:,3]


ax1.scatter(published_flux_x, published_flux_y*(published_flux_x**3), label = "published" )
ax1.legend(fontsize=15)
ax1.set_xscale("log")  

ax2.set_xlabel("Energy/Gev",fontsize = 16,fontweight = 'bold')
ax2.set_ylabel("YN/published", fontsize = 16,fontweight = 'bold')
ax2.scatter(Energy_bins, el_flux/published_flux_y[:-1])
ax2.grid()
ax2.set_ylim(0,2)
ax2.set_xscale("log") 
     
plt.show()
figure.savefig("/Users/yasaman/AMS02/plots/flux/electron_Scaled_electron_flux.pdf" , dpi=250)

################################################################################positron Scaled Flux

figure = plt.figure(figsize=(12, 10))
ax1, ax2 = figure.subplots(2,1,gridspec_kw={'height_ratios': [3,1]})
ax1.set_ylabel(r"$ \rm E^3 \; \Phi_{\rm e^+}/( GeV^{2} \; m^{-2} \; sr^{-1} \; s^{-1})$", fontsize = 16,fontweight = 'bold')
ax1.errorbar(Energy_bins[:-1],(pos_flux[:-1]*Energy_bins[:-1]**3),(pos_flux_err[:-1]*Energy_bins[:-1]**3),fmt='.',markersize=16,color='m',label='YN')

ref_flux_file_published = '/Users/yasaman/AMS02/data/positron_published_data/positron flux/ssdc_cosmicrays_2023-10-05-2/T1_AMS_rev_000.txt'
file = np.loadtxt(ref_flux_file_published)
published_flux_x = file[:,0]
published_flux_y = file[:,3]


ax1.scatter(published_flux_x, published_flux_y*(published_flux_x**3), label = "published" )
ax1.legend(fontsize=15)
ax1.set_xscale("log")  

ax2.set_xlabel("Energy/Gev",fontsize = 16,fontweight = 'bold')
ax2.set_ylabel("YN/published", fontsize = 16,fontweight = 'bold')
ax2.scatter(Energy_bins, pos_flux/published_flux_y)
ax2.grid()
ax2.set_ylim(0,2)
ax2.set_xscale("log") 
     
plt.show()
figure.savefig("/Users/yasaman/AMS02/plots/flux/positron_Scaled_electron_flux.pdf" , dpi=250)

############################################################################### Measuring time 

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
plot.set_xlim(0.5,1000)
# plot.set_ylim(bottom=0)
for axis in ['top','bottom','left','right']:
    plot.spines[axis].set_linewidth(2)
plot.set_xscale("log")      
plot.step(np.concatenate(([time_edge[0]], time_edge)), np.concatenate(([0], time_values/86400 , [0])), where="post",linewidth=3.5, color='m')
plt.show() 
figure.savefig("/Users/yasaman/AMS02/plots/flux/MTime.pdf" , dpi=250)

############################################################################### Acceptance

accfile = np.loadtxt('/Users/yasaman/AMS02/data/T2_AMS_rev_000.txt', comments='#')

with uproot.open(accfile) as file:
    ref_acceptance_el = file['SxFluxTools_Electron_RWTH/grElectron_RWTH_RawAcceptance'].values()

figure = plt.figure(figsize=(12, 10))
ax1, ax2 = figure.subplots(2,1,gridspec_kw={'height_ratios': [3,1]})

ax1.scatter(ref_acceptance_el[0][1:],ref_acceptance_el[1][1:], label="Niko's data", color='k')
ax1.scatter(Energy_bins[1:], (acceptance[1:]), label= "my data",color='m')
ax1.set_xscale("log")   
ax1.set_ylabel(r"$\rm Acceptance/(m^2 \; sr)$",fontsize=20, fontweight='bold')

ax2.grid()
ax2.scatter(Energy_bins[1:], (acceptance[1:])/ref_acceptance_el[1][1:], label="my data/ Niko's data", color='r')
ax2.hlines(1,0.5,1000,linestyle='--', linewidth=3, color='k', label="y=1")
ax2.set_xlabel("Energy/Gev",fontsize = 20,fontweight = 'bold')
ax2.set_ylabel("Acceptance ratio",fontsize = 20,fontweight = 'bold')

ax2.set_xscale("log")
ax1.legend(loc='upper right')
  
plt.show()
figure.savefig("/Users/yasaman/AMS02/plots/flux/acceptance.pdf" , dpi=250)

############################################################################### electron number

figure = plt.figure(figsize=(12, 10))
plot= figure.subplots(1, 1)

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

plot.errorbar(Energy_bins[1:], el_number[1:],el_error[1:], fmt='o', label= "electron",color="m") 
plot.errorbar(Energy_bins[1:], pos_number[1:],pos_error[1:], fmt='o', label= "positron",color="k") 
plt.legend(fontsize=15)

plot.set_xscale("log")   
plot.set_yscale("log")
plot.set_ylabel("Events number", fontsize=26,fontweight='bold')
plot.set_xlabel("Energy/Gev",fontsize = 26,fontweight = 'bold')

plt.show()
figure.savefig("/Users/yasaman/AMS02/plots/flux/el_pos_number.pdf" , dpi=250)   

############################################################################### positron number

figure = plt.figure(figsize=(12, 10))
plot= figure.subplots(1, 1)

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

plot.errorbar(Energy_bins[1:], pos_number[1:],pos_error[1:], fmt='o', label= "my data",color="m") 
plot.set_xscale("log")   
plot.set_yscale("log")
plot.set_ylabel("positron number", fontsize=26,fontweight='bold')
plot.set_xlabel("Energy/Gev",fontsize = 26,fontweight = 'bold')

plt.show()
figure.savefig("/Users/yasaman/AMS02/plots/flux/pos_number.pdf" , dpi=250)  

############################################################################### positron flux ratio
figure = plt.figure(figsize=(12, 10))
ax1, ax2 = figure.subplots(2,1,gridspec_kw={'height_ratios': [3,1]})


flux_ratio = pos_number/(el_number+pos_number)
flux_ratio_err = np.sqrt((pos_error**2)*(el_number/(pos_number+el_number)**2)**2 + (el_error**2)*(pos_number/(pos_number+el_number)**2)**2)


ax1.set_ylabel("flux ratio", fontsize = 16,fontweight = 'bold')
ax1.set_xlabel("Energy GeV", fontsize = 16,fontweight = 'bold')



ref_flux_file_published = "/Users/yasaman/AMS02/data/ssdc_cosmicrays_2023-08-21/T3_AMS_rev_000.txt"
file = np.loadtxt(ref_flux_file_published)
published_fluxratio_x = file[:,0]
published_fluxratio_y = file[:,3]
published_fluxratio_error = file[:,5]

ax1.errorbar(Energy_bins,flux_ratio,flux_ratio_err, fmt='.',color='m',ms=15,label='YN')
ax1.errorbar(published_fluxratio_x, published_fluxratio_y, published_fluxratio_error, fmt='.',ms=15,label = "published" )
ax1.legend(fontsize=15)
ax1.set_xscale("log")  

ax2.set_xlabel("Energy/Gev",fontsize = 16,fontweight = 'bold')
ax2.set_ylabel("YN/published", fontsize = 16,fontweight = 'bold')
ax2.scatter(Energy_bins[:-1],flux_ratio[:-1]/published_fluxratio_y[:-1], color='k')
ax2.grid()
#ax2.set_ylim(0.95,1.05)
ax2.set_xscale("log") 
     
plt.show()
figure.savefig("/Users/yasaman/AMS02/plots/flux/positron_flux_ratio.pdf" , dpi=250)
 














