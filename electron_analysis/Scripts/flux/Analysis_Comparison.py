#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 11:30:41 2022

@author: yasaman
"""

import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
from numpy import genfromtxt


ref_flux_file = '/Users/yasaman/AMS02/data/Niko/LeptonFluxes_Zimmermann_25_10_2018_2D.root'
#ref_acceptance_file = '/Users/yasaman/AMS02/data/Niko/LeptonAnalysis_EffectiveAcceptance_Average (1).root'

Trigger_file = '/Users/yasaman/AMS02/data/TriggerPass6/Triggercount.root'
Time_file = '/Users/yasaman/AMS02/data/time/MeasuringTime_00000.root'
MC_momentum_file = '/Users/yasaman/AMS02/data/MCmomentumPass6/Electron_Mc_Momentum_Edependent_result_Test_good.npz' 
el_number_adress = '/Users/yasaman/AMS02/plots/template_fit/'

def lafferty_whyatt(edges, gamma):
     ex = 1 - gamma
     rmin = edges[:-1]
     rmax = edges[1:]
     return ((rmax - rmin) * ex / (rmax**ex - rmin**ex))**(1 / gamma)
#-----------------------------------------------------------------------------------    
 
with uproot.open(ref_flux_file) as file:
    
    Graf=file['SxFluxTools_Electron_RWTH/grElectron_RWTH_FluxBeforeUnfolding']
    ref_Energy, ref_flux = Graf.values()
    _ , ref_flux_error = Graf.errors('mean')
    
    Graf=file['SxFluxTools_Electron_RWTH/grElectron_RWTH_FluxBeforeUnfoldingE3']
    ref_Energy_E3, ref_flux_E3 = Graf.values()
    _ , ref_flux_error_E3 = Graf.errors('mean')
    
    ref_trigger_efficiency = file['SxFluxTools_Electron_RWTH/grElectron_RWTH_TriggerEfficiency'].values()
    ref_Ecal_efficiency = file['SxFluxTools_Electron_RWTH/grElectron_RWTH_EcalBdtEfficiency'].values()
        
    
    # Graf=file['SxFluxTools_Electron_RWTH/grElectron_RWTH_MeasurementTime']
    # _, ref_time = Graf.values()
    ref_time = file['SxFluxTools_Electron_RWTH/grElectron_RWTH_MeasurementTime'].values()
    
    el_refnumber = file['SxFluxTools_Electron_RWTH/grElectron_RWTH_Raw'].values() 
    el_refnumber_err = file ['SxFluxTools_Electron_RWTH/grElectron_RWTH_RelErrorBeforeUnfoldingStat'].values()
    print(el_refnumber_err[0] )
        
    ref_acceptance = file['SxFluxTools_Electron_RWTH/grElectron_RWTH_RawAcceptance'].values()
    
#------------------------------------------------------------------------------------  
    
with uproot.open(Trigger_file) as file:
    trigger_values , trigger_edge = file['TriggerCounts'].to_numpy()

with uproot.open(Time_file) as file:
    time_values , time_edge = file['MeasuringTime/fIntegratedMeasuringTimeOverCutOff'].to_numpy() 
    
with np.load(MC_momentum_file) as file:
    MC_binning = file["var1_binning"]
    MC_momentum_values = file["Events"]    
    
#print(MC_momentum_values)    
el_number = np.loadtxt(el_number_adress+'el_count.txt')
el_error = np.loadtxt(el_number_adress+"el_count_error.txt")
#------------------------------------------------------------------------------------
E = np.array([0.5,0.65,0.82,1.01,1.22,1.46,1.72,2.0,2.31,2.65,3.0,3.36,3.73,
             4.12,4.54,5.0,5.49,6.0,6.54,7.1,7.69,8.3,8.95,9.62,10.32,11.04,
             11.8,12.59,13.41,14.25,15.14,16.05,17.0,17.98,18.99,20.04,21.13,
             22.25,23.42,24.62,25.9,27.25,28.68,30.21,31.82,33.53,35.36,37.31,
             39.39,41.61,44.0,46.57,49.33,52.33,55.58,59.13,63.02,67.3,72.05,
             77.37,83.36,90.19,98.08,107.3,118.4,132.1,148.8,169.9,197.7,237.2,
             290.0,370.0,500.0,700.0,1000.0])

deltaE = E[1:]-E[:-1]
Energy_bins= lafferty_whyatt(E, 3)
#------------------------------------------------------------------------------------

MC_momentum = MC_momentum_values.sum(axis=0)    
acceptance = (3.9)**2 *np.pi*(MC_momentum/trigger_values)
flux = el_number/(acceptance*time_values*deltaE*ref_trigger_efficiency[1])    
flux_err = el_error/(acceptance*time_values*deltaE*ref_trigger_efficiency[1])
NikoFlux= el_refnumber[1]/(ref_acceptance[1]*ref_time[1]*deltaE*ref_trigger_efficiency[1])
NikoFlux_err = el_refnumber[1]*el_refnumber_err[1]/(ref_acceptance[1]*ref_time[1]*deltaE*ref_trigger_efficiency[1])
#------------------------------------------------------------------------------------- 


###########Flux Plot
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

plot.errorbar(ref_Energy[1:],NikoFlux[1:],ref_flux_error[1:],fmt='.',marker='^',markersize=10, color='k',label="Nikolas")
plot.errorbar(Energy_bins[1:],flux[1:],flux_err[1:],fmt='.',markersize=16,color='m',label='Yasaman')
plt.legend(fontsize=25)
 
plot.set_xscale("log")       
plt.show()
figure.savefig("/Users/yasaman/AMS02/plots/flux/flux.pdf" , dpi=250)
#--------------------------------------------------------------------------------------
#######Scaled Flux
figure = plt.figure(figsize=(12, 10))
ax1, ax2 = figure.subplots(2,1,gridspec_kw={'height_ratios': [3,1]})
#plot = figure.subplots(1, 1)
#ax1.set_xlabel("Energy/Gev",fontsize = 16,fontweight = 'bold')
FluxRatio_err = (flux/NikoFlux)*np.sqrt((flux_err/flux)**2 + (ref_flux_error/NikoFlux)**2)
ax1.set_ylabel(r"$ \rm E^3 \; \Phi_{\rm e^-}/( GeV^{2} \; m^{-2} \; sr^{-1} \; s^{-1})$", fontsize = 16,fontweight = 'bold')
ax1.errorbar(ref_Energy[1:],NikoFlux[1:]*ref_Energy[1:]**3,ref_flux_error[1:]*ref_Energy[1:]**3,markersize=16,fmt='.',color='g',label="Nikolas")
ax1.errorbar(Energy_bins[1:],(flux[1:]*Energy_bins[1:]**3),(flux_err[1:]*Energy_bins[1:]**3),fmt='.',markersize=16,color='m',label='Yasaman')
ax1.legend(fontsize=15)
ax1.set_xscale("log")  

ax2.set_xlabel("Energy/Gev",fontsize = 16,fontweight = 'bold')
ax2.set_ylabel("My flux/Niko's flux", fontsize = 16,fontweight = 'bold')
ax2.errorbar(Energy_bins[1:],flux[1:]/NikoFlux[1:],(flux_err[1:])/NikoFlux[1:], fmt='.', color='k')
ax2.grid()
ax2.set_xscale("log") 
     
plt.show()
figure.savefig("/Users/yasaman/AMS02/plots/flux/Scaled_flux.pdf" , dpi=250)
#---------------------------------------------------------------------------------------

#########flux ratio

figure = plt.figure(figsize=(12, 10))
ax1, ax2 = figure.subplots(2,1,gridspec_kw={'height_ratios': [3,1]})
plot = figure.subplots(1, 1)
plot.set_xlabel("Energy/Gev",fontsize = 26,fontweight = 'bold')
plot.set_ylabel("My flux/Niko's flux", fontsize = 26,fontweight = 'bold')
plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
for axis in ['top','bottom','left','right']:
    plot.spines[axis].set_linewidth(2)
    
plot.scatter(Energy_bins,flux/NikoFlux)
plot.grid()
plot.set_ylim(0.8,1.1)
plt.hlines(1.05,0.5,1000)
plt.hlines(0.955,0.5,1000) 
plot.set_xscale("log")       
plt.show()
figure.savefig("/Users/yasaman/AMS02/plots/flux/fluxRatio.pdf" , dpi=250)
#--------------------------------------------------------------------------------------
######### Measuring time 
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
plot.step(np.concatenate(([time_edge[0]], time_edge)), np.concatenate(([0], time_values/86400 , [0])), where="post",linewidth=3.5, color='m', label="Yasaman")
plot.step(np.concatenate(([time_edge[0]], time_edge)), np.concatenate(([0], ref_time[1]/86400 , [0])), where="post",linewidth=3.5, color='k', label = "Niko")
plt.legend(fontsize=25)
plt.show() 
figure.savefig("/Users/yasaman/AMS02/plots/MTimenew.pdf" , dpi=250)   
#----------------------------------------------------------------------------------------
########## Acceptance
ref_Ac_after_Id = genfromtxt('/Users/yasaman/AMS02/data/Acceptance_digitizer/Acceptance_AfterIDcuts.csv', delimiter=',')

figure = plt.figure(figsize=(12, 10))
ax1, ax2 = figure.subplots(2,1,gridspec_kw={'height_ratios': [3,1]})

ax1.scatter(ref_acceptance[0][1:],ref_acceptance[1][1:], label="Niko's data", color='k')
ax1.scatter(Energy_bins[1:], (acceptance[1:]), label= "my data",color='m')
#l6=ax1.scatter(ref_Ac_after_Id[:,0],ref_Ac_after_Id[:,1]*10**-4, c ='r', label="Niko: After Identification cuts (digitizer)") 
ax1.set_xscale("log")   
ax1.set_ylabel(r"$\rm Acceptance/(m^2 \; sr)$",fontsize=20, fontweight='bold')

ax2.grid()
ax2.scatter(Energy_bins[1:], (acceptance[1:])/ref_acceptance[1][1:], label="my data/ Niko's data", color='r')
ax2.hlines(1,0.5,1000,linestyle='--', linewidth=3, color='k', label="y=1")
ax2.set_xlabel("Energy/Gev",fontsize = 20,fontweight = 'bold')
ax2.set_ylabel("Acceptance ratio",fontsize = 20,fontweight = 'bold')

ax2.set_xscale("log")
ax1.legend(loc='upper right')
#ax2.set_ylim(0.9,1.5)
#figure.legend([l1, l2, l3, l4,l6], labels=["Niko's data","My data","My data/Niko's data","y=1","Niko: After Identification cuts (digitizer)" ], bbox_to_anchor=(0.4, 0.5))  
 
plt.show()
figure.savefig("/Users/yasaman/AMS02/plots/acceptance.pdf" , dpi=250)
#-----------------------------------------------------------------------------------
########electron number

figure = plt.figure(figsize=(12, 10))
ax1, ax2 = figure.subplots(2,1,gridspec_kw={'height_ratios': [3,1]})

l1 =ax1.errorbar(el_refnumber[0][1:],el_refnumber[1][1:],el_refnumber_err[1][1:], fmt='o', label="Niko's data", color='k')
l2 =ax1.errorbar(Energy_bins[1:], el_number[1:],el_error[1:], fmt='o', label= "my data",color="m") 
ax1.set_xscale("log")   
ax1.set_yscale("log")
ax1.set_ylabel("electron number", fontsize=16,fontweight='bold')


l3 = ax2.errorbar(Energy_bins[1:], el_number[1:]/el_refnumber[1][1:],el_error[1:]/el_refnumber[1][1:], fmt='o',label="my data/ Niko's data", color="r")
#ax2.scatter(Energy_bins, el_number/el_refnumber[1], label="my data/ Niko's data", color="r")
l4 =ax2.hlines(1,0.5,1000,linestyle='--', linewidth=3, color='k', label="y=1")
ax2.set_xlabel("Energy/Gev",fontsize = 26,fontweight = 'bold')
ax2.set_ylabel("electron number ratio",fontsize=16, fontweight='bold')

#ax2.set_ylim(0.8,1.1)
ax2.grid()
ax2.set_xscale("log")
figure.legend([l1, l2, l3, l4], labels=["Niko's data","My data","y=1","My data/Niko's data"], bbox_to_anchor=(0.4, 0.5))

    
plt.show()
figure.savefig("/Users/yasaman/AMS02/plots/el_number.pdf" , dpi=250)   
#-------------------------------------------------------------------------------------
###### comparison plot
figure = plt.figure(figsize=(16, 10))
plot = figure.subplots(1, 1)
plot.set_xlabel("Energy/Gev",fontsize = 26,fontweight = 'bold')
plot.set_ylabel(r"$\rm \Phi_{e^-} /( GeV^{-1} \; m^{-2} \; sr^{-1} \; s^{-1})$", fontsize = 26,fontweight = 'bold')
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
    
plt.hlines(0,0.5,1000,linestyle='--', color='k')
average = (flux + NikoFlux)/2
myfluxerr = np.sqrt(flux_err**2 + (0.5*np.sqrt(flux_err**2 + ref_flux_error**2 ))**2)
nikofluxerr = np.sqrt(ref_flux_error**2 + (0.5*np.sqrt(flux_err**2 + ref_flux_error**2 ))**2)
plt.errorbar(Energy_bins[1:], (flux[1:] - average[1:])/average[1:],myfluxerr[1:]/average[1:], fmt='o', color='m' , label = "My Flux")
plt.errorbar(Energy_bins[1:], (NikoFlux[1:] - average[1:])/average[1:],nikofluxerr[1:]/average[1:], fmt='o', color='g' , label = "Niko's Flux") 
plt.xlabel("Energy/Gev",fontsize = 26,fontweight = 'bold')
plt.ylabel(r"$( \rm Flux - Average Flux)/Average Flux$",fontsize = 20,fontweight = 'bold')
plt.xscale('log')
plt.legend(fontsize=25, loc="upper left")
plt.ylim(-0.1,0.1)
plt.show()
figure.savefig("/Users/yasaman/AMS02/plots/flux/fluxcomparison.pdf" , dpi=250)
