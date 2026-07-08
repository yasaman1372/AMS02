#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:47:13 2022

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
 
file_name='/Users/yasaman/AMS02/data/efficiency/MC/'    
with np.load(file_name+"results.npz") as result_file:
    Energy_binning = result_file["var_binning"]
    MCerr = result_file["err"]
    MCeff = result_file["eff"] 
    
file_name='/Users/yasaman/AMS02/data/efficiency/ISS/'    
with np.load(file_name+"results.npz") as result_file:
    Energy_binning = result_file["var_binning"]
    ISSeff = result_file["eff"] 
    ISSerr =result_file["err"]

correction = np.ones(len(ISSeff[0]))
for i in range(len(MCeff)):
    correction = correction * (ISSeff[i]/MCeff[i])    
 
 
with uproot.open('/Users/yasaman/AMS02/data/Niko/LeptonFluxes_Zimmermann_06_08_2019_2D.root') as file:
    Graf=file['SxFluxTools_Electron_RWTH/grElectron_RWTH_FluxBeforeUnfolding']
    ref_Energy, ref_flux = Graf.values()
    _ , ref_flux_error = Graf.errors('mean')
    
    Graf=file['SxFluxTools_Electron_RWTH/grElectron_RWTH_MeasurementTime']
    _, ref_time = Graf.values()
    
with uproot.open('/Users/yasaman/AMS02/data/Niko/LeptonAnalysis_EffectiveAcceptance_Average (1).root') as file:
    print(list(file))
    ref_acceptance = file['McEffectiveAcceptance_Electrons'].values()
print(ref_acceptance[0], ref_Energy, "ref acceptance")    
    
with uproot.open('/Users/yasaman/AMS02/data/Niko/LeptonFluxes_Zimmermann_06_08_2019_2D.root') as file:
    el_refnumber = file['SxFluxTools_Electron_RWTH/grElectron_RWTH_Raw'].values()    
    
   
# print(list(Graf))/Users/yasaman/Downloads/LeptonFluxes_Zimmermann_09_09_2019_2D (1).root
# ref_Energy=Graf[
# ref_flux=Graf[:,3]
# ref_flux_error=Graf[:,4]

E = np.array([0.5,0.65,0.82,1.01,1.22,1.46,1.72,2.0,2.31,2.65,3.0,3.36,3.73,
                             4.12,4.54,5.0,5.49,6.0,6.54,7.1,7.69,8.3,8.95,9.62,10.32,11.04,
                             11.8,12.59,13.41,14.25,15.14,16.05,17.0,17.98,18.99,20.04,21.13,
                             22.25,23.42,24.62,25.9,27.25,28.68,30.21,31.82,33.53,35.36,37.31,
                             39.39,41.61,44.0,46.57,49.33,52.33,55.58,59.13,63.02,67.3,72.05,
                             77.37,83.36,90.19,98.08,107.3,118.4,132.1,148.8,169.9,197.7,237.2,
                             290.0,370.0,500.0,700.0,1000.0])
deltaE = E[1:]-E[:-1]
x= lafferty_whyatt(E, 3)

print(ref_Energy.shape, x.shape, E.shape, ref_Energy, x, E) 

with uproot.open('/Users/yasaman/AMS02/data/Triggercount.root') as trigger_file:
    trigger_values , trigger_edge = trigger_file['TriggerCounts'].to_numpy()

with uproot.open('/Users/yasaman/AMS02/data/time/MeasuringTime_00000.root') as time_file:
    time_values , time_edge = time_file['MeasuringTime/fIntegratedMeasuringTimeOverCutOff'].to_numpy() 

print(max(time_values))    
    
 
file_name='/Users/yasaman/AMS02/data/MCmomentum/'    
with np.load(file_name+"results.npz") as result_file:
    MC_binning = result_file["var1_binning"]
    # MC_momentum_values = result_file["Events"][10:-40].sum(axis=0) 
    MC_momentum = result_file["Events"]
    MC_momentum_values = MC_momentum[-1][10:-40].sum(axis=0)
    print("TRD efficiency cut =", result_file["Events"][10:-40].sum(axis=0)/result_file["Events"].sum(axis=0)  )
    print(MC_momentum_values.shape)
    
el_number = np.loadtxt('/Users/yasaman/AMS02/plots/template_fit/el_count.txt')
el_error = np.loadtxt("/Users/yasaman/AMS02/plots/template_fit/el_count_error.txt")
acceptance  = np.loadtxt('/Users/yasaman/AMS02/data/acceptance_After_Id.txt')

# time_values= time_values/86400
#acceptance= (3.9)**2 *np.pi*(MC_momentum_values/trigger_values)
print(el_refnumber[1].shape, ref_acceptance[1].shape, ref_time.shape)
flux=el_number/(acceptance*time_values*deltaE)
#ref_flux = el_refnumber[1]*1e4/(ref_acceptance[1][1:-1]*ref_time*deltaE)
el_error= el_error/(acceptance*time_values*deltaE)


########flux
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

plot.errorbar(ref_Energy,ref_flux,ref_flux_error,fmt='.',marker='^',markersize=10, color='k',label="Niko's flux")
plot.errorbar(x[3:],flux[3:],el_error[3:],fmt='.',markersize=16,color='m',label='My flux')
# plot.errorbar(x[3:],flux[3:]/correction[3:],el_error[3:]/correction[3:],fmt='.',markersize=16,color='m',label='Corrected flux')

plt.legend(fontsize=25)
 
plot.set_xscale("log")       
plt.show()
figure.savefig("/Users/yasaman/AMS02/plots/flux/flux.pdf" , dpi=250)

######flux ratio
figure = plt.figure(figsize=(12, 10))
plot = figure.subplots(1, 1)
plot.set_xlabel("Energy/Gev",fontsize = 26,fontweight = 'bold')
plot.set_ylabel("My flux/Niko's flux", fontsize = 26,fontweight = 'bold')
# plot.set_yscale("log")
plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
for axis in ['top','bottom','left','right']:
    plot.spines[axis].set_linewidth(2)
    
plot.scatter(x[3:],flux[3:]/ref_flux[3:])
plot.grid()
plot.set_ylim(0.6,1.1)
plt.hlines(1.05,0.5,1000)
plt.hlines(0.955,0.5,1000) 
plot.set_xscale("log")       
plt.show()
figure.savefig("/Users/yasaman/AMS02/plots/flux/fluxRatio.pdf" , dpi=250)

####Time ratio
figure = plt.figure(figsize=(12, 10))
plot = figure.subplots(1, 1)
plot.set_xlabel("Energy/Gev",fontsize = 26,fontweight = 'bold')
plot.set_ylabel("Measuring time ratio", fontsize = 20,fontweight = 'bold')
# plot.set_yscale("log")
plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
for axis in ['top','bottom','left','right']:
    plot.spines[axis].set_linewidth(2)
    
plot.scatter(x,time_values/ref_time, label="My measuring time/ Niko's measuring time")
plot.grid()
plot.set_ylim(0.98,1)
plot.legend(fontsize=18, loc='lower left')
 
plot.set_xscale("log")       
plt.show()
figure.savefig("/Users/yasaman/AMS02/plots/flux/Timeratio.pdf" , dpi=250)
######### Measuring time 
figure = plt.figure(figsize=(12, 10))
plot = figure.subplots(1, 1)
plot.set_xlabel("Energy/Gev",fontsize = 26,fontweight = 'bold')
plot.set_ylabel("Measuring time/s", fontsize = 26,fontweight = 'bold')
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
plot.step(np.concatenate(([time_edge[0]], time_edge)), np.concatenate(([0], Time_values, [0])), where="post",linewidth=3.5, color='m')
plt.show()
figure.savefig("/Users/yasaman/AMS02/plots/flux/time.pdf" , dpi=250)





########## Acceptance
figure = plt.figure(figsize=(12, 10))
plot = figure.subplots(1, 1)
plot.set_xlabel("Energy/Gev",fontsize = 26,fontweight = 'bold')
plot.set_ylabel(r"$\rm A_{\rm eff}/(cm^2 \; sr)$", fontsize = 23,fontweight = 'bold')
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
plot.scatter(x , acceptance*1e4)     
#plot.step(np.concatenate(([time_edge[0]], time_edge)), np.concatenate(([0], acceptance, [0])), where="post",linewidth=3.5,color='m')
plt.show()
figure.savefig("/Users/yasaman/AMS02/plots/flux/acceptance.pdf" , dpi=250)

#######Scaled Flux
figure = plt.figure(figsize=(12, 10))
plot = figure.subplots(1, 1)
plot.set_xlabel("Energy/Gev",fontsize = 26,fontweight = 'bold')
plot.set_ylabel(r"$ \rm E^3 \; \Phi_{\rm e^-}/( GeV^{2} \; m^{-2} \; sr^{-1} \; s^{-1})$", fontsize = 26,fontweight = 'bold')
plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
for axis in ['top','bottom','left','right']:
    plot.spines[axis].set_linewidth(2)

plot.errorbar(x,(flux*x**3),(el_error*x**3),fmt='.',markersize=16,color='m',label='My flux')
#plot.errorbar(x[3:],((flux/correction)*x**3)[3:],((el_error/correction)*x**3)[3:],fmt='.',markersize=16,color='g',label='corrected flux')
plot.errorbar(ref_Energy,ref_flux*ref_Energy**3,ref_flux_error*ref_Energy**3,markersize=16,fmt='.',color='k',label="Niko's flux")
plt.legend(fontsize=25)
 
plot.set_xscale("log")       
plt.show()
figure.savefig("/Users/yasaman/AMS02/plots/flux/Scaled_flux.pdf" , dpi=250)




