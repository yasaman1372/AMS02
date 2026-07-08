#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:33:44 2023

@author: yasaman
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import SymLogNorm, LogNorm
from matplotlib.gridspec import GridSpec
from scipy.interpolate import UnivariateSpline
import os
# from scipy.signal import savgol_filter


#parameters_all = np.load("/Users/yasaman/AMS02/plots/template_fit/TRD-CCMVABDT/pass8/all/parameters_all.npz")
#parameters_all_err = np.load("/Users/yasaman/AMS02/plots/template_fit/TRD-CCMVABDT/pass8/all/errors_all.npz")
#parameters_st = np.load("/Users/yasaman/AMS02/plots/template_fit/TRD-CCMVABDT/pass8/single/parameters_single.npz") 
#parameters_mt = np.load("/Users/yasaman/AMS02/plots/template_fit/TRD-CCMVABDT/pass8/multiple/parameters_multiple.npz") 

os.makedirs("results/plots/electron/", exist_ok = True)
os.makedirs("results/plots/proton/", exist_ok = True)
os.makedirs("results/plots/ccproton/", exist_ok = True)

proton_parameters_st = np.load("/hpcwork/jara0052/yasaman/MyElectronAnalysis_cbk/1dTrdTemplateFitISS8Single/results/proton/proton_parameters_single.npz")
proton_parameters_mt = np.load("/hpcwork/jara0052/yasaman/MyElectronAnalysis_cbk/1dTrdTemplateFitISS8Multiple/results/proton/proton_parameters_multiple.npz")

electron_parameters_st = np.load("/hpcwork/jara0052/yasaman/MyElectronAnalysis_cbk/1dTrdTemplateFitISS8Single/results/electron/electron_parameters_single.npz")
electron_parameters_mt = np.load("/hpcwork/jara0052/yasaman/MyElectronAnalysis_cbk/1dTrdTemplateFitISS8Multiple/results/electron/electron_parameters_multiple.npz")

ccproton_parameters_st = np.load("/hpcwork/jara0052/yasaman/MyElectronAnalysis_cbk/1dTrdTemplateFitISS8Single/results/ccproton/ccproton_parameters_single.npz")
ccproton_parameters_mt = np.load("/hpcwork/jara0052/yasaman/MyElectronAnalysis_cbk/1dTrdTemplateFitISS8Multiple/results/ccproton/ccproton_parameters_multiple.npz")

proton_parameters_mt_smooth = dict(proton_parameters_mt.items())
proton_parameters_st_smooth = dict(proton_parameters_st.items())

electron_parameters_st_smooth = dict(electron_parameters_st.items())
electron_parameters_mt_smooth = dict(electron_parameters_mt.items())

ccproton_parameters_mt_smooth = dict(ccproton_parameters_st.items())
ccproton_parameters_st_smooth = dict(ccproton_parameters_mt.items())



Energy_binning = np.array([0.5,0.65,0.82,1.01,1.22,1.46,1.72,2.0,2.31,2.65,3.0,3.36,3.73,
             4.12,4.54,5.0,5.49,6.0,6.54,7.1,7.69,8.3,8.95,9.62,10.32,11.04,
             11.8,12.59,13.41,14.25,15.14,16.05,17.0,17.98,18.99,20.04,21.13,
             22.25,23.42,24.62,25.9,27.25,28.68,30.21,31.82,33.53,35.36,37.31,
             39.39,41.61,44.0,46.57,49.33,52.33,55.58,59.13,63.02,67.3,72.05,
             77.37,83.36,90.19,98.08,107.3,118.4,132.1,148.8,169.9,197.7,237.2,
             290.0,370.0,500.0,700.0,1000.0])


########### proton multi track
    
figure = plt.figure(figsize=(12, 10))
ax = figure.add_subplot(111)
ax.set_ylabel("m2_p", fontsize = 26)
ax.set_xlabel("Energy GeV", fontsize = 26)
ax.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
ax.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
    # ax1.rcParams['xtick.top'] = True
    # ax1.rcParams['ytick.right'] = True
ax.minorticks_on()
    # ax1.rcParams["font.weight"] = "bold"
    # ax1.rcParams["axes.labelweight"] = "bold"
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
         
ax.set_xscale("log")
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"  
     
x = (Energy_binning[1:]+Energy_binning[:-1])/2
spl = UnivariateSpline(x, proton_parameters_mt["m2_p"], s=0.001)
    
ax.plot(x,spl(x), color='r')
ax.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, proton_parameters_mt["m2_p"])

proton_parameters_mt_smooth["m2_p"] = spl(x)
    
figure.savefig("results/plots/proton/mt_m2_p.pdf" , dpi=250)
plt.close(figure)    
        
figure = plt.figure(figsize=(12, 10))
ax = figure.add_subplot(111)
ax.set_ylabel("s2_p", fontsize = 26)
ax.set_xlabel("Energy GeV", fontsize = 26)
ax.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
ax.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
    # ax1.rcParams['xtick.top'] = True
    # ax1.rcParams['ytick.right'] = True
ax.minorticks_on()
    # ax1.rcParams["font.weight"] = "bold"
    # ax1.rcParams["axes.labelweight"] = "bold"
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
         
ax.set_xscale("log")
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"  
     
x = (Energy_binning[1:]+Energy_binning[:-1])/2
spl = UnivariateSpline(x, proton_parameters_mt["s2_p"], s=0.0006)
    
ax.plot(x,spl(x), color='r')
ax.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, proton_parameters_mt["s2_p"])

proton_parameters_mt_smooth["s2_p"] = spl(x)
    
figure.savefig("results/plots/proton/mt_s2_p.pdf" , dpi=250)
plt.close(figure) 

figure = plt.figure(figsize=(12, 10))
ax = figure.add_subplot(111)
ax.set_ylabel("t_p", fontsize = 26)
ax.set_xlabel("Energy GeV", fontsize = 26)
ax.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
ax.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
    # ax1.rcParams['xtick.top'] = True
    # ax1.rcParams['ytick.right'] = True
ax.minorticks_on()
    # ax1.rcParams["font.weight"] = "bold"
    # ax1.rcParams["axes.labelweight"] = "bold"
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
         
ax.set_xscale("log")
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"  
     
x = (Energy_binning[1:]+Energy_binning[:-1])/2
spl = UnivariateSpline(x, proton_parameters_mt["t_p"], s=0.009)
proton_parameters_mt_smooth["t_p"] = spl(x)
  
ax.plot(x,spl(x), color='r')
ax.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, proton_parameters_mt["t_p"])
    
figure.savefig("results/plots/proton/mt_t_p.pdf" , dpi=250)
plt.close(figure)   


########Proton single track

figure = plt.figure(figsize=(12, 10))
ax = figure.add_subplot(111)
ax.set_ylabel("m2_p", fontsize = 26)
ax.set_xlabel("Energy GeV", fontsize = 26)
ax.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
ax.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
    # ax1.rcParams['xtick.top'] = True
    # ax1.rcParams['ytick.right'] = True
ax.minorticks_on()
    # ax1.rcParams["font.weight"] = "bold"
    # ax1.rcParams["axes.labelweight"] = "bold"
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
         
ax.set_xscale("log")
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"  
     
x = (Energy_binning[1:]+Energy_binning[:-1])/2
spl = UnivariateSpline(x, proton_parameters_st["m2_p"], s=0.001)
proton_parameters_st_smooth["m2_p"] = spl(x)
    
ax.plot(x,spl(x), color='r')
ax.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, proton_parameters_st["m2_p"])
    
figure.savefig("results/plots/proton/st_m2_p.pdf" , dpi=250)
plt.close(figure)    
        
figure = plt.figure(figsize=(12, 10))
ax = figure.add_subplot(111)
ax.set_ylabel("s2_p", fontsize = 26)
ax.set_xlabel("Energy GeV", fontsize = 26)
ax.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
ax.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
    # ax1.rcParams['xtick.top'] = True
    # ax1.rcParams['ytick.right'] = True
ax.minorticks_on()
    # ax1.rcParams["font.weight"] = "bold"
    # ax1.rcParams["axes.labelweight"] = "bold"
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
         
ax.set_xscale("log")
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"  
     
x = (Energy_binning[1:]+Energy_binning[:-1])/2
spl = UnivariateSpline(x, proton_parameters_st["s2_p"], s=0.0006)
proton_parameters_st_smooth["s2_p"] = spl(x)
    
ax.plot(x,spl(x), color='r')
ax.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, proton_parameters_st["s2_p"])
    
figure.savefig("results/plots/proton/st_s2_p.pdf" , dpi=250)
plt.close(figure) 

figure = plt.figure(figsize=(12, 10))
ax = figure.add_subplot(111)
ax.set_ylabel("t_p", fontsize = 26)
ax.set_xlabel("Energy GeV", fontsize = 26)
ax.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
ax.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
    # ax1.rcParams['xtick.top'] = True
    # ax1.rcParams['ytick.right'] = True
ax.minorticks_on()
    # ax1.rcParams["font.weight"] = "bold"
    # ax1.rcParams["axes.labelweight"] = "bold"
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
         
ax.set_xscale("log")
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"  
     
x = (Energy_binning[1:]+Energy_binning[:-1])/2
spl = UnivariateSpline(x, proton_parameters_st["t_p"], s=0.003)
proton_parameters_st_smooth["t_p"] = spl(x)
    
ax.plot(x,spl(x), color='r')
ax.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, proton_parameters_st["t_p"])
    
figure.savefig("results/plots/proton/st_t_p.pdf" , dpi=250)
plt.close(figure)    

################  Electron Single Track

figure = plt.figure(figsize=(12, 10))
ax = figure.add_subplot(111)
ax.set_ylabel("m2_el", fontsize = 26)
ax.set_xlabel("Energy GeV", fontsize = 26)
ax.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
ax.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
    # ax1.rcParams['xtick.top'] = True
    # ax1.rcParams['ytick.right'] = True
ax.minorticks_on()
    # ax1.rcParams["font.weight"] = "bold"
    # ax1.rcParams["axes.labelweight"] = "bold"
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
         
ax.set_xscale("log")
        
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"  
     
x = (Energy_binning[1:]+Energy_binning[:-1])/2
spl = UnivariateSpline(x, electron_parameters_st["m2_el"], s=0.0004)
electron_parameters_st_smooth["m2_el"]=spl(x)
    
ax.plot(x,spl(x), color='r')
ax.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, electron_parameters_st["m2_el"])
    
figure.savefig("results/plots/electron/st_m2_el.pdf" , dpi=250)
plt.close(figure)        


figure = plt.figure(figsize=(12, 10))
ax = figure.add_subplot(111)
ax.set_ylabel("t_el", fontsize = 26)
ax.set_xlabel("Energy GeV", fontsize = 26)
ax.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
ax.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
    # ax1.rcParams['xtick.top'] = True
    # ax1.rcParams['ytick.right'] = True
ax.minorticks_on()
    # ax1.rcParams["font.weight"] = "bold"
    # ax1.rcParams["axes.labelweight"] = "bold"
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
         
ax.set_xscale("log")
        
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"  
     
x = (Energy_binning[1:]+Energy_binning[:-1])/2
spl = UnivariateSpline(x, electron_parameters_st["t_el"], s=0.007)
electron_parameters_st_smooth["t_el"]=spl(x)
    
ax.plot(x,spl(x), color='r')
ax.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, electron_parameters_st["t_el"])
    
figure.savefig("results/plots/electron/st_t_el.pdf" , dpi=250)
plt.close(figure) 


figure = plt.figure(figsize=(12, 10))
ax = figure.add_subplot(111)
ax.set_ylabel("s2_el", fontsize = 26)
ax.set_xlabel("Energy GeV", fontsize = 26)
ax.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
ax.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
    # ax1.rcParams['xtick.top'] = True
    # ax1.rcParams['ytick.right'] = True
ax.minorticks_on()
    # ax1.rcParams["font.weight"] = "bold"
    # ax1.rcParams["axes.labelweight"] = "bold"
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
         
ax.set_xscale("log")
        
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"  
     
x = (Energy_binning[1:]+Energy_binning[:-1])/2
spl = UnivariateSpline(x, electron_parameters_st["s2_el"], s=0.0001)
electron_parameters_st_smooth["s2_el"]=spl(x)
    
ax.plot(x,spl(x), color='r')
ax.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, electron_parameters_st["s2_el"])
    
figure.savefig("results/plots/electron/st_s2_el.pdf" , dpi=250)
plt.close(figure) 

############## Electron multiple track

figure = plt.figure(figsize=(12, 10))
ax = figure.add_subplot(111)
ax.set_ylabel("m2_el", fontsize = 26)
ax.set_xlabel("Energy GeV", fontsize = 26)
ax.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
ax.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
    # ax1.rcParams['xtick.top'] = True
    # ax1.rcParams['ytick.right'] = True
ax.minorticks_on()
    # ax1.rcParams["font.weight"] = "bold"
    # ax1.rcParams["axes.labelweight"] = "bold"
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
         
ax.set_xscale("log")
        
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"  
     
x = (Energy_binning[1:]+Energy_binning[:-1])/2
spl = UnivariateSpline(x, electron_parameters_mt["m2_el"], s=0.0001)
electron_parameters_mt_smooth["m2_el"]=spl(x)
    
ax.plot(x,spl(x), color='r')
ax.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, electron_parameters_mt["m2_el"])
    
figure.savefig("results/plots/electron/mt_m2_el.pdf" , dpi=250)

plt.close(figure)        


figure = plt.figure(figsize=(12, 10))
ax = figure.add_subplot(111)
ax.set_ylabel("t_el", fontsize = 26)
ax.set_xlabel("Energy GeV", fontsize = 26)
ax.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
ax.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
    # ax1.rcParams['xtick.top'] = True
    # ax1.rcParams['ytick.right'] = True
ax.minorticks_on()
    # ax1.rcParams["font.weight"] = "bold"
    # ax1.rcParams["axes.labelweight"] = "bold"
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
         
ax.set_xscale("log")
        
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"  
     
x = (Energy_binning[1:]+Energy_binning[:-1])/2
spl = UnivariateSpline(x, electron_parameters_mt["t_el"], s=0.004)
electron_parameters_mt_smooth["t_el"]=spl(x)
    
ax.plot(x,spl(x), color='r')
ax.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, electron_parameters_mt["t_el"])
    
figure.savefig("results/plots/electron/mt_t_el.pdf" , dpi=250)
plt.close(figure) 


figure = plt.figure(figsize=(12, 10))
ax = figure.add_subplot(111)
ax.set_ylabel("s2_el", fontsize = 26)
ax.set_xlabel("Energy GeV", fontsize = 26)
ax.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
ax.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
    # ax1.rcParams['xtick.top'] = True
    # ax1.rcParams['ytick.right'] = True
ax.minorticks_on()
    # ax1.rcParams["font.weight"] = "bold"
    # ax1.rcParams["axes.labelweight"] = "bold"
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
         
ax.set_xscale("log")
        
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"  
     
x = (Energy_binning[1:]+Energy_binning[:-1])/2
spl = UnivariateSpline(x, electron_parameters_mt["s2_el"], s=0.001)
electron_parameters_mt_smooth["s2_el"]=spl(x)
    
ax.plot(x,spl(x), color='r')
ax.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, electron_parameters_mt["s2_el"])
    
figure.savefig("results/plots/electron/mt_s2_el.pdf" , dpi=250)
plt.close(figure) 

#####################CCproton single track            

figure = plt.figure(figsize=(12, 10))
ax = figure.add_subplot(111)
ax.set_ylabel("delta_m_ccp", fontsize = 26)
ax.set_xlabel("Energy GeV", fontsize = 26)
ax.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
ax.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
    # ax1.rcParams['xtick.top'] = True
    # ax1.rcParams['ytick.right'] = Trueax.minorticks_on()
    # ax1.rcParams["font.weight"] = "bold"
    # ax1.rcParams["axes.labelweight"] = "bold"
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
         
ax.set_xscale("log")
        
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"  
     
x = (Energy_binning[1:]+Energy_binning[:-1])/2
spl = UnivariateSpline(x, ccproton_parameters_st["delta_m_ccp"], s=0.01)
ccproton_parameters_st_smooth['delta_m_ccp']=spl(x)
    
ax.plot(x,spl(x), color='r')
ax.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, ccproton_parameters_st["delta_m_ccp"])
    
figure.savefig("results/plots/ccproton/st_delta_m_ccp.pdf" , dpi=250)
plt.close(figure) 


figure = plt.figure(figsize=(12, 10))
ax = figure.add_subplot(111)
ax.set_ylabel("s_ccp", fontsize = 26)
ax.set_xlabel("Energy GeV", fontsize = 26)
ax.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
ax.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
    # ax1.rcParams['xtick.top'] = True
    # ax1.rcParams['ytick.right'] = Trueax.minorticks_on()
    # ax1.rcParams["font.weight"] = "bold"
    # ax1.rcParams["axes.labelweight"] = "bold"
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
         
ax.set_xscale("log")
        
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"  
     
x = (Energy_binning[1:]+Energy_binning[:-1])/2
spl = UnivariateSpline(x, ccproton_parameters_st["s_ccp"], s=0.0003)
ccproton_parameters_st_smooth['s_ccp']=spl(x)
    
ax.plot(x,spl(x), color='r')
ax.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, ccproton_parameters_st["s_ccp"])
    
figure.savefig("results/plots/ccproton/st_s_ccp.pdf" , dpi=250)
plt.close(figure)  

figure = plt.figure(figsize=(12, 10))
ax = figure.add_subplot(111)
ax.set_ylabel("t_ccp", fontsize = 26)
ax.set_xlabel("Energy GeV", fontsize = 26)
ax.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
ax.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
    # ax1.rcParams['xtick.top'] = True
    # ax1.rcParams['ytick.right'] = Trueax.minorticks_on()
    # ax1.rcParams["font.weight"] = "bold"
    # ax1.rcParams["axes.labelweight"] = "bold"
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
         
ax.set_xscale("log")
        
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"  
     
x = (Energy_binning[1:]+Energy_binning[:-1])/2
spl = UnivariateSpline(x, ccproton_parameters_st["t_ccp"], s=0.05)
ccproton_parameters_st_smooth['t_ccp']=spl(x)
    
ax.plot(x,spl(x), color='r')
ax.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, ccproton_parameters_st["t_ccp"])
    
figure.savefig("results/plots/ccproton/st_t_ccp.pdf" , dpi=250)
plt.close(figure) 
    
###############CCproton multi track    

figure = plt.figure(figsize=(12, 10))
ax = figure.add_subplot(111)
ax.set_ylabel("delta_m_ccp", fontsize = 26)
ax.set_xlabel("Energy GeV", fontsize = 26)
ax.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
ax.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
    # ax1.rcParams['xtick.top'] = True
    # ax1.rcParams['ytick.right'] = Trueax.minorticks_on()
    # ax1.rcParams["font.weight"] = "bold"
    # ax1.rcParams["axes.labelweight"] = "bold"
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
         
ax.set_xscale("log")
        
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"  
     
x = (Energy_binning[1:]+Energy_binning[:-1])/2
spl = UnivariateSpline(x, ccproton_parameters_mt["delta_m_ccp"], s=0.003)
ccproton_parameters_mt_smooth['delta_m_ccp']=spl(x)
    
ax.plot(x,spl(x), color='r')
ax.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, ccproton_parameters_mt["delta_m_ccp"])
    
figure.savefig("results/plots/ccproton/mt_delta_m_ccp.pdf" , dpi=250)
plt.close(figure) 



figure = plt.figure(figsize=(12, 10))
ax = figure.add_subplot(111)
ax.set_ylabel("s_ccp", fontsize = 26)
ax.set_xlabel("Energy GeV", fontsize = 26)
ax.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
ax.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
    # ax1.rcParams['xtick.top'] = True
    # ax1.rcParams['ytick.right'] = Trueax.minorticks_on()
    # ax1.rcParams["font.weight"] = "bold"
    # ax1.rcParams["axes.labelweight"] = "bold"
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
         
ax.set_xscale("log")
        
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"  
     
x = (Energy_binning[1:]+Energy_binning[:-1])/2
spl = UnivariateSpline(x, ccproton_parameters_mt["s_ccp"], s=0.0003)
ccproton_parameters_mt_smooth['s_ccp']=spl(x)
    
ax.plot(x,spl(x), color='r')
ax.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, ccproton_parameters_mt["s_ccp"])
    
figure.savefig("results/plots/ccproton/mt_s_ccp.pdf" , dpi=250)
plt.close(figure)  


figure = plt.figure(figsize=(12, 10))
ax = figure.add_subplot(111)
ax.set_ylabel("t_ccp", fontsize = 26)
ax.set_xlabel("Energy GeV", fontsize = 26)
ax.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
ax.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
    # ax1.rcParams['xtick.top'] = True
    # ax1.rcParams['ytick.right'] = Trueax.minorticks_on()
    # ax1.rcParams["font.weight"] = "bold"
    # ax1.rcParams["axes.labelweight"] = "bold"
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
         
ax.set_xscale("log")
        
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"  
     
x = (Energy_binning[1:]+Energy_binning[:-1])/2
spl = UnivariateSpline(x, ccproton_parameters_mt["t_ccp"], s=0.05)
ccproton_parameters_st_smooth['t_ccp']=spl(x)
    
ax.plot(x,spl(x), color='r')
ax.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, ccproton_parameters_mt["t_ccp"])
    
figure.savefig("results/plots/ccproton/mt_t_ccp.pdf" , dpi=250)
plt.close(figure)



np.savez("results/proton_parameters_multiple_smooth.npz", **proton_parameters_mt_smooth)
np.savez("results/electron_parameters_multiple_smooth.npz", **electron_parameters_mt_smooth)
np.savez("results/ccproton_parameters_multiple_smooth.npz", **ccproton_parameters_mt_smooth)

np.savez("results/proton_parameters_single_smooth.npz", **proton_parameters_st_smooth)
np.savez("results/electron_parameters_single_smooth.npz", **electron_parameters_st_smooth)
np.savez("results/ccproton_parameters_single_smooth.npz", **ccproton_parameters_st_smooth)

np.savez("results/proton_parameters_multiple.npz", **proton_parameters_mt)
np.savez("results/electron_parameters_multiple.npz", **electron_parameters_mt)
np.savez("results/ccproton_parameters_multiple.npz", **ccproton_parameters_mt)

np.savez("results/proton_parameters_single.npz", **proton_parameters_st)
np.savez("results/electron_parameters_single.npz", **electron_parameters_st)
np.savez("results/ccproton_parameters_single.npz", **ccproton_parameters_st)



