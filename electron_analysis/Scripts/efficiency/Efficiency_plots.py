#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 15:47:35 2023

@author: yasaman
"""

import multiprocessing as mp
import os
import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
from scipy.signal import savgol_filter
from scipy import interpolate


   

def lafferty_whyatt(edges, gamma):
     ex = 1 - gamma
     rmin = edges[:-1]
     rmax = edges[1:]
     return ((rmax - rmin) * ex / (rmax**ex - rmin**ex))**(1 / gamma)
 
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataversion", default="pass6", help="version of the data set (e.g. pass6)")
    parser.add_argument("--datatype", default="ISS", help="title of the input data (e.g. ISS)")
    parser.add_argument("--tracknumber", default="single", help="number of tracks (single or multiple)")
    parser.add_argument("--filepathMC", default= '/Users/yasaman/AMS02/data/efficiency/Efficiency_corrections_MC.npz',help="")
    parser.add_argument("--filepathISS", default= '/Users/yasaman/AMS02/data/efficiency/Efficiency_corrections_ISS.npz',help="") 
    parser.add_argument("--filepath_measuringtime", default='/Users/yasaman/AMS02/data/time/MeasuringTime_00000.root',help="")
    args = parser.parse_args()

    with uproot.open(args.filepath_measuringtime) as file:
        time_values , time_edge = file['MeasuringTime/fIntegratedMeasuringTimeOverCutOff'].to_numpy()
    
    weight_correction = time_values/np.max(time_values)     
        
    with np.load('/Users/yasaman/AMS02/data/efficiency/Efficiency_corrections_uppertof_ISS.npz') as result_file:
        TCEvents_ISS = result_file["TCEvents"].sum(axis=0)
        TCEvents_ISS_TRD = result_file["TCEvents"].sum(axis=1)
        TCEvents_ISS_passed = result_file["TCEvents_passed"].sum(axis=0)
        TCEvents_ISS_passed_TRD = result_file["TCEvents_passed"].sum(axis=1)
        ISS_TOF_Events = result_file["Events"]
        TOF_charge_binning = result_file["TOF_charge_binning"]
        
    with np.load('/Users/yasaman/AMS02/data/efficiency/Efficiency_corrections_uppertof_MC.npz') as result_file:  
        MC_TOF_Events = weight_correction * result_file["Events"]
        
    
    with np.load(os.path.join(args.filepathMC)) as result_file:
        TRD_binning = result_file["TRD_Estimator_binning"]
        Energy_binning = result_file["Energy_binning"]
        UTTEvents_MC = (weight_correction * result_file["UTTEvents"]).sum(axis=0)     #At least one useful TRD track
        UTCCEvents_MC = (weight_correction * result_file["UTCCEvents"]).sum(axis=0)   #At least one useful TOF cluster combination
        TCEvents_MC = (weight_correction * result_file["TCEvents"]).sum(axis=0)       #Upper TOF charge
        EALEvents_MC = (weight_correction * result_file["EALEvents"]).sum(axis=0)     #Enough active layers in TRD
        TTGEvents_MC = (weight_correction * result_file["TTGEvents"]).sum(axis=0)     #Tracker track goodness-of-fit in Y-projection
        ERMEvents_MC = (weight_correction * result_file["ERMEvents"]).sum(axis=0)     #Energy rigidity matching
        TEMEvents_MC = (weight_correction * result_file["TEMEvents"]).sum(axis=0)     #Tracker ECAL matching in X-projection
        
        UTTEvents_MC_passed = (weight_correction * result_file["UTTEvents_passed"]).sum(axis=0)     #At least one useful TRD track
        UTCCEvents_MC_passed = (weight_correction * result_file["UTCCEvents_passed"]).sum(axis=0)   #At least one useful TOF cluster combination
        TCEvents_MC_passed = (weight_correction * result_file["TCEvents_passed"]).sum(axis=0)       #Upper TOF charge
        EALEvents_MC_passed = (weight_correction * result_file["EALEvents_passed"]).sum(axis=0)     #Enough active layers in TRD
        TTGEvents_MC_passed = (weight_correction * result_file["TTGEvents_passed"]).sum(axis=0)     #Tracker track goodness-of-fit in Y-projection
        ERMEvents_MC_passed = (weight_correction * result_file["ERMEvents_passed"]).sum(axis=0)     #Energy rigidity matching
        TEMEvents_MC_passed = (weight_correction * result_file["TEMEvents_passed"]).sum(axis=0)     #Tracker ECAL matching in X-projection
        

        UTTEvents_MC_TRD = (weight_correction * result_file["UTTEvents"]).sum(axis=1)     #At least one useful TRD track
        UTCCEvents_MC_TRD = (weight_correction * result_file["UTCCEvents"]).sum(axis=1)   #At least one useful TOF cluster combination
        TCEvents_MC_TRD = (weight_correction * result_file["TCEvents"]).sum(axis=1)       #Upper TOF charge
        EALEvents_MC_TRD = (weight_correction * result_file["EALEvents"]).sum(axis=1)     #Enough active layers in TRD
        TTGEvents_MC_TRD = (weight_correction * result_file["TTGEvents"]).sum(axis=1)     #Tracker track goodness-of-fit in Y-projection
        ERMEvents_MC_TRD = (weight_correction * result_file["ERMEvents"]).sum(axis=1)     #Energy rigidity matching
        TEMEvents_MC_TRD = (weight_correction * result_file["TEMEvents"]).sum(axis=1)     #Tracker ECAL matching in X-projection
        
        UTTEvents_MC_passed_TRD = (weight_correction * result_file["UTTEvents_passed"]).sum(axis=1)     #At least one useful TRD track
        UTCCEvents_MC_passed_TRD = (weight_correction * result_file["UTCCEvents_passed"]).sum(axis=1)   #At least one useful TOF cluster combination
        TCEvents_MC_passed_TRD = (weight_correction * result_file["TCEvents_passed"]).sum(axis=1)       #Upper TOF charge
        EALEvents_MC_passed_TRD = (weight_correction * result_file["EALEvents_passed"]).sum(axis=1)     #Enough active layers in TRD
        TTGEvents_MC_passed_TRD = (weight_correction * result_file["TTGEvents_passed"]).sum(axis=1)     #Tracker track goodness-of-fit in Y-projection
        ERMEvents_MC_passed_TRD = (weight_correction * result_file["ERMEvents_passed"]).sum(axis=1)     #Energy rigidity matching
        TEMEvents_MC_passed_TRD = (weight_correction * result_file["TEMEvents_passed"]).sum(axis=1)     #Tracker ECAL matching in X-projection
        
    
    
    with np.load(os.path.join(args.filepathISS)) as result_file:
        UTTEvents_ISS = result_file["UTTEvents"].sum(axis=0)     #At least one useful TRD track
        UTCCEvents_ISS = result_file["UTCCEvents"].sum(axis=0)   #At least one useful TOF cluster combination
        #TCEvents_ISS = result_file["TCEvents"].sum(axis=0)       #Upper TOF charge
        EALEvents_ISS = result_file["EALEvents"].sum(axis=0)     #Enough active layers in TRD
        TTGEvents_ISS = result_file["TTGEvents"].sum(axis=0)     #Tracker track goodness-of-fit in Y-projection
        ERMEvents_ISS = result_file["ERMEvents"].sum(axis=0)     #Energy rigidity matching
        TEMEvents_ISS = result_file["TEMEvents"].sum(axis=0)     #Tracker ECAL matching in X-projection
        
        UTTEvents_ISS_passed = result_file["UTTEvents_passed"].sum(axis=0)     #At least one useful TRD track
        UTCCEvents_ISS_passed = result_file["UTCCEvents_passed"].sum(axis=0)   #At least one useful TOF cluster combination
        #TCEvents_ISS_passed = result_file["TCEvents_passed"].sum(axis=0)       #Upper TOF charge
        EALEvents_ISS_passed = result_file["EALEvents_passed"].sum(axis=0)     #Enough active layers in TRD
        TTGEvents_ISS_passed = result_file["TTGEvents_passed"].sum(axis=0)     #Tracker track goodness-of-fit in Y-projection
        ERMEvents_ISS_passed = result_file["ERMEvents_passed"].sum(axis=0)     #Energy rigidity matching
        TEMEvents_ISS_passed = result_file["TEMEvents_passed"].sum(axis=0)     #Tracker ECAL matching in X-projection  
        
        
        UTTEvents_ISS_TRD = result_file["UTTEvents"].sum(axis=1)     #At least one useful TRD track
        UTCCEvents_ISS_TRD = result_file["UTCCEvents"].sum(axis=1)   #At least one useful TOF cluster combination
        #TCEvents_ISS_TRD = result_file["TCEvents"].sum(axis=1)       #Upper TOF charge
        EALEvents_ISS_TRD = result_file["EALEvents"].sum(axis=1)     #Enough active layers in TRD
        TTGEvents_ISS_TRD = result_file["TTGEvents"].sum(axis=1)     #Tracker track goodness-of-fit in Y-projection
        ERMEvents_ISS_TRD = result_file["ERMEvents"].sum(axis=1)     #Energy rigidity matching
        TEMEvents_ISS_TRD = result_file["TEMEvents"].sum(axis=1)     #Tracker ECAL matching in X-projection
        
        UTTEvents_ISS_passed_TRD = result_file["UTTEvents_passed"].sum(axis=1)     #At least one useful TRD track
        UTCCEvents_ISS_passed_TRD = result_file["UTCCEvents_passed"].sum(axis=1)   #At least one useful TOF cluster combination
        #TCEvents_ISS_passed_TRD = result_file["TCEvents_passed"].sum(axis=1)       #Upper TOF charge
        EALEvents_ISS_passed_TRD = result_file["EALEvents_passed"].sum(axis=1)     #Enough active layers in TRD
        TTGEvents_ISS_passed_TRD = result_file["TTGEvents_passed"].sum(axis=1)     #Tracker track goodness-of-fit in Y-projection
        ERMEvents_ISS_passed_TRD = result_file["ERMEvents_passed"].sum(axis=1)     #Energy rigidity matching
        TEMEvents_ISS_passed_TRD = result_file["TEMEvents_passed"].sum(axis=1)     #Tracker ECAL matching in X-projection  
        
    
        
        UTT_eff_MC = UTTEvents_MC_passed/UTTEvents_MC  
        UTCC_eff_MC = UTCCEvents_MC_passed/UTCCEvents_MC
        TC_eff_MC = TCEvents_MC_passed/TCEvents_MC
        EAL_eff_MC = EALEvents_MC_passed/EALEvents_MC
        TTG_eff_MC = TTGEvents_MC_passed/TTGEvents_MC
        ERM_eff_MC = ERMEvents_MC_passed/ERMEvents_MC
        TEM_eff_MC = TEMEvents_MC_passed/TEMEvents_MC
        
        UTT_eff_ISS = UTTEvents_ISS_passed/UTTEvents_ISS 
        UTCC_eff_ISS = UTCCEvents_ISS_passed/UTCCEvents_ISS
        TC_eff_ISS = TCEvents_ISS_passed/TCEvents_ISS
        EAL_eff_ISS = EALEvents_ISS_passed/EALEvents_ISS
        TTG_eff_ISS = TTGEvents_ISS_passed/TTGEvents_ISS
        ERM_eff_ISS = ERMEvents_ISS_passed/ERMEvents_ISS
        TEM_eff_ISS = TEMEvents_ISS_passed/TEMEvents_ISS
        
    
   
 #######################################################################       
    #####At least one useful TRD track    


    #smoothing parameters
    
    # Window size of the filter (odd number)
    window_size = 50

    # Polynomial order of the filter (usually smaller than the window size)
    poly_order = 2
    
    #the number of last data points that are cutted away
    cut = 1
    
    
    figure = plt.figure(figsize=(12, 10))
    ax1 = figure.add_subplot(211)

    ax1.set_ylabel("efficiency", fontsize = 15, fontweight = 'bold')
    ax1.set_xscale("log")
    

    ax1.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
    ax1.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)
        

    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2) 
        
    ax1.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, UTT_eff_MC, marker ="s", label ="MC")  
    ax1.scatter((Energy_binning[1:-cut]+Energy_binning[:-cut-1])/2, UTT_eff_ISS[:-cut],marker ="s", label ="ISS") 
    ax1.set_title("At least one useful TRD track")
    ax1.legend(fontsize=15,frameon=True,loc='lower left')
    
    
    ax2 = figure.add_subplot(212,sharex=ax1) 
    ax2.set_ylabel("ISS Eff/ MC Eff", fontsize = 15, fontweight = 'bold')
    ax2.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
    ax2.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)         
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(2)
    
    UTT_eff_ratio = UTT_eff_ISS/UTT_eff_MC    
    ax2.scatter((Energy_binning[1:-cut]+Energy_binning[:-1-cut])/2, UTT_eff_ratio[:-cut], marker ="o")
    
    smoothed_eff_ratio = savgol_filter(UTT_eff_ISS[:-cut]/UTT_eff_MC[:-cut], window_size, poly_order)
    extended_eff_ratio = interpolate.interp1d((Energy_binning[1:-cut]+Energy_binning[:-cut-1])/2, smoothed_eff_ratio, fill_value='extrapolate')
    UTT_eff_ratio_smoothed = extended_eff_ratio((Energy_binning[1:]+Energy_binning[:-1])/2)
    ax2.plot((Energy_binning[1:]+Energy_binning[:-1])/2,UTT_eff_ratio_smoothed, color='r')
        
    # Adjust the size of the upper subplot to be larger than the lower one
    ax1.set_position([0.1, 0.4, 0.8, 0.5])
    ax2.set_position([0.1, 0.1, 0.8, 0.3])
    
    ax1.set_ylim(0,1.4)
    ax2.set_ylim(0.94,1.028)
    ax2.hlines(1,min((Energy_binning[1:]+Energy_binning[:-1])/2), max((Energy_binning[1:]+Energy_binning[:-1])/2), linestyles='--', color="k")
    ax2.set_xlabel("Ecal Energy / GeV", fontsize=15)
        
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
        
    figure.savefig("/Users/yasaman/AMS02/plots/Efficiency/UTT_eff.pdf" , dpi=250)
    plt.close(figure)    
    
    ######################################################################################
    ####At least one useful TOF cluster combination
    
    #smoothing parameters
    
    # Window size of the filter (odd number)
    window_size = 50

    # Polynomial order of the filter (usually smaller than the window size)
    poly_order = 2
    
    #the number of last data points that are cutted away
    cut = 1
    

    figure = plt.figure(figsize=(12, 10))
    ax1 = figure.add_subplot(211)

    ax1.set_ylabel("efficiency", fontsize = 15, fontweight = 'bold')
    ax1.set_xscale("log")
    

    ax1.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
    ax1.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)
        

    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2) 
        
    ax1.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, UTCC_eff_MC, marker= 's', label ="MC")  
    ax1.scatter((Energy_binning[1:-cut]+Energy_binning[:-cut-1])/2, UTCC_eff_ISS[:-cut], marker= 's', label ="ISS")
    ax1.set_title('At least one useful TOF cluster combination')
    ax1.legend(fontsize=15,frameon=True,loc='lower left')
    
    ax2 = figure.add_subplot(212,sharex=ax1) 
    ax2.set_ylabel("ISS Eff/ MC Eff", fontsize = 15, fontweight = 'bold')
    ax2.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
    ax2.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)         
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(2)
     
    UTCC_eff_ratio = UTCC_eff_ISS/UTCC_eff_MC   
    ax2.scatter((Energy_binning[1:-cut]+Energy_binning[:-cut-1])/2,UTCC_eff_ratio[:-cut], marker='o', color ='g')
    
    smoothed_eff_ratio = savgol_filter(UTCC_eff_ISS[:-cut]/UTCC_eff_MC[:-cut], window_size, poly_order)
    extended_eff_ratio = interpolate.interp1d((Energy_binning[1:-cut]+Energy_binning[:-cut-1])/2, smoothed_eff_ratio, fill_value='extrapolate')
    UTCC_eff_ratio_smoothed = extended_eff_ratio((Energy_binning[1:]+Energy_binning[:-1])/2)
    ax2.plot((Energy_binning[1:]+Energy_binning[:-1])/2,UTCC_eff_ratio_smoothed,color='r')
        
    # Adjust the size of the upper subplot to be larger than the lower one
    ax1.set_position([0.1, 0.4, 0.8, 0.5])
    ax2.set_position([0.1, 0.1, 0.8, 0.3])   
    
    ax1.set_ylim(0,1.4)
    ax2.set_ylim(0.94,1.028)
    ax2.hlines(1,min((Energy_binning[1:]+Energy_binning[:-1])/2), max((Energy_binning[1:]+Energy_binning[:-1])/2), linestyles='--',color='k')
    ax2.set_xlabel("Ecal Energy / GeV", fontsize=15)
        
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
        
    figure.savefig("/Users/yasaman/AMS02/plots/Efficiency/UTCC_eff.pdf" , dpi=250)
    plt.close(figure)   
    
################################################################################################
    #Upper TOF charge
    
    #smoothing parameters
    
    # Window size of the filter (odd number)
    window_size = 50

    # Polynomial order of the filter (usually smaller than the window size)
    poly_order = 2
    
    #the number of last data points that are cutted away
    cut = 1
    

    figure = plt.figure(figsize=(12, 10))
    ax1 = figure.add_subplot(211)

    ax1.set_ylabel("efficiency", fontsize = 15, fontweight = 'bold')
    ax1.set_xscale("log")
    

    ax1.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
    ax1.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)
        

    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2) 
        
    ax1.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, TC_eff_MC, marker="s", label ="MC")  
    ax1.scatter((Energy_binning[1:-cut]+Energy_binning[:-1-cut])/2, TC_eff_ISS[:-cut], marker="s", label ="ISS") 
    ax1.set_title('Upper TOF charge')
    ax1.legend(fontsize=15,frameon=True,loc='lower left')
    
    ax2 = figure.add_subplot(212,sharex=ax1) 
    ax2.set_ylabel("ISS Eff/ MC Eff", fontsize = 15, fontweight = 'bold')
    ax2.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
    ax2.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)         
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(2)
    
    TC_eff_ratio = TC_eff_ISS/TC_eff_MC 
    ax2.scatter((Energy_binning[1:-cut]+Energy_binning[:-1-cut])/2, TC_eff_ratio[:-cut], marker = "o",color='g')
    
    smoothed_eff_ratio = savgol_filter(TC_eff_ISS[:-cut]/TC_eff_MC[:-cut], window_size, poly_order)
    extended_eff_ratio = interpolate.interp1d((Energy_binning[1:-cut]+Energy_binning[:-cut-1])/2, smoothed_eff_ratio, fill_value='extrapolate')
    TC_eff_ratio_smoothed = extended_eff_ratio((Energy_binning[1:]+Energy_binning[:-1])/2)
    #ax2.plot((Energy_binning[1:]+Energy_binning[:-1])/2,TC_eff_ratio_smoothed, color='r')
        
        
    # Adjust the size of the upper subplot to be larger than the lower one
    ax1.set_position([0.1, 0.4, 0.8, 0.5])
    ax2.set_position([0.1, 0.1, 0.8, 0.3])  
    
    ax1.set_ylim(0,1.4)
    #ax2.set_ylim(0.94,1.028)
    ax2.hlines(1,min((Energy_binning[1:]+Energy_binning[:-1])/2), max((Energy_binning[1:]+Energy_binning[:-1])/2), linestyles='--',color='k')
    ax2.set_xlabel("Ecal Energy / GeV", fontsize=15)
        
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
        
    figure.savefig("/Users/yasaman/AMS02/plots/Efficiency/TC_eff.pdf" , dpi=250)
    plt.close(figure)  
########################################################################################
    #Enough active layers in TRD
    
    #smoothing parameters
    
    # Window size of the filter (odd number)
    window_size = 50

    # Polynomial order of the filter (usually smaller than the window size)
    poly_order = 2
    
    #the number of last data points that are cutted away
    cut = 1
    

    figure = plt.figure(figsize=(12, 10))
    ax1 = figure.add_subplot(211)

    ax1.set_ylabel("efficiency", fontsize = 15, fontweight = 'bold')
    ax1.set_xscale("log")
    

    ax1.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
    ax1.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)
        

    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2) 
        
    ax1.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, EAL_eff_MC, marker='s', label ="MC")  
    ax1.scatter((Energy_binning[1:-cut]+Energy_binning[:-cut-1])/2, EAL_eff_ISS[:-cut], marker='s', label ="ISS") 
    ax1.set_title('Enough active layers in TRD')
    ax1.legend(fontsize=15,frameon=True,loc='lower left')
    
    ax2 = figure.add_subplot(212,sharex=ax1) 
    ax2.set_ylabel("ISS Eff/ MC Eff", fontsize = 15, fontweight = 'bold')
    ax2.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
    ax2.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)         
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(2)
    
    EAL_eff_ratio = EAL_eff_ISS/EAL_eff_MC   
    ax2.scatter((Energy_binning[1:-cut]+Energy_binning[:-cut-1])/2, EAL_eff_ratio[:-cut], marker='o',color='g')
    
    smoothed_eff_ratio = savgol_filter(EAL_eff_ISS[:-cut]/EAL_eff_MC[:-cut], window_size, poly_order)
    extended_eff_ratio = interpolate.interp1d((Energy_binning[1:-cut]+Energy_binning[:-cut-1])/2, smoothed_eff_ratio, fill_value='extrapolate')
    EAL_eff_ratio_smoothed = extended_eff_ratio((Energy_binning[1:]+Energy_binning[:-1])/2)
    ax2.plot((Energy_binning[1:]+Energy_binning[:-1])/2,EAL_eff_ratio_smoothed, color='r')
        
        
    # Adjust the size of the upper subplot to be larger than the lower one
    ax1.set_position([0.1, 0.4, 0.8, 0.5])
    ax2.set_position([0.1, 0.1, 0.8, 0.3])   
    
    ax1.set_ylim(0,1.4)
    ax2.set_ylim(0.94,1.028)
    ax2.hlines(1,min((Energy_binning[1:]+Energy_binning[:-1])/2), max((Energy_binning[1:]+Energy_binning[:-1])/2), linestyles='--', color='k')
    ax2.set_xlabel("Ecal Energy / GeV", fontsize=15)
        
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
        
    figure.savefig("/Users/yasaman/AMS02/plots/Efficiency/EAL_eff.pdf" , dpi=250)
    plt.close(figure)   
#######################################################################################
    #Tracker track goodness-of-fit in Y-projection
    
    #smoothing parameters
    
    # Window size of the filter (odd number)
    window_size = 50

    # Polynomial order of the filter (usually smaller than the window size)
    poly_order = 2
    
    #the number of last data points that are cutted away
    cut = 1

    figure = plt.figure(figsize=(12, 10))
    ax1 = figure.add_subplot(211)

    ax1.set_ylabel("efficiency", fontsize = 15, fontweight = 'bold')
    ax1.set_xscale("log")
    

    ax1.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
    ax1.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)
        

    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2) 
        
    ax1.plot((Energy_binning[1:]+Energy_binning[:-1])/2, TTG_eff_MC, marker='x', label ="MC")  
    ax1.plot((Energy_binning[1:-2]+Energy_binning[:-3])/2, TTG_eff_ISS[:-2], marker='x',label ="ISS") 
    ax1.set_title('Tracker track goodness-of-fit in Y-projection')
    ax1.legend(fontsize=15,frameon=True,loc='lower left')
    
    ax2 = figure.add_subplot(212,sharex=ax1) 
    ax2.set_ylabel("ISS Eff/ MC Eff", fontsize = 15, fontweight = 'bold')
    ax2.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
    ax2.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)         
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(2)
     
    TTG_eff_ratio =  TTG_eff_ISS/TTG_eff_MC 
    ax2.scatter((Energy_binning[1:-cut]+Energy_binning[:-1-cut])/2, TTG_eff_ratio[:-cut], marker='s',color='g')
    
    smoothed_eff_ratio = savgol_filter(TTG_eff_ISS[:-cut]/TTG_eff_MC[:-cut], window_size, poly_order)
    extended_eff_ratio = interpolate.interp1d((Energy_binning[1:-cut]+Energy_binning[:-cut-1])/2, smoothed_eff_ratio, fill_value='extrapolate')
    TTG_eff_ratio_smoothed= extended_eff_ratio((Energy_binning[1:]+Energy_binning[:-1])/2)
    ax2.plot((Energy_binning[1:]+Energy_binning[:-1])/2,TTG_eff_ratio_smoothed, color='r')
        
    # Adjust the size of the upper subplot to be larger than the lower one
    ax1.set_position([0.1, 0.4, 0.8, 0.5])
    ax2.set_position([0.1, 0.1, 0.8, 0.3])  
    
    ax1.set_ylim(0,1.4)
    ax2.set_ylim(0.94,1.028)
    ax2.hlines(1,min((Energy_binning[1:]+Energy_binning[:-1])/2), max((Energy_binning[1:]+Energy_binning[:-1])/2), linestyles='--',color='k')
    ax2.set_xlabel("Ecal Energy / GeV", fontsize=15)
        
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
        
    figure.savefig("/Users/yasaman/AMS02/plots/Efficiency/TTG_eff.pdf" , dpi=250)
    plt.close(figure)    

##############################################################################################
    #Energy rigidity matching
    
    #smoothing parameters
    
    # Window size of the filter (odd number)
    window_size = 50

    # Polynomial order of the filter (usually smaller than the window size)
    poly_order = 2
    
    #the number of last data points that are cutted away
    cut = 1    
    

    figure = plt.figure(figsize=(12, 10))
    ax1 = figure.add_subplot(211)

    ax1.set_ylabel("efficiency", fontsize = 15, fontweight = 'bold')
    ax1.set_xscale("log")
    

    ax1.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
    ax1.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)
        

    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2) 
        
    ax1.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, ERM_eff_MC, marker='s',label ="MC")  
    ax1.scatter((Energy_binning[1:-cut]+Energy_binning[:-1-cut])/2, ERM_eff_ISS[:-cut],marker='s', label ="ISS") 
    ax1.set_title('Energy rigidity matching')
    ax1.legend(fontsize=15,frameon=True,loc='lower left')
    
    ax2 = figure.add_subplot(212,sharex=ax1) 
    ax2.set_ylabel("ISS Eff/ MC Eff", fontsize = 15, fontweight = 'bold')
    ax2.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
    ax2.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)         
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(2)
    
    ERM_eff_ratio = ERM_eff_ISS/ERM_eff_MC 
    ax2.scatter((Energy_binning[1:-cut]+Energy_binning[:-1-cut])/2,ERM_eff_ratio[:-cut],marker='s',color='g')
    
    
    smoothed_eff_ratio = savgol_filter(ERM_eff_ISS[:-cut]/ERM_eff_MC[:-cut], window_size, poly_order)
    extended_eff_ratio = interpolate.interp1d((Energy_binning[1:-cut]+Energy_binning[:-cut-1])/2, smoothed_eff_ratio, fill_value='extrapolate')
    ERM_eff_ratio_smoothed = extended_eff_ratio((Energy_binning[1:]+Energy_binning[:-1])/2)
    ax2.plot((Energy_binning[1:]+Energy_binning[:-1])/2,ERM_eff_ratio_smoothed, color='r')
        
    # Adjust the size of the upper subplot to be larger than the lower one
    ax1.set_position([0.1, 0.4, 0.8, 0.5])
    ax2.set_position([0.1, 0.1, 0.8, 0.3])  
    
    ax1.set_ylim(0,1.4)
    ax2.set_ylim(0.90,1.028)
    ax2.hlines(1,min((Energy_binning[1:]+Energy_binning[:-1])/2), max((Energy_binning[1:]+Energy_binning[:-1])/2), linestyles='--',color='k')
    ax2.set_xlabel("Ecal Energy / GeV", fontsize=15)
        
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
        
    figure.savefig("/Users/yasaman/AMS02/plots/Efficiency/ERM_eff.pdf" , dpi=250)
    plt.close(figure)   
    
#########################################################################################
    #Tracker ECAL matching in X-projection
    
    #smoothing parameters
    
    # Window size of the filter (odd number)
    window_size = 50

    # Polynomial order of the filter (usually smaller than the window size)
    poly_order = 2
    
    #the number of last data points that are cutted away
    cut = 1   

    figure = plt.figure(figsize=(12, 10))
    ax1 = figure.add_subplot(211)

    ax1.set_ylabel("efficiency", fontsize = 15, fontweight = 'bold')
    ax1.set_xscale("log")
    

    ax1.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
    ax1.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)
        

    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2) 
    
     
    ax1.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, TEM_eff_MC, marker='s',label ="MC")  
    ax1.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, TEM_eff_ISS, marker='s',label ="ISS") 
    ax1.set_title('Tracker ECAL matching in X-projection')
    ax1.legend(fontsize=15,frameon=True,loc='lower left')
    
    ax2 = figure.add_subplot(212,sharex=ax1) 
    ax2.set_ylabel("ISS Eff/ MC Eff", fontsize = 15, fontweight = 'bold')
    ax2.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
    ax2.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)         
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(2)
    
    TEM_eff_ratio =  TEM_eff_ISS/TEM_eff_MC   
    ax2.scatter((Energy_binning[1:-cut]+Energy_binning[:-1-cut])/2,TEM_eff_ratio[:-cut],marker='s',color='g')
    smoothed_eff_ratio = savgol_filter(TEM_eff_ISS[:-cut]/TEM_eff_MC[:-cut], window_size, poly_order)
    extended_eff_ratio = interpolate.interp1d((Energy_binning[1:-cut]+Energy_binning[:-cut-1])/2, smoothed_eff_ratio, fill_value='extrapolate')
    TEM_eff_ratio_smoothed = extended_eff_ratio((Energy_binning[1:]+Energy_binning[:-1])/2)
    ax2.plot((Energy_binning[1:]+Energy_binning[:-1])/2,TEM_eff_ratio_smoothed, color='r')
        
    # Adjust the size of the upper subplot to be larger than the lower one
    ax1.set_position([0.1, 0.4, 0.8, 0.5])
    ax2.set_position([0.1, 0.1, 0.8, 0.3])  
    
    ax1.set_ylim(0,1.4)
    ax2.set_ylim(0.94,1.028)
    ax2.hlines(1,min((Energy_binning[1:]+Energy_binning[:-1])/2), max((Energy_binning[1:]+Energy_binning[:-1])/2), linestyles='--', color='k')
    ax2.set_xlabel("Ecal Energy / GeV", fontsize=15)
        
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
        
    figure.savefig("/Users/yasaman/AMS02/plots/Efficiency/TEM_eff.pdf" , dpi=250)
    plt.close(figure) 

##############################################################################################

    ref_flux_file = '/Users/yasaman/AMS02/data/Niko/LeptonFluxes_Zimmermann_25_10_2018_2D.root'
    MC_momentum_file = '/Users/yasaman/AMS02/data/MCmomentumPass6/Electron_Mc_Momentum_Edependent_result_Test_good.npz' 
    Trigger_file = '/Users/yasaman/AMS02/data/TriggerPass6/Triggercount.root'
    ecal_eff_file_singletrack = "/Users/yasaman/AMS02/plots/template_fit/TRD/electron/eff_parameters_single.npz"
    ecal_eff_file_multitrack = "/Users/yasaman/AMS02/plots/template_fit/TRD/electron/eff_parameters_single.npz"
    all_track_sample_parameters_file = "/Users/yasaman/AMS02/plots/template_fit/TRD-CCMVABDT/all/parameters_all.npz"
    ref_file ='/Users/yasaman/AMS02/data/Niko/LeptonFluxes_Zimmermann_25_10_2018_2D.root'
    Time_file = '/Users/yasaman/AMS02/data/time/MeasuringTime_00000.root'
    
    ref_flux_file_published = '/Users/yasaman/AMS02/data/T2_AMS_rev_000.txt'
    file = np.loadtxt(ref_flux_file_published)
    published_flux_x = file[:,0]
    published_flux_y = file[:,3]
    # print(published_flux_x)
    
    file = np.load("/Users/yasaman/AMS02/plots/template_fit/TRD-CCMVABDT/all/parameters_all.npz") 
    error_file = np.load("/Users/yasaman/AMS02/plots/template_fit/TRD-CCMVABDT/all/errors_all.npz")
    el_number = file["nel"]  
    el_error = error_file["nel"]
    
    with uproot.open(Time_file) as file:
        time_values , time_edge = file['MeasuringTime/fIntegratedMeasuringTimeOverCutOff'].to_numpy() 

    with uproot.open(ref_file) as file:           
        ref_acceptance_el = file['SxFluxTools_Electron_RWTH/grElectron_RWTH_EffectiveAcceptance'].values()
        #print(ref_acceptance_el[0],ref_acceptance_el[1])

    with np.load(MC_momentum_file) as file:
        MC_binning = file["var1_binning"]
        MC_momentum_values = file["Events"] 
        MC_momentum = MC_momentum_values.sum(axis=0)
    
    with uproot.open(Trigger_file) as file:
        trigger_values , trigger_edge = file['TriggerCounts'].to_numpy()    

        
    file = np.load(ecal_eff_file_singletrack) 
    ecal_eff_single_track = file["e_sig"]
    
    file = np.load(ecal_eff_file_multitrack) 
    ecal_eff_multiple_track = file["e_sig"]
    
    file = np.load(all_track_sample_parameters_file)  
    alpha_el = file["alpha_el"]
    
    ecal_eff = alpha_el*ecal_eff_single_track + (1-alpha_el)*ecal_eff_multiple_track
    

    smoothed_eff_correction_without_EcalEff = TEM_eff_ratio_smoothed * ERM_eff_ratio_smoothed * TTG_eff_ratio_smoothed * EAL_eff_ratio_smoothed* TC_eff_ratio_smoothed * UTCC_eff_ratio_smoothed * UTT_eff_ratio_smoothed
    
    smoothed_eff_correction = TEM_eff_ratio_smoothed * ERM_eff_ratio_smoothed * TTG_eff_ratio_smoothed * EAL_eff_ratio_smoothed * TC_eff_ratio_smoothed * UTCC_eff_ratio_smoothed * UTT_eff_ratio_smoothed * ecal_eff
    
    eff_correction_without_EcalEff = TEM_eff_ratio * ERM_eff_ratio * TTG_eff_ratio * EAL_eff_ratio  * TC_eff_ratio * UTCC_eff_ratio * UTT_eff_ratio
    
    eff_correction = TEM_eff_ratio * ERM_eff_ratio * TTG_eff_ratio * EAL_eff_ratio  * TC_eff_ratio * UTCC_eff_ratio * UTT_eff_ratio * ecal_eff
    
    raw_acceptance = (3.9)**2 *np.pi*(MC_momentum/trigger_values) 
    acceptance_smoothed = raw_acceptance *  smoothed_eff_correction
    acceptance_smoothed_without_EcalEff = raw_acceptance *  smoothed_eff_correction_without_EcalEff
    acceptance = raw_acceptance * eff_correction
    acceptance_without_EcalEff = raw_acceptance *  eff_correction_without_EcalEff
    
    deltaE = Energy_binning[1:]-Energy_binning[:-1]
    Energy_bins= lafferty_whyatt(Energy_binning, 3)
    
    el_flux = el_number/(acceptance_smoothed*time_values*deltaE)    
    el_flux_err = el_error/(acceptance*time_values*deltaE*acceptance_smoothed_without_EcalEff)
    
#####################################################################################   
 
    figure = plt.figure(figsize=(12, 10))
    ax1 = figure.add_subplot(111)

    ax1.set_ylabel("acceptance", fontsize = 15, fontweight = 'bold')
    ax1.set_xscale("log")
    

    ax1.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
    ax1.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)
        

    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2) 
    
    ax1.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, raw_acceptance, label = "raw acceptance" )     
    # ax1.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, acceptance_smoothed, label = "smoothed acceptance" )   
    ax1.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, acceptance_smoothed_without_EcalEff, label= "smoothed acceptance without Ecal efficiency corrections" ) 
    # ax1.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, acceptance, label= "acceptance" )
    # ax1.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, acceptance_without_EcalEff, label= "acceptance without Ecal efficiency corrections" )
    ax1.scatter(ref_acceptance_el[0],ref_acceptance_el[1], label= "Niko's acceptance" )
    ax1.legend()
    
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
        
    figure.savefig("/Users/yasaman/AMS02/plots/Efficiency/acceptance_plot.pdf" , dpi=250)
    plt.close(figure) 
    
    
    
     
    figure = plt.figure(figsize=(12, 10))
    ax1 = figure.add_subplot(111)

    ax1.set_ylabel("scaled flux", fontsize = 15, fontweight = 'bold')
    ax1.set_xscale("log")
    

    ax1.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
    ax1.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)
        

    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2) 
    
    ax1.scatter(Energy_bins, el_flux*(Energy_bins**3) , label = "myflux" )     
    ax1.scatter(published_flux_x, published_flux_y*(published_flux_x**3), label = "published" ) 
    plt.legend()

    
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
        
    figure.savefig("/Users/yasaman/AMS02/plots/Efficiency/scaledflux_plot.pdf" , dpi=250)
    plt.close(figure)
    
    
######################################################################################    
    
    figure = plt.figure(figsize=(12, 10))
    plot = figure.subplots(1, 1)
    plot.set_xlabel("Energy/ GeV",fontsize = 26,fontweight = 'bold')
    plot.set_ylabel("Normalaized Events", fontsize = 26,fontweight = 'bold')
    plot.set_xscale("log")
    plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
    plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)
             
    UTTEvents_MC= UTTEvents_MC/np.sum(UTTEvents_MC)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], UTTEvents_MC, [0])), where="post",linewidth=2,color='b', label='total events MC')
    plt.fill_between(Energy_binning, np.concatenate(([0], UTTEvents_MC)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='b')
    
    UTTEvents_MC_passed= UTTEvents_MC_passed/np.sum(UTTEvents_MC_passed)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], UTTEvents_MC_passed, [0])), where="post",linewidth=2,color='k',label='passed events MC')
    plt.fill_between(Energy_binning, np.concatenate(([0], UTTEvents_MC_passed)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='k')
   
    UTTEvents_ISS= UTTEvents_ISS/np.sum(UTTEvents_ISS)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], UTTEvents_ISS, [0])), where="post",linewidth=2, color='r', label='total events ISS')
    plt.fill_between(Energy_binning, np.concatenate(([0], UTTEvents_ISS)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='r')
    
    UTTEvents_ISS_passed= UTTEvents_ISS_passed/np.sum(UTTEvents_ISS_passed)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], UTTEvents_ISS_passed, [0])), where="post",linewidth=2,color='m', label='passed events ISS')
    plt.fill_between(Energy_binning, np.concatenate(([0], UTTEvents_ISS_passed)),step="pre",alpha=0.4,color='none',hatch='//', edgecolor='m')
    plt.ylim(bottom=0.00)
    
    
    plt.legend()  
    plt.title("At least one useful TRD track")

    figure.savefig("/Users/yasaman/AMS02/plots/Efficiency/UTT.pdf" , dpi=250)
    plt.close(figure)
 
#####################################################################################

    #At least one useful TOF cluster combination
    figure = plt.figure(figsize=(12, 10))
    plot = figure.subplots(1, 1)
    plot.set_xlabel("Energy/ GeV",fontsize = 26,fontweight = 'bold')
    plot.set_ylabel("Normalaized Events", fontsize = 26,fontweight = 'bold')
    plot.set_xscale("log")
    plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
    plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)
             
    UTCCEvents_MC= UTCCEvents_MC/np.sum(UTCCEvents_MC)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], UTCCEvents_MC, [0])), where="post",linewidth=2,color='b', label='total events MC')
    plt.fill_between(Energy_binning, np.concatenate(([0], UTCCEvents_MC)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='b')
    
    UTCCEvents_MC_passed= UTCCEvents_MC_passed/np.sum(UTCCEvents_MC_passed)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], UTCCEvents_MC_passed, [0])), where="post",linewidth=2,color='k',label='passed events MC')
    plt.fill_between(Energy_binning, np.concatenate(([0], UTCCEvents_MC_passed)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='k')
   
    UTCCEvents_ISS= UTCCEvents_ISS/np.sum(UTCCEvents_ISS)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], UTCCEvents_ISS, [0])), where="post",linewidth=2, color='r', label='total events ISS')
    plt.fill_between(Energy_binning, np.concatenate(([0], UTCCEvents_ISS)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='r')
    
    UTCCEvents_ISS_passed= UTCCEvents_ISS_passed/np.sum(UTCCEvents_ISS_passed)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], UTCCEvents_ISS_passed, [0])), where="post",linewidth=2,color='m', label='passed events ISS')
    plt.fill_between(Energy_binning, np.concatenate(([0], UTCCEvents_ISS_passed)),step="pre",alpha=0.4,color='none',hatch='//', edgecolor='m')
    plt.ylim(bottom=0.00)
    
    
    plt.legend()  
    plt.title("At least one useful TOF cluster combination")

    figure.savefig("/Users/yasaman/AMS02/plots/Efficiency/UTCC.pdf" , dpi=250)
    plt.close(figure)
    
#######################################################################################   
    #Upper TOF charge
    
    figure = plt.figure(figsize=(12, 10))
    plot = figure.subplots(1, 1)
    plot.set_xlabel("Energy/ GeV",fontsize = 26,fontweight = 'bold')
    plot.set_ylabel("Normalaized Events", fontsize = 26,fontweight = 'bold')
    plot.set_xscale("log")
    plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
    plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)
             
    TCEvents_MC= TCEvents_MC/np.sum(TCEvents_MC)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], TCEvents_MC, [0])), where="post",linewidth=2,color='b', label='total events MC')
    plt.fill_between(Energy_binning, np.concatenate(([0], TCEvents_MC)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='b')
    
    TCEvents_MC_passed= TCEvents_MC_passed/np.sum(TCEvents_MC_passed)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], TCEvents_MC_passed, [0])), where="post",linewidth=2,color='k',label='passed events MC')
    plt.fill_between(Energy_binning, np.concatenate(([0], TCEvents_MC_passed)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='k')
   
    TCEvents_ISS= TCEvents_ISS/np.sum(TCEvents_ISS)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], TCEvents_ISS, [0])), where="post",linewidth=2, color='r', label='total events ISS')
    plt.fill_between(Energy_binning, np.concatenate(([0], TCEvents_ISS)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='r')
    
    TCEvents_ISS_passed= TCEvents_ISS_passed/np.sum(TCEvents_ISS_passed)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], TCEvents_ISS_passed, [0])), where="post",linewidth=2,color='m', label='passed events ISS')
    plt.fill_between(Energy_binning, np.concatenate(([0], TCEvents_ISS_passed)),step="pre",alpha=0.4,color='none',hatch='//', edgecolor='m')
    plt.ylim(bottom=0.00)
    
    
    plt.legend()  
    plt.title("Upper TOF charge")

    figure.savefig("/Users/yasaman/AMS02/plots/Efficiency/TC.pdf" , dpi=250)
    plt.close(figure)
#################################################################################

  #Enough active layers in TRD
    
    figure = plt.figure(figsize=(12, 10))
    plot = figure.subplots(1, 1)
    plot.set_xlabel("Energy/ GeV",fontsize = 26,fontweight = 'bold')
    plot.set_ylabel("Normalaized Events", fontsize = 26,fontweight = 'bold')
    plot.set_xscale("log")
    plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
    plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)
             
    EALEvents_MC= EALEvents_MC/np.sum(EALEvents_MC)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], EALEvents_MC, [0])), where="post",linewidth=2,color='b', label='total events MC')
    plt.fill_between(Energy_binning, np.concatenate(([0], EALEvents_MC)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='b')
    
    EALEvents_MC_passed= EALEvents_MC_passed/np.sum(EALEvents_MC_passed)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], EALEvents_MC_passed, [0])), where="post",linewidth=2,color='k',label='passed events MC')
    plt.fill_between(Energy_binning, np.concatenate(([0], EALEvents_MC_passed)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='k')
   
    EALEvents_ISS= EALEvents_ISS/np.sum(EALEvents_ISS)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], EALEvents_ISS, [0])), where="post",linewidth=2, color='r', label='total events ISS')
    plt.fill_between(Energy_binning, np.concatenate(([0], EALEvents_ISS)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='r')
    
    EALEvents_ISS_passed= EALEvents_ISS_passed/np.sum(EALEvents_ISS_passed)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], EALEvents_ISS_passed, [0])), where="post",linewidth=2,color='m', label='passed events ISS')
    plt.fill_between(Energy_binning, np.concatenate(([0], EALEvents_ISS_passed)),step="pre",alpha=0.4,color='none',hatch='//', edgecolor='m')
    plt.ylim(bottom=0.00)
    
    
    plt.legend()  
    plt.title("Enough active layers in TRD")

    figure.savefig("/Users/yasaman/AMS02/plots/Efficiency/EAL.pdf" , dpi=250)
    plt.close(figure)
    
#################################################################################

  #Tracker track goodness-of-fit in Y-projection
    
    figure = plt.figure(figsize=(12, 10))
    plot = figure.subplots(1, 1)
    plot.set_xlabel("Energy/ GeV",fontsize = 26,fontweight = 'bold')
    plot.set_ylabel("Normalaized Events", fontsize = 26,fontweight = 'bold')
    plot.set_xscale("log")
    plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
    plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)
             
    TTGEvents_MC= TTGEvents_MC/np.sum(TTGEvents_MC)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], TTGEvents_MC, [0])), where="post",linewidth=2,color='b', label='total events MC')
    plt.fill_between(Energy_binning, np.concatenate(([0], TTGEvents_MC)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='b')
    
    TTGEvents_MC_passed= TTGEvents_MC_passed/np.sum(TTGEvents_MC_passed)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], TTGEvents_MC_passed, [0])), where="post",linewidth=2,color='k',label='passed events MC')
    plt.fill_between(Energy_binning, np.concatenate(([0], TTGEvents_MC_passed)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='k')
   
    TTGEvents_ISS= TTGEvents_ISS/np.sum(TTGEvents_ISS)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], TTGEvents_ISS, [0])), where="post",linewidth=2, color='r', label='total events ISS')
    plt.fill_between(Energy_binning, np.concatenate(([0], TTGEvents_ISS)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='r')
    
    TTGEvents_ISS_passed= TTGEvents_ISS_passed/np.sum(TTGEvents_ISS_passed)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], TTGEvents_ISS_passed, [0])), where="post",linewidth=2,color='m', label='passed events ISS')
    plt.fill_between(Energy_binning, np.concatenate(([0], TTGEvents_ISS_passed)),step="pre",alpha=0.4,color='none',hatch='//', edgecolor='m')
    plt.ylim(bottom=0.00)
    
    
    plt.legend()  
    plt.title("Tracker track goodness-of-fit in Y-projection")

    figure.savefig("/Users/yasaman/AMS02/plots/Efficiency/TTG.pdf" , dpi=250)
    plt.close(figure)    
  
#################################################################################

  #Energy rigidity matching
    
    figure = plt.figure(figsize=(12, 10))
    plot = figure.subplots(1, 1)
    plot.set_xlabel("Energy/ GeV",fontsize = 26,fontweight = 'bold')
    plot.set_ylabel("Normalaized Events", fontsize = 26,fontweight = 'bold')
    plot.set_xscale("log")
    plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
    plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)
             
    ERMEvents_MC= ERMEvents_MC/np.sum(ERMEvents_MC)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], ERMEvents_MC, [0])), where="post",linewidth=2,color='b', label='total events MC')
    plt.fill_between(Energy_binning, np.concatenate(([0], ERMEvents_MC)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='b')
    
    ERMEvents_MC_passed= ERMEvents_MC_passed/np.sum(ERMEvents_MC_passed)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], ERMEvents_MC_passed, [0])), where="post",linewidth=2,color='k',label='passed events MC')
    plt.fill_between(Energy_binning, np.concatenate(([0], ERMEvents_MC_passed)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='k')
   
    ERMEvents_ISS= ERMEvents_ISS/np.sum(ERMEvents_ISS)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], ERMEvents_ISS, [0])), where="post",linewidth=2, color='r', label='total events ISS')
    plt.fill_between(Energy_binning, np.concatenate(([0], ERMEvents_ISS)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='r')
    
    ERMEvents_ISS_passed= ERMEvents_ISS_passed/np.sum(ERMEvents_ISS_passed)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], ERMEvents_ISS_passed, [0])), where="post",linewidth=2,color='m', label='passed events ISS')
    plt.fill_between(Energy_binning, np.concatenate(([0], ERMEvents_ISS_passed)),step="pre",alpha=0.4,color='none',hatch='//', edgecolor='m')
    plt.ylim(bottom=0.00)
    
    
    plt.legend()  
    plt.title("Energy rigidity matching")

    figure.savefig("/Users/yasaman/AMS02/plots/Efficiency/ERM.pdf" , dpi=250)
    plt.close(figure)  

#################################################################################

  #Tracker ECAL matching in X-projection
    
    figure = plt.figure(figsize=(12, 10))
    plot = figure.subplots(1, 1)
    plot.set_xlabel("Energy/ GeV",fontsize = 26,fontweight = 'bold')
    plot.set_ylabel("Normalaized Events", fontsize = 26,fontweight = 'bold')
    plot.set_xscale("log")
    plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
    plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)
             
    TEMEvents_MC= TEMEvents_MC/np.sum(TEMEvents_MC)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], TEMEvents_MC, [0])), where="post",linewidth=2,color='b', label='total events MC')
    plt.fill_between(Energy_binning, np.concatenate(([0], TEMEvents_MC)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='b')
    
    TEMEvents_MC_passed= TEMEvents_MC_passed/np.sum(TEMEvents_MC_passed)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], TEMEvents_MC_passed, [0])), where="post",linewidth=2,color='k',label='passed events MC')
    plt.fill_between(Energy_binning, np.concatenate(([0], TEMEvents_MC_passed)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='k')
   
    TEMEvents_ISS= TEMEvents_ISS/np.sum(TEMEvents_ISS)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], TEMEvents_ISS, [0])), where="post",linewidth=2, color='r', label='total events ISS')
    plt.fill_between(Energy_binning, np.concatenate(([0], TEMEvents_ISS)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='r')
    
    TEMEvents_ISS_passed= TEMEvents_ISS_passed/np.sum(TEMEvents_ISS_passed)
    plot.step(np.concatenate(([Energy_binning[0]], Energy_binning)), np.concatenate(([0], TEMEvents_ISS_passed, [0])), where="post",linewidth=2,color='m', label='passed events ISS')
    plt.fill_between(Energy_binning, np.concatenate(([0], TEMEvents_ISS_passed)),step="pre",alpha=0.4,color='none',hatch='//', edgecolor='m')
    plt.ylim(bottom=0.00)
    
    
    plt.legend()  
    plt.title("Tracker ECAL matching in X-projection")

    figure.savefig("/Users/yasaman/AMS02/plots/Efficiency/TEM.pdf" , dpi=250)
    plt.close(figure)  
    
####################################################################

    figure = plt.figure(figsize=(12, 10))
    plot = figure.subplots(1, 1)
    plot.set_xlabel(r'$\Lambda_{\rm TRD}$',fontsize = 26,fontweight = 'bold')
    plot.set_ylabel("Normalaized Events", fontsize = 26,fontweight = 'bold')
    plot.set_xscale("log")
    plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
    plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)
             
    UTTEvents_MC_TRD= UTTEvents_MC_TRD/np.sum(UTTEvents_MC_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], UTTEvents_MC_TRD, [0])), where="post",linewidth=2,color='b', label='total events MC')
    plt.fill_between(TRD_binning, np.concatenate(([0], UTTEvents_MC_TRD)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='b')
    
    UTTEvents_MC_passed_TRD= UTTEvents_MC_passed_TRD/np.sum(UTTEvents_MC_passed_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], UTTEvents_MC_passed_TRD, [0])), where="post",linewidth=2,color='k',label='passed events MC')
    plt.fill_between(TRD_binning, np.concatenate(([0], UTTEvents_MC_passed_TRD)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='k')
   
    UTTEvents_ISS_TRD= UTTEvents_ISS_TRD/np.sum(UTTEvents_ISS_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], UTTEvents_ISS_TRD, [0])), where="post",linewidth=2, color='r', label='total events ISS')
    plt.fill_between(TRD_binning, np.concatenate(([0], UTTEvents_ISS_TRD)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='r')
    
    UTTEvents_ISS_passed_TRD= UTTEvents_ISS_passed_TRD/np.sum(UTTEvents_ISS_passed_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], UTTEvents_ISS_passed_TRD, [0])), where="post",linewidth=2,color='m', label='passed events ISS')
    plt.fill_between(TRD_binning, np.concatenate(([0], UTTEvents_ISS_passed_TRD)),step="pre",alpha=0.4,color='none',hatch='//', edgecolor='m')
    plt.ylim(bottom=0.00)
    
    
    plt.legend()  
    plt.title("At least one useful TRD track")

    figure.savefig("/Users/yasaman/AMS02/plots/Efficiency/UTT_TRD.pdf" , dpi=250)
    plt.close(figure)  
    
#########################################################################
    #At least one useful TOF cluster combination

    figure = plt.figure(figsize=(12, 10))
    plot = figure.subplots(1, 1)
    plot.set_xlabel(r'$\Lambda_{\rm TRD}$',fontsize = 26,fontweight = 'bold')
    plot.set_ylabel("Normalaized Events", fontsize = 26,fontweight = 'bold')
    plot.set_xscale("log")
    plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
    plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)
             
    UTCCEvents_MC_TRD= UTCCEvents_MC_TRD/np.sum(UTCCEvents_MC_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], UTCCEvents_MC_TRD, [0])), where="post",linewidth=2,color='b', label='total events MC')
    plt.fill_between(TRD_binning, np.concatenate(([0], UTCCEvents_MC_TRD)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='b')
    
    UTCCEvents_MC_passed_TRD= UTCCEvents_MC_passed_TRD/np.sum(UTCCEvents_MC_passed_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], UTCCEvents_MC_passed_TRD, [0])), where="post",linewidth=2,color='k',label='passed events MC')
    plt.fill_between(TRD_binning, np.concatenate(([0], UTCCEvents_MC_passed_TRD)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='k')
   
    UTCCEvents_ISS_TRD= UTCCEvents_ISS_TRD/np.sum(UTCCEvents_ISS_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], UTCCEvents_ISS_TRD, [0])), where="post",linewidth=2, color='r', label='total events ISS')
    plt.fill_between(TRD_binning, np.concatenate(([0], UTCCEvents_ISS_TRD)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='r')
    
    UTCCEvents_ISS_passed_TRD= UTCCEvents_ISS_passed_TRD/np.sum(UTCCEvents_ISS_passed_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], UTCCEvents_ISS_passed_TRD, [0])), where="post",linewidth=2,color='m', label='passed events ISS')
    plt.fill_between(TRD_binning, np.concatenate(([0], UTCCEvents_ISS_passed_TRD)),step="pre",alpha=0.4,color='none',hatch='//', edgecolor='m')
    plt.ylim(bottom=0.00)
    
    
    plt.legend()  
    plt.title("At least one useful TOF cluster combination")

    figure.savefig("/Users/yasaman/AMS02/plots/Efficiency/UTCC_TRD.pdf" , dpi=250)
    plt.close(figure) 

##########################################################################
    #Upper TOF charge

    figure = plt.figure(figsize=(12, 10))
    plot = figure.subplots(1, 1)
    plot.set_xlabel(r'$\Lambda_{\rm TRD}$',fontsize = 26,fontweight = 'bold')
    plot.set_ylabel("Normalaized Events", fontsize = 26,fontweight = 'bold')
    plot.set_xscale("log")
    plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
    plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)
             
    TCEvents_MC_TRD= TCEvents_MC_TRD/np.sum(TCEvents_MC_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], TCEvents_MC_TRD, [0])), where="post",linewidth=2,color='b', label='total events MC')
    plt.fill_between(TRD_binning, np.concatenate(([0], TCEvents_MC_TRD)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='b')
    
    TCEvents_MC_passed_TRD= TCEvents_MC_passed_TRD/np.sum(TCEvents_MC_passed_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], TCEvents_MC_passed_TRD, [0])), where="post",linewidth=2,color='k',label='passed events MC')
    plt.fill_between(TRD_binning, np.concatenate(([0], TCEvents_MC_passed_TRD)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='k')
   
    TCEvents_ISS_TRD= TCEvents_ISS_TRD/np.sum(TCEvents_ISS_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], TCEvents_ISS_TRD, [0])), where="post",linewidth=2, color='r', label='total events ISS')
    plt.fill_between(TRD_binning, np.concatenate(([0], TCEvents_ISS_TRD)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='r')
    
    TCEvents_ISS_passed_TRD= TCEvents_ISS_passed_TRD/np.sum(TCEvents_ISS_passed_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], TCEvents_ISS_passed_TRD, [0])), where="post",linewidth=2,color='m', label='passed events ISS')
    plt.fill_between(TRD_binning, np.concatenate(([0], TCEvents_ISS_passed_TRD)),step="pre",alpha=0.4,color='none',hatch='//', edgecolor='m')
    plt.ylim(bottom=0.00)
    
    
    plt.legend()  
    plt.title("Upper TOF charge")

    figure.savefig("/Users/yasaman/AMS02/plots/Efficiency/TC_TRD.pdf" , dpi=250)
    plt.close(figure)  
    
    
#########################################################################
    #Enough active layers in TRD

    figure = plt.figure(figsize=(12, 10))
    plot = figure.subplots(1, 1)
    plot.set_xlabel(r'$\Lambda_{\rm TRD}$',fontsize = 26,fontweight = 'bold')
    plot.set_ylabel("Normalaized Events", fontsize = 26,fontweight = 'bold')
    plot.set_xscale("log")
    plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
    plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)
             
    EALEvents_MC_TRD= EALEvents_MC_TRD/np.sum(EALEvents_MC_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], EALEvents_MC_TRD, [0])), where="post",linewidth=2,color='b', label='total events MC')
    plt.fill_between(TRD_binning, np.concatenate(([0], EALEvents_MC_TRD)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='b')
    
    EALEvents_MC_passed_TRD= EALEvents_MC_passed_TRD/np.sum(EALEvents_MC_passed_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], EALEvents_MC_passed_TRD, [0])), where="post",linewidth=2,color='k',label='passed events MC')
    plt.fill_between(TRD_binning, np.concatenate(([0], EALEvents_MC_passed_TRD)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='k')
   
    EALEvents_ISS_TRD= EALEvents_ISS_TRD/np.sum(EALEvents_ISS_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], EALEvents_ISS_TRD, [0])), where="post",linewidth=2, color='r', label='total events ISS')
    plt.fill_between(TRD_binning, np.concatenate(([0], EALEvents_ISS_TRD)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='r')
    
    EALEvents_ISS_passed_TRD= EALEvents_ISS_passed_TRD/np.sum(EALEvents_ISS_passed_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], EALEvents_ISS_passed_TRD, [0])), where="post",linewidth=2,color='m', label='passed events ISS')
    plt.fill_between(TRD_binning, np.concatenate(([0], EALEvents_ISS_passed_TRD)),step="pre",alpha=0.4,color='none',hatch='//', edgecolor='m')
    plt.ylim(bottom=0.00)
    
    
    plt.legend()  
    plt.title("Enough active layers in TRD")

    figure.savefig("/Users/yasaman/AMS02/plots/Efficiency/EAL_TRD.pdf" , dpi=250)
    plt.close(figure)   

###########################################################################
    #Tracker track goodness-of-fit in Y-projection


    figure = plt.figure(figsize=(12, 10))
    plot = figure.subplots(1, 1)
    plot.set_xlabel(r'$\Lambda_{\rm TRD}$',fontsize = 26,fontweight = 'bold')
    plot.set_ylabel("Normalaized Events", fontsize = 26,fontweight = 'bold')
    plot.set_xscale("log")
    plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
    plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)
             
    TTGEvents_MC_TRD= TTGEvents_MC_TRD/np.sum(TTGEvents_MC_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], TTGEvents_MC_TRD, [0])), where="post",linewidth=2,color='b', label='total events MC')
    plt.fill_between(TRD_binning, np.concatenate(([0], TTGEvents_MC_TRD)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='b')
    
    TTGEvents_MC_passed_TRD= TTGEvents_MC_passed_TRD/np.sum(TTGEvents_MC_passed_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], TTGEvents_MC_passed_TRD, [0])), where="post",linewidth=2,color='k',label='passed events MC')
    plt.fill_between(TRD_binning, np.concatenate(([0], TTGEvents_MC_passed_TRD)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='k')
   
    TTGEvents_ISS_TRD= TTGEvents_ISS_TRD/np.sum(TTGEvents_ISS_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], TTGEvents_ISS_TRD, [0])), where="post",linewidth=2, color='r', label='total events ISS')
    plt.fill_between(TRD_binning, np.concatenate(([0], TTGEvents_ISS_TRD)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='r')
    
    TTGEvents_ISS_passed_TRD= TTGEvents_ISS_passed_TRD/np.sum(TTGEvents_ISS_passed_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], TTGEvents_ISS_passed_TRD, [0])), where="post",linewidth=2,color='m', label='passed events ISS')
    plt.fill_between(TRD_binning, np.concatenate(([0], TTGEvents_ISS_passed_TRD)),step="pre",alpha=0.4,color='none',hatch='//', edgecolor='m')
    plt.ylim(bottom=0.00)
    
    
    plt.legend()  
    plt.title("Tracker track goodness-of-fit in Y-projection")

    figure.savefig("/Users/yasaman/AMS02/plots/Efficiency/TTG_TRD.pdf" , dpi=250)
    plt.close(figure)    

###########################################################################
 #Energy rigidity matching

    figure = plt.figure(figsize=(12, 10))
    plot = figure.subplots(1, 1)
    plot.set_xlabel(r'$\Lambda_{\rm TRD}$',fontsize = 26,fontweight = 'bold')
    plot.set_ylabel("Normalaized Events", fontsize = 26,fontweight = 'bold')
    plot.set_xscale("log")
    plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
    plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)
             
    ERMEvents_MC_TRD= ERMEvents_MC_TRD/np.sum(ERMEvents_MC_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], ERMEvents_MC_TRD, [0])), where="post",linewidth=2,color='b', label='total events MC')
    plt.fill_between(TRD_binning, np.concatenate(([0], ERMEvents_MC_TRD)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='b')
    
    ERMEvents_MC_passed_TRD= ERMEvents_MC_passed_TRD/np.sum(ERMEvents_MC_passed_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], ERMEvents_MC_passed_TRD, [0])), where="post",linewidth=2,color='k',label='passed events MC')
    plt.fill_between(TRD_binning, np.concatenate(([0], ERMEvents_MC_passed_TRD)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='k')
   
    ERMEvents_ISS_TRD= ERMEvents_ISS_TRD/np.sum(ERMEvents_ISS_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], ERMEvents_ISS_TRD, [0])), where="post",linewidth=2, color='r', label='total events ISS')
    plt.fill_between(TRD_binning, np.concatenate(([0], ERMEvents_ISS_TRD)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='r')
    
    ERMEvents_ISS_passed_TRD= ERMEvents_ISS_passed_TRD/np.sum(ERMEvents_ISS_passed_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], ERMEvents_ISS_passed_TRD, [0])), where="post",linewidth=2,color='m', label='passed events ISS')
    plt.fill_between(TRD_binning, np.concatenate(([0], ERMEvents_ISS_passed_TRD)),step="pre",alpha=0.4,color='none',hatch='//', edgecolor='m')
    plt.ylim(bottom=0.00)
    
    
    plt.legend()  
    plt.title("Energy rigidity matching")

    figure.savefig("/Users/yasaman/AMS02/plots/Efficiency/ERM_TRD.pdf" , dpi=250)
    plt.close(figure)    


###########################################################################
#Tracker ECAL matching in X-projection

    figure = plt.figure(figsize=(12, 10))
    plot = figure.subplots(1, 1)
    plot.set_xlabel(r'$\Lambda_{\rm TRD}$',fontsize = 26,fontweight = 'bold')
    plot.set_ylabel("Normalaized Events", fontsize = 26,fontweight = 'bold')
    plot.set_xscale("log")
    plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
    plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)
             
    TEMEvents_MC_TRD= TEMEvents_MC_TRD/np.sum(TEMEvents_MC_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], TEMEvents_MC_TRD, [0])), where="post",linewidth=2,color='b', label='total events MC')
    plt.fill_between(TRD_binning, np.concatenate(([0], TEMEvents_MC_TRD)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='b')
    
    TEMEvents_MC_passed_TRD= TEMEvents_MC_passed_TRD/np.sum(TEMEvents_MC_passed_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], TEMEvents_MC_passed_TRD, [0])), where="post",linewidth=2,color='k',label='passed events MC')
    plt.fill_between(TRD_binning, np.concatenate(([0], TEMEvents_MC_passed_TRD)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='k')
   
    TEMEvents_ISS_TRD= TEMEvents_ISS_TRD/np.sum(TEMEvents_ISS_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], TEMEvents_ISS_TRD, [0])), where="post",linewidth=2, color='r', label='total events ISS')
    plt.fill_between(TRD_binning, np.concatenate(([0], TEMEvents_ISS_TRD)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='r')
    
    TEMEvents_ISS_passed_TRD= TEMEvents_ISS_passed_TRD/np.sum(TEMEvents_ISS_passed_TRD)
    plot.step(np.concatenate(([TRD_binning[0]], TRD_binning)), np.concatenate(([0], TEMEvents_ISS_passed_TRD, [0])), where="post",linewidth=2,color='m', label='passed events ISS')
    plt.fill_between(TRD_binning, np.concatenate(([0], TEMEvents_ISS_passed_TRD)),step="pre",alpha=0.4,color='none',hatch='//', edgecolor='m')
    plt.ylim(bottom=0.00)
    
    
    plt.legend()  
    plt.title("Tracker ECAL matching in X-projection")

    figure.savefig("/Users/yasaman/AMS02/plots/Efficiency/TEM_TRD.pdf" , dpi=250)
    plt.close(figure)    
        
##############################################################

    var_name= 'TofUpperCharge'
    var_min = 0
    var_max = 2.5
    bin_num = 250    
    var_binning = TOF_charge_binning
    
    for i in range(len(Energy_binning)-1):
        
        a= Energy_binning[i]
        b= Energy_binning[i+1]
        
        MC_eff = np.sum(MC_TOF_Events[:,i][var_binning[:-1] <= 2])/np.sum(MC_TOF_Events[:,i])
        ISS_eff = np.sum(ISS_TOF_Events[:,i][var_binning[:-1] <= 2])/np.sum(ISS_TOF_Events[:,i])
                
        figure = plt.figure(figsize=(12, 10))
        plot = figure.subplots(1, 1)
        plot.set_xlabel('Upper TOF charge / e',fontsize = 26,fontweight = 'bold')
        plot.set_ylabel("Normalaized Events", fontsize = 26,fontweight = 'bold')
        plot.set_yscale("log")
        plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
        plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.minorticks_on()
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        # plot.set_xlim(0,2.2)
        # plot.set_ylim(10**-5,0.5)
        for axis in ['top','bottom','left','right']:
            plot.spines[axis].set_linewidth(2)
            
        print(MC_TOF_Events.shape)
        MC_Events= MC_TOF_Events[:,i]/np.sum(MC_TOF_Events[:,i])
        ISS_Events= ISS_TOF_Events[:,i]/np.sum(ISS_TOF_Events[:,i])

        plot.step(np.concatenate(([var_binning[0]], var_binning)), np.concatenate(([0], MC_Events, [0])), where="post",linewidth=2, label="MC")
        plt.fill_between(var_binning, np.concatenate(([0], MC_Events)),step="pre",alpha=0.4,color='none',hatch='//')
    
        plot.step(np.concatenate(([var_binning[0]], var_binning)), np.concatenate(([0], ISS_Events, [0])), where="post",linewidth=2, label="ISS")
        plt.fill_between(var_binning, np.concatenate(([0], ISS_Events)),step="pre",alpha=0.4,color='none',hatch='//')
    
        plt.axvline(x=2 ,linestyle=':',color='k',linewidth=5)

        
        info = [f"Energy = [{a} , {b}]GeV \n",
                f"ISS eff = {ISS_eff} \n",
                f"MC eff = {MC_eff}"]
            # f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {CCMVA_chi[binn]:.1f} \n",
            # f"$\\p value$ = {CCMVA_pvalue[binn]:.1f}"]
        
        plt.legend(title=''.join(info),title_fontsize=15,fontsize=15,frameon=True, loc='upper left')
    
        figure.savefig("/Users/yasaman/AMS02/plots/Efficiency/UpperTof_"+str(i)+".pdf" , dpi=250)
        plt.close(figure)  
                
        
        
if __name__ == "__main__":
    main()         
        
        
        
        