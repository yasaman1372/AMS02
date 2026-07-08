#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 14:52:23 2023

@author: yasaman
"""

import multiprocessing as mp
import os

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors


from Cuts.ElectronTagCuts import *
from Cuts.ApplyCuts import *
from tools.roottree import read_tree


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", default="", help="type of the input data (e.g. ISS)")
    parser.add_argument("--filenamepass8",default="", help="Path to the data file pass8")
    parser.add_argument("--filenamepass6",default="", help="Path to the data file pass6")

    args = parser.parse_args()
    
    with np.load(os.path.join(args.filenamepass6)) as result_file:
        Tof_binning = result_file["var1_binning"]
        Events_pass6 = result_file["Events"]
        Passed_Events_pass6 = result_file["Passed_Events"]
        
        
    with np.load(os.path.join(args.filenamepass8)) as result_file:
        Tof_binning = result_file["var1_binning"]
        Events_pass8 = result_file["Events"]
        Passed_Events_pass8 = result_file["Passed_Events"]    
        
        
    
    
    
    figure = plt.figure(figsize=(10, 10))
    ax1 = figure.add_subplot(111)

    
  
    # ax1.set_xlabel('Upper TOF charge / e',fontsize = 26,fontweight = 'bold')
    # ax1.set_ylabel("Normalaized Events", fontsize = 26,fontweight = 'bold')
    ax1.set_yscale("log")
    ax1.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
    ax1.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    ax1.set_xlim(0.3,2.2)
    ax1.set_ylim(10**-5,0.5)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2)
        
    ax1.set_xlabel('Upper TOF charge / e',fontsize = 26,fontweight = 'bold')
    # ax2.set_ylabel("Efficiency", fontsize = 26,fontweight = 'bold') 
    ax1.set_ylabel("Normalaized Events", fontsize = 26,fontweight = 'bold')  
    
    # ax2.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
    # ax2.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
    # for axis in ['top','bottom','left','right']:
    #     ax2.spines[axis].set_linewidth(2)
        
           
    #Events_pass6= Events_pass6/np.sum(Events_pass6)
    eff_pass6 = np.sum(Passed_Events_pass6)/np.sum(Events_pass6)
    Events_pass6= Events_pass6/np.sum(Events_pass6)

    ax1.step(np.concatenate(([Tof_binning[0]], Tof_binning)), np.concatenate(([0], Events_pass6, [0])), where="post",linewidth=2, label ="pass6")
    plt.fill_between(Tof_binning, np.concatenate(([0], Events_pass6)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='r')
    
    # ax2.scatter(Tof_binning, eff_pass6, label="pass6")
    
    #Events_pass8= Events_pass8/np.sum(Events_pass8)
    eff_pass8 = np.sum(Passed_Events_pass8)/np.sum(Events_pass8)
    Events_pass8= Events_pass8/np.sum(Events_pass8)

    ax1.step(np.concatenate(([Tof_binning[0]], Tof_binning)), np.concatenate(([0], Events_pass8, [0])), where="post",linewidth=2, label ="pass8")
    plt.fill_between(Tof_binning, np.concatenate(([0], Events_pass8)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='b')
    
    # ax2.scatter(Tof_binning, eff_pass8, label="pass8")
    
    info = [f"MC data \n"
            f"Eifficiency pass6 = {eff_pass6:.3f} \n",
            f"Eifficiency pass8 = {eff_pass8:.3f}"]
            
        
        #ax1.legend(title=''.join(info),title_fontsize=6,fontsize=6,frameon=False, loc="upper right")
    ax1.legend(title=''.join(info),title_fontsize=15,fontsize=15,frameon=False, loc="upper left")
    
    ax1.axvline(x=2 ,linestyle=':',color='k',linewidth=5)
    
    #plt.legend()
    
    

    figure.savefig("CutTofUpperCharge.pdf" , dpi=250)
    plt.close(figure)
  
    
if __name__ == "__main__":
    main()     
