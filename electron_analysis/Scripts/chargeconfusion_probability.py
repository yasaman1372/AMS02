#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:49:19 2023

@author: yasaman
"""

import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
from numpy import genfromtxt



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataversion", default="pass6", help="version of the data set (e.g. pass6)")
    parser.add_argument("--datatype", default="ISS", help="title of the input data (e.g. ISS)")
    parser.add_argument("--data_filepath", default= "/Users/yasaman/AMS02/YasamanAnalysis/Scripts/template_fit/results/parameters_all.npz",help="Path to root file to read tree from. Can also be path to a file with a list of root files or a pattern of root files (e.g. results/ExampleAnalysisTree*.root)")
    parser.add_argument("--error_filepath", default= "/Users/yasaman/AMS02/YasamanAnalysis/Scripts/template_fit/results/errors_all.npz",help="Path to root file to read tree from. Can also be path to a file with a list of root files or a pattern of root files (e.g. results/ExampleAnalysisTree*.root)")
    args = parser.parse_args()
    
    parameters_all = np.load(args.data_filepath)
    parameters_all_err = np.load(args.error_filepath)
    
    
    p_cce =(parameters_all['alpha_cce']* parameters_all['nel'] * parameters_all["fecc_st"] + (1 - parameters_all['alpha_cce'])* parameters_all['nel'] * parameters_all["fecc_mt"])/parameters_all["nel"]
    p_ccpos = (parameters_all['alpha_ccpos']* parameters_all['npos'] * parameters_all["fecc_st"] + (1 - parameters_all['alpha_ccpos'])* parameters_all['npos'] * parameters_all["fecc_mt"])/parameters_all["npos"]
    
    
    Energy_binning = np.array([0.5,0.65,0.82,1.01,1.22,1.46,1.72,2.0,2.31,2.65,3.0,3.36,3.73,
                 4.12,4.54,5.0,5.49,6.0,6.54,7.1,7.69,8.3,8.95,9.62,10.32,11.04,
                 11.8,12.59,13.41,14.25,15.14,16.05,17.0,17.98,18.99,20.04,21.13,
                 22.25,23.42,24.62,25.9,27.25,28.68,30.21,31.82,33.53,35.36,37.31,
                 39.39,41.61,44.0,46.57,49.33,52.33,55.58,59.13,63.02,67.3,72.05,
                 77.37,83.36,90.19,98.08,107.3,118.4,132.1,148.8,169.9,197.7,237.2,
                 290.0,370.0,500.0,700.0,1000.0])
    
    figure = plt.figure(figsize=(12, 10))
    ax = figure.add_subplot(111)
    ax.set_ylabel(r"$\rm P_{\rm cc}$", fontsize = 26)
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
    #ax.set_yscale("log")
            
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"  
         
    #x = (Energy_binning[1:]+Energy_binning[:-1])/2


    ax.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, p_cce, color = 'k',marker='s', s=50,label="charged confused electron")
    ax.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, p_ccpos, color='m',marker='^',label="charged confused positron")
    plt.legend(fontsize=15)
    plt.show()
    
        
    figure.savefig("/Users/yasaman/AMS02/plots/ccprobability" , dpi=250)
    plt.close(figure)

if __name__ == "__main__":
    main()