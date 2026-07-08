#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 11:37:37 2022

@author: yasaman
"""

import multiprocessing as mp
import os

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors


from Selection_Cuts import *
from tools.roottree import read_tree


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="MC", help="title of the input data (e.g. ISS)")
    parser.add_argument("--filename", default="/Users/yasaman/AMS02/data/HeliumRejection/", help="Path to root file to read tree from. Can also be path to a file with a list of root files or a pattern of root files (e.g. results/ExampleAnalysisTree*.root)")
    parser.add_argument("--treename", default="ExampleAnalysisTree", help="Name of the tree in the root file.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--resultdir", default="results", help="Directory to store plots and result files in.")

    args = parser.parse_args()
    
    
    with np.load(os.path.join(args.filename, "results.npz")) as result_file:
        var1_binning = result_file["var1_binning"]
        var2_binning = result_file["var2_binning"]
        Events = result_file["Events"]
        
    #font_size = 15
    figure = plt.figure(figsize=(12, 10))
    plot = figure.subplots(1, 1)
    plot.set_xlabel("Ecal Energy/ GeV",fontsize = 26)
    plot.set_ylabel("Helium Estimator", fontsize = 26)
    plot.set_xscale("log")
    plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=20)
    plot.tick_params(axis='both', which="minor",direction='in',length=7, width=1.5, labelsize=20)
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    # plot.set_xlim(0.72,1.35)
    # plot.set_ylim(10**-4,1)
    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)
        
        
    events = np.ma.masked_where(Events == 0, Events).transpose()
    scale=1/events.sum(axis=0)
        
    mesh = plot.pcolormesh((var1_binning[1:]+var1_binning[:-1])/2, (var2_binning[1:]+var2_binning[:-1])/2 , events*scale, norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap =plt.cm.get_cmap('jet'))
    cbar = plt.colorbar(mesh, ax=plot, aspect=10)
    cbar.set_label("Normalized Events",fontsize=24)
    cbar.ax.tick_params(which="major",direction='in', length=10, width=1.5,labelsize=20)
    cbar.ax.tick_params(which="minor",direction='in',length=7, width=1.5, labelsize=20)
    # plt.hlines(1.8,0,900,linestyle='dashed',color='k',linewidth=5)
    # plt.hlines(0.5,0,900,linestyle='dashed',color='k',linewidth=5)
    # plot.plot([],[],' ',label="MC data")
    # plot.plot([],[],' ',label="lower cut= 0.5")
    # plot.plot([],[],' ',label="upper cut= 1.8")
    #plt.legend(loc="lower right",fontsize=16,frameon=False)
    plt.legend(bbox_to_anchor=(1, 1.17),fontsize=13,frameon=True)
        
                  
            
    figure.savefig("/Users/yasaman/AMS02/plots/selection_cut/HeliumRejection.pdf", dpi=250)
    
if __name__ == "__main__":
    main() 