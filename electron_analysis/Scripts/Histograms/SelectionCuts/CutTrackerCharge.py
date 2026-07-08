#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 21:12:12 2022

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
from matplotlib.gridspec import GridSpec
from Cuts.ElectronSelectionCuts import *
from Cuts.ElectronIdentificationCuts import *
from scipy import interpolate
from Cuts.ElectronTagCuts import *
from Cuts.ApplyCuts import *
from tools.roottree import read_tree

   

Preselection_Branches = Preselection_Cuts_Branch_List()
Selection_Branches = Selection_Cuts_Branch_List()
Identification_Branches = Identification_Cuts_Branch_List()



Histogram_branches = ["TotalEnergy3D",'TrackerCharge', "EcalEnergyElectronNewMaximumShower", "McEventWeightElectron"]

Branches = Preselection_Branches + Histogram_branches + Selection_Branches + Identification_Branches
 

def handle_file(arg):
    # This function is called once for each process.
    # rank and nranks specifies which process out of how many processes this one is.
    # Based on that, read_tree knows which files to read.

    filename, treename, chunk_size, rank, nranks, kwargs = arg
    data_type = kwargs["data_type"]
    resultdir = kwargs["resultdir"]
    
    var1_name = "TotalEnergy3D"
    var1_binning = np.array([0.5,0.65,0.82,1.01,1.22,1.46,1.72,2.0,2.31,2.65,3.0,3.36,3.73,
                              4.12,4.54,5.0,5.49,6.0,6.54,7.1,7.69,8.3,8.95,9.62,10.32,11.04,
                              11.8,12.59,13.41,14.25,15.14,16.05,17.0,17.98,18.99,20.04,21.13,
                              22.25,23.42,24.62,25.9,27.25,28.68,30.21,31.82,33.53,35.36,37.31,
                              39.39,41.61,44.0,46.57,49.33,52.33,55.58,59.13,63.02,67.3,72.05,
                              77.37,83.36,90.19,98.08,107.3,118.4,132.1,148.8,169.9,197.7,237.2,
                              290.0,370.0,500.0,700.0,1000.0])
    
    
    bin1_num = len(var1_binning) -1

    var2_name= 'TrackerCharge'
    var2_min = 0
    var2_max = 2.5
    bin2_num = 250
    var2_binning = np.linspace(var2_min, var2_max, bin2_num +1)
    
    Events = np.zeros((bin1_num, bin2_num))
    Passed_Events = np.zeros((bin1_num, bin2_num))
    
    
    for events in read_tree(filename, treename, chunk_size=chunk_size, rank=rank, nranks=nranks, branches=Branches):
        events = ApplyPreselectionCuts(events) #preselection cuts are applied
        events = ApplyIdentificationCuts(events)

        #Selection Cuts:
        events = CutTofBeta(events)
        events = CutTofUpperCharge(events)
        events = CutTrdActiveLayers(events)
        events = CutTrdNoHelium(events)
        events = CutTrackerPatternSortedByMDR(events)
        events = CutTrackerChiSquareY(events)
        events = GeomagneticCutoff(events)

        passed_events = CutTrackerCharge(events)
        
       # for cut in Tag_Cuts:
           # events = cut(events) #Tag cuts are applied
        
        v1 = ak.to_numpy(events[var1_name])
        v2 = ak.to_numpy(events[var2_name])     
        
        passed_v1 = ak.to_numpy(passed_events[var1_name])
        passed_v2 = ak.to_numpy(passed_events[var2_name]) 
        # v2= ComputeEcalChiSquareLateralNormalized(events, data_type)
        
        if data_type == 'MC': 
            hist_values, edges = np.histogramdd((v1,v2), bins= (var1_binning, var2_binning), weights=ak.to_numpy(events.McEventWeightElectron))
            passed_hist_values, passed_edges = np.histogramdd((passed_v1,passed_v2), bins= (var1_binning, var2_binning), weights=ak.to_numpy(passed_events.McEventWeightElectron))
            
        
        elif data_type == 'ISS':
            hist_values, edges = np.histogramdd((v1,v2), bins= (var1_binning, var2_binning))
            passed_hist_values, edges = np.histogramdd((passed_v1,passed_v2), bins= (var1_binning, var2_binning))
                     
        Events += hist_values
        Passed_Events += passed_hist_values
        


    # this process is done, save result
    np.savez(os.path.join(resultdir, f"results_{rank}.npz"), var1_binning=var1_binning,var2_binning=var2_binning,Events=Events, Passed_Events = Passed_Events)
    
def make_args(filename, treename, chunk_size, nranks, **kwargs):
    # this function creates the arguments for handle_file for each process
    for rank in range(nranks):
        yield (filename, treename, chunk_size, rank, nranks, kwargs)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", default="MC", help="title of the input data (e.g. ISS)")
    parser.add_argument("filename", help="Path to root file to read tree from. Can also be path to a file with a list of root files or a pattern of root files (e.g. results/ExampleAnalysisTree*.root)")
    parser.add_argument("--treename", default="LeptonAnalysisTree", help="Name of the tree in the root file.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--resultdir", default="results", help="Directory to store plots and result files in.")

    args = parser.parse_args()

    
    # make sure the directory we will store results in exists
    os.makedirs(args.resultdir, exist_ok=True)
    
    # create pool of worker processes
    with mp.Pool(args.nprocesses) as pool:
        # create arguments for the individual processes
        pool_args = make_args(args.filename, args.treename, args.chunk_size, args.nprocesses, resultdir=args.resultdir, data_type = args.data_type)
        # and execute handle_file for each process
        for _ in pool.imap_unordered(handle_file, pool_args):
            pass

    # now all processes are done, load the results and merge them
    var1_binning = None
    var2_binning = None
    Events = None
    for rank in range(args.nprocesses):
        filename = os.path.join(args.resultdir, f"results_{rank}.npz")
        with np.load(filename) as result_file:
            if var1_binning is None or var2_binning is None :
                var1_binning = result_file["var1_binning"]
                var2_binning = result_file["var2_binning"]
            else:
                # make sure all processes used the same binning
                assert np.all(var1_binning == result_file["var1_binning"])
                assert np.all(var2_binning == result_file["var2_binning"])
                
            if Events is None:
                Events = result_file["Events"]
                Passed_Events = result_file["Passed_Events"]
            else:
                Events += result_file["Events"]
                Passed_Events += result_file["Passed_Events"]

    # now save merged result
    np.savez(os.path.join(args.resultdir, "results.npz"),var2_binning=var2_binning, var1_binning=var1_binning, Events=Events, Passed_Events = Passed_Events)
   
    figure = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, width_ratios=(10, 1),height_ratios=(2, 1),hspace=0.1)
    ax1 = figure.add_subplot(gs[0,0])
    ax1.set_xticklabels([]) # remove the x axis numbers
    ax2 = figure.add_subplot(gs[1,0], sharex=ax1)
    cax = figure.add_subplot(gs[0,1])
    
    
    ax2.set_xlabel("Ecal Energy/ GeV",fontsize = 26)
    ax1.set_ylabel("Inner tracker charge / e", fontsize = 26)
    ax1.set_xscale("log")
    ax1.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=20)
    ax1.tick_params(axis='both', which="minor",direction='in',length=7, width=1.5, labelsize=20)
    ax1.set_xlim(0.5,max((var1_binning[1:]+var1_binning[:-1])/2))
    #ax1.set_xticklabels([]) # remove the x axis numbers

    ax2.set_ylabel("Efficiency(%)", fontsize = 26)
    ax2.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=20)
    ax2.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=20)         
    
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2)
        ax2.spines[axis].set_linewidth(2)
        

    #ax1.set_position([0.1, 0.4, 0.8, 0.5])
    #ax2.set_position([0.1, 0.2, 0.62, 0.3])    
    
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"    
        
    # hist_cut = F_cut((var1_binning[1:]+var1_binning[:-1])/2)
    # xbin = (var1_binning[1:]+var1_binning[:-1])/2
    # ybin = (var2_binning[1:]+var2_binning[:-1])/2
    events = np.ma.masked_where(Events == 0, Events).transpose()
    scale=1/events.sum(axis=0)
    events = scale*events
    print(np.shape(events))
    
    
    mesh = ax1.pcolormesh(var1_binning,var2_binning ,events, norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap =plt.cm.get_cmap('jet'))
    cbar = plt.colorbar(mesh, cax=cax, aspect=10)
    cbar.set_label("Normalized Events",fontsize=24)
    cbar.ax.tick_params(which="major",direction='in', length=10, width=1.5,labelsize=20)
    cbar.ax.tick_params(which="minor",direction='in',length=7, width=1.5, labelsize=20)        
    
    ax1.set_xlim(left=0.5)
    ax1.axhline(y = 0.5 ,linestyle=':',color='k',linewidth=5) 
    ax1.axhline(y = 1.8 ,linestyle=':',color='k',linewidth=5)
    # ax1.scatter((var1_binning[1:]+var1_binning[:-1])/2,hist_cut,c="k")  
    
    eff = np.zeros((len((var1_binning[1:]+var1_binning[:-1])/2)))
    for i in range(len((var1_binning[1:]+var1_binning[:-1])/2)):
        eff[i] = np.sum(Passed_Events[i,:])/np.sum(Events[i,:])
    ax2.scatter((var1_binning[1:]+var1_binning[:-1])/2,eff*100,c="k")  
    ax1.legend(title=args.data_type+" electron pass8",title_fontsize=20,frameon=False, loc='upper left')


    figure.savefig("CutTrackerCharge.pdf" , dpi=250)
    plt.close(figure)
    
    
    # figure = plt.figure(figsize=(12, 10))
    # plot = figure.subplots(1, 1)
    # plot.set_xlabel("Ecal Energy/ GeV",fontsize = 26)
    # plot.set_ylabel("Inner tracker charge / e", fontsize = 26)
    # plot.set_xscale("log")
    # plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=20)
    # plot.tick_params(axis='both', which="minor",direction='in',length=7, width=1.5, labelsize=20)
    # plt.rcParams['xtick.top'] = True
    # plt.rcParams['ytick.right'] = True
    # plt.minorticks_on()
    # plt.rcParams["font.weight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"
    # #plot.set_ylim(-2,15)
    # plot.set_xlim(0.5,max((var1_binning[1:]+var1_binning[:-1])/2))
    # for axis in ['top','bottom','left','right']:
    #     plot.spines[axis].set_linewidth(2)
        
        
    # events = np.ma.masked_where(Events == 0, Events).transpose()
    # scale=1/events.sum(axis=0)
    # events = scale*events
    # events = np.ma.masked_where(events == 1e-4, events)	    
    
    # mesh = plot.pcolormesh(var1_binning,var2_binning, events, norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap =plt.cm.get_cmap('jet'))
    # cbar = plt.colorbar(mesh, ax=plot, aspect=10)
    # cbar.set_label("Normalized Events",fontsize=24)
    # cbar.ax.tick_params(which="major",direction='in', length=10, width=1.5,labelsize=20)
    # cbar.ax.tick_params(which="minor",direction='in',length=7, width=1.5, labelsize=20)        
    
    # plt.xlim(left=0.5)
    # plt.axhline(y = 0.5 ,linestyle=':',color='k',linewidth=5) 
    # plt.axhline(y = 1.8 ,linestyle=':',color='k',linewidth=5)
    # figure.savefig("CutTrackerCharge.pdf" , dpi=250)
    # plt.close(figure)
    
    # figure = plt.figure(figsize=(12, 10))
    # plot = figure.subplots(1, 1)
    # plot.set_xlabel("Ecal Energy/ GeV",fontsize = 26)
    # plot.set_ylabel("Efficiency", fontsize = 26)
    # plot.set_xscale("log")
    # plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=20)
    # plot.tick_params(axis='both', which="minor",direction='in',length=7, width=1.5, labelsize=20)
    # plt.rcParams['xtick.top'] = True
    # plt.rcParams['ytick.right'] = True
    # plt.minorticks_on()
    # plt.rcParams["font.weight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"
    #     #plot.set_ylim(-2,15)
    # plot.set_xlim(0.5,max((var1_binning[1:]+var1_binning[:-1])/2))
    # for axis in ['top','bottom','left','right']:
    #     plot.spines[axis].set_linewidth(2)
  
    # eff = np.zeros((len((var1_binning[1:]+var1_binning[:-1])/2)))
    
    # for i in range(len((var1_binning[1:]+var1_binning[:-1])/2)):
    #     print(len(Passed_Events[i,:]))    
    #     eff[i] = np.sum(Passed_Events[i,:])/np.sum(Events[i,:])
    # plt.scatter((var1_binning[1:]+var1_binning[:-1])/2,eff,c="k")   
        
        
    # figure.savefig("TrackerChargeEfficiency..pdf" , dpi=250)
    # plt.close(figure) 
  
    
if __name__ == "__main__":
    main()
