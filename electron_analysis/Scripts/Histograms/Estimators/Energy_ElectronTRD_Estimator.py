#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:00:09 2023

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

   

Preselection_Branches = Preselection_Cuts_Branch_List()
Selection_Cuts_Branches = Selection_Cuts_Branch_List()
Identification_Cuts_Branches =Identification_Cuts_Branch_List()
Hist_branches = ["TrdPLikelihoodTrackerHitsElectron","TrdPLikelihoodHybridHitsElectron","TotalEnergy3D",'EcalBDT3D','EcalBDTSmoothed_v7_EnergyD',"TrackerTrackGBLMaxSpanRigidity","EcalEnergyElectronNewMaximumShower", "TrdPLikelihoodHybridHitsElectronECAL", 
                 "TrdPLikelihoodHybridHitsProtonECAL"]

Branches = Preselection_Branches + Selection_Cuts_Branches + Identification_Cuts_Branches + Hist_branches


def handle_file(arg):
    # This function is called once for each process.
    # rank and nranks specifies which process out of how many processes this one is.
    # Based on that, read_tree knows which files to read.

    filename, treename, chunk_size, rank, nranks, kwargs = arg
    data_type= kwargs["data_type"]
    resultdir = kwargs["resultdir"]

    var2_name= "TrdPLikelihoodHybridHitsElectronECAL"
    var2_min = np.log10(4)
    var2_max = np.log10(12)
    bin2_num = 100
    var2_binning = np.logspace(var2_min, var2_max, bin2_num +1)
  
    var1_name = "TotalEnergy3D" 
    var1_binning = np.array([0.5,0.65,0.82,1.01,1.22,1.46,1.72,2.0,2.31,2.65,3.0,3.36,3.73,
                              4.12,4.54,5.0,5.49,6.0,6.54,7.1,7.69,8.3,8.95,9.62,10.32,11.04,
                              11.8,12.59,13.41,14.25,15.14,16.05,17.0,17.98,18.99,20.04,21.13,
                              22.25,23.42,24.62,25.9,27.25,28.68,30.21,31.82,33.53,35.36,37.31,
                              39.39,41.61,44.0,46.57,49.33,52.33,55.58,59.13,63.02,67.3,72.05,
                              77.37,83.36,90.19,98.08,107.3,118.4,132.1,148.8,169.9,197.7,237.2,
                              290.0,370.0,500.0,700.0,1000.0])
    
    bin1_num = len(var1_binning) -1
  
    Events = np.zeros((bin1_num, bin2_num))

    for events in read_tree(filename, treename, chunk_size=chunk_size, rank=rank, nranks=nranks):
        

        events = ApplyPreselectionCuts(events) #preselection cuts are applied
        events = ApplySelectionCuts(events)
        events = ApplyIdentificationCuts(events)

        
        v2 = -np.log(ak.to_numpy(events[var2_name]))
        v1=ak.to_numpy(events[var1_name])

        if data_type == "MC":
            hist_values, edges = np.histogramdd((v1,v2), bins= (var1_binning, var2_binning), weights=ak.to_numpy(events.McEventWeightElectron))
       # weights=ak.to_numpy(events.McEventWeightElectron)
        

        elif data_type == "ISS":
            hist_values, edges = np.histogramdd((v1,v2), bins= (var1_binning, var2_binning))       
        Events += hist_values
        
    # this process is done, save result
    np.savez(os.path.join(resultdir, f"results_{rank}.npz"), var1_binning=var1_binning, var2_binning=var2_binning, Events = Events)
    
def make_args(filename, treename, chunk_size, nranks, **kwargs):
    # this function creates the arguments for handle_file for each process
    for rank in range(nranks):
        yield (filename, treename, chunk_size, rank, nranks, kwargs)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", default="ISS", help="title of the input data (e.g. ISS)")
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
            if var1_binning is None or var2_binning is None:
                var1_binning = result_file["var1_binning"]
                var2_binning = result_file["var2_binning"]

            else:
                # make sure all processes used the same binning
                assert np.all(var1_binning == result_file["var1_binning"])
                assert np.all(var2_binning == result_file["var2_binning"])
            
                
            if Events is None:
                Events = result_file["Events"]

            else:
                Events += result_file["Events"]

    # now save merged result
    np.savez(os.path.join(args.resultdir, "results.npz") , var2_binning=var2_binning, var1_binning = var1_binning, Events = Events)
    

        
    figure = plt.figure(figsize=(12, 10))
    plot = figure.subplots(1, 1)
    plot.set_ylabel(r'$\L^{\rm e}_{\rm TRD}$',fontsize = 26)
    plot.set_xlabel("Energy GeV", fontsize = 26)
    plot.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
    plot.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
    #plot.set_yscale("log")
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)


    events = np.ma.masked_where(Events == 0, Events).transpose()
    events = events/np.sum(events, axis=0)
    plot.set_xscale("log")
    Energy_binn = (var1_binning[1:]+var1_binning[:-1])/2
    TRD_binn= (var2_binning[1:]+var2_binning[:-1])/2
        
    mesh = plot.pcolormesh(Energy_binn,TRD_binn,events, norm=colors.LogNorm(), cmap =plt.cm.get_cmap('jet'))
    cbar = plt.colorbar(mesh, ax=plot, aspect=10)
    cbar.set_label("Normalized Events",fontsize=25)
        
    cbar.ax.tick_params(which="major",direction='in', length=7, width=1.5,labelsize=20)
    cbar.ax.tick_params(which="minor",direction='in')
        
   
    plot.plot([],[],' ',label=args.data_type + " data pass8")
    plt.legend(fontsize=17,frameon=True,labelcolor='k')
        
        
    figure.savefig("2D_Energy_TRD_Estimator_"+args.data_type+".pdf" , dpi=250)

  
        
if __name__ == "__main__":
    main() 
