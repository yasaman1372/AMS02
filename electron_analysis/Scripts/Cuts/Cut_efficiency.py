#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:18:31 2022

@author: yasaman
"""
import multiprocessing as mp
import os

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors


from Cuts.Selection_Cuts import *
from tools.roottree import read_tree


def calculate_efficiency(passed, all):
    return (passed + 1) / (all + 2)

def calculate_efficiency_error(passed, all):
    k = passed
    n = all
    return np.sqrt(((k + 1) * (k + 2)) / ((n + 2) * (n + 3)) - (k + 1)**2 / (n + 2)**2)

def calculate_efficiency_and_error(passed, all):
    return calculate_efficiency(passed, all), calculate_efficiency_error(passed, all)



def handle_file(arg):
    # This function is called once for each process.
    # rank and nranks specifies which process out of how many processes this one is.
    # Based on that, read_tree knows which files to read.
    

    filename, treename, chunk_size, rank, nranks, kwargs = arg

    resultdir = kwargs["resultdir"]

    
    var_name = "TotalEnergy3D"

    var_binning = np.array([0.5,0.65,0.82,1.01,1.22,1.46,1.72,2.0,2.31,2.65,3.0,3.36,3.73,
                              4.12,4.54,5.0,5.49,6.0,6.54,7.1,7.69,8.3,8.95,9.62,10.32,11.04,
                              11.8,12.59,13.41,14.25,15.14,16.05,17.0,17.98,18.99,20.04,21.13,
                              22.25,23.42,24.62,25.9,27.25,28.68,30.21,31.82,33.53,35.36,37.31,
                              39.39,41.61,44.0,46.57,49.33,52.33,55.58,59.13,63.02,67.3,72.05,
                              77.37,83.36,90.19,98.08,107.3,118.4,132.1,148.8,169.9,197.7,237.2,
                              290.0,370.0,500.0,700.0,1000.0])


    selections = [TofVelocity, UpperTofCharge, TrdActiveLayers,TrackerHitPattern, InnerTrackerCharge,
                  TrackerChiSquareY, ERmatching, TrdTrackEcalMatch, TrackerTrackEcalMatch, GeomagneticCutoff]  
    
    
    Events = np.zeros(len(var_binning) -1)
    Events_after_selection = np.zeros((len(selections),len(var_binning) -1))
            
    for events in read_tree(filename, treename, chunk_size=chunk_size, rank=rank, nranks=nranks):
        
        events = NegativeRigidity(events)
        
        for i in range(len(selections)):
            events_after_selection = events
            for inner_selection in selections:
                if selections[i] != inner_selection:
                    events_after_selection = inner_selection(events_after_selection)
                    
            hist_values_after_selection, edges_after_selection = np.histogram(ak.to_numpy(events_after_selection[var_name]), bins= var_binning)        
            Events_after_selection[i] += hist_values_after_selection        
                    
        for j in range(len(selections)):
            events = selections[j](events)
            
                                    
        hist_values, edges = np.histogram(ak.to_numpy(events[var_name]), bins= var_binning)        
        Events += hist_values
        


    
    np.savez(os.path.join(resultdir, f"results_{rank}.npz"),Events=Events, Events_after_selection=Events_after_selection, var_binning=var_binning)
    
def make_args(filename, treename, chunk_size, nranks, **kwargs):
    # this function creates the arguments for handle_file for each process
    for rank in range(nranks):
        yield (filename, treename, chunk_size, rank, nranks, kwargs)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)")
    parser.add_argument("filename",help="Path to root file to read tree from. Can also be path to a file with a list of root files or a pattern of root files (e.g. results/ExampleAnalysisTree*.root)")
    #parser.add_argument("--filename", default="/Users/yasaman/AMS02/data/ISS/ExampleAnalysis_Tree_00079_00047.root",help="Path to root file to read tree from. Can also be path to a file with a list of root files or a pattern of root files (e.g. results/ExampleAnalysisTree*.root)")
    parser.add_argument("--treename", default="ExampleAnalysisTree", help="Name of the tree in the root file.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--resultdir", default="results", help="Directory to store plots and result files in.")

    args = parser.parse_args()

    
    # make sure the directory we will store results in exists
    os.makedirs(args.resultdir, exist_ok=True)
    
    # create pool of worker processes
    with mp.Pool(args.nprocesses) as pool:
        # create arguments for the individual processes
        pool_args = make_args(args.filename, args.treename, args.chunk_size, args.nprocesses, resultdir=args.resultdir)
        # and execute handle_file for each process
        for _ in pool.imap_unordered(handle_file, pool_args):
            pass

    # now all processes are done, load the results and merge them
    # var1_binning = None
    var_binning = None
    Events = None
    Events_after_selection = None
    for rank in range(args.nprocesses):
        filename = os.path.join(args.resultdir, f"results_{rank}.npz")
        with np.load(filename) as result_file:
            if var_binning is None:
                # var1_binning = result_file["var1_binning"]
                var_binning = result_file["var_binning"]
            else:
                # make sure all processes used the same binning
                # assert np.all(var1_binning == result_file["var1_binning"])
                assert np.all(var_binning == result_file["var_binning"])
                   
                
            if Events is None:
                Events = result_file["Events"]

            else:
                Events += result_file["Events"]
                
            if Events_after_selection is None:      
                Events_after_selection = result_file["Events_after_selection"]

            else:
                Events_after_selection += result_file["Events_after_selection"]    

    cut_names=["TofVelocity", "UpperTofCharge", "TrdActiveLayers","InnerTrackerCharge","TrackerHitPattern","TrackerChiSquareY",
                "ERmatching","TrdTrackEcalMatch","TrackerTrackEcalMatch","GeomagneticCutoff" ]

    eff = np.zeros((len(cut_names),len(var_binning)-1))
    err = np.zeros((len(cut_names),len(var_binning)-1))
    
    for i in range(len(cut_names)):
        
        eff[i], err[i] = calculate_efficiency_and_error( Events, Events_after_selection[i])
        
    
        
        figure = plt.figure(figsize=(12, 6.15))
        plot = figure.subplots(1, 1)
        plot.set_title(args.title)
        plot.set_xlabel("Energy/Gev")
        plot.set_ylabel(cut_names[i]+r"$\;$"+"Efficiency")
        plot.set_xscale("log")
        plot.errorbar((var_binning[1:] + var_binning[:-1]) / 2,eff[i],err[i],fmt='.')
        figure.savefig(cut_names[i]+'.pdf', dpi=250)
        plt.show()

    np.savez(os.path.join(args.resultdir, "results.npz"), var_binning=var_binning, eff = eff, err=err)
    

if __name__ == "__main__":
    main()
