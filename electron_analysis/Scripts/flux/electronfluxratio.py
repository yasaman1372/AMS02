#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 09:44:58 2022

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


def handle_file(arg):
    # This function is called once for each process.
    # rank and nranks specifies which process out of how many processes this one is.
    # Based on that, read_tree knows which files to read.

    filename, treename, chunk_size, rank, nranks, kwargs = arg

    resultdir = kwargs["resultdir"]

    var1_name= 'CalculateElectronProtonLikelihood'
    var1_min = 0
    var1_max = 2
    bin1_num = 100
        
    var2_name= 'EcaLIntegralLikelihood3D'
    var2_min = -2
    var2_max = 10
    bin2_num = 100
    
    var3_name = "TotalEnergy3D"
    # var3_min = 0
    # var3_max = 4
    # bin3_num = 20
    
    var1_binning = np.linspace(var1_min, var1_max, bin1_num +1)
    var2_binning = np.linspace(var2_min, var2_max, bin2_num +1)
    var3_binning = np.array([0.5,0.65,0.82,1.01,1.22,1.46,1.72,2.0,2.31,2.65,3.0,3.36,3.73,
                             4.12,4.54,5.0,5.49,6.0,6.54,7.1,7.69,8.3,8.95,9.62,10.32,11.04,
                             11.8,12.59,13.41,14.25,15.14,16.05,17.0,17.98,18.99,20.04,21.13,
                             22.25,23.42,24.62,25.9,27.25,28.68,30.21,31.82,33.53,35.36,37.31,
                             39.39,41.61,44.0,46.57,49.33,52.33,55.58,59.13,63.02,67.3,72.05,
                             77.37,83.36,90.19,98.08,107.3,118.4,132.1,148.8,169.9,197.7,237.2,
                             290.0,370.0,500.0,700.0,1000.0])
    
    bin3_num = len(var3_binning) -1
    el_Events = np.zeros((bin1_num, bin2_num, bin3_num))
    p_Events = np.zeros((bin1_num, bin2_num, bin3_num))
    
    
    for events in read_tree(filename, treename, chunk_size=chunk_size, rank=rank, nranks=nranks):
        
        events = TofVelocity(events)
        events = UpperTofCharge(events)
        events = TrdActiveLayers(events)
        events = InnerTrackerCharge(events)
        events = TrackerChiSquareY(events)
        events = TrdTrackEcalMatch(events)
        events = TrackerTrackEcalMatch(events)
        events = TrackerHitPattern(events)
        events = ERmatching(events)
        
        el_events = NegativeRigidity(events)
        p_events = PositiveRigidity(events)

        elv1=ak.to_numpy(el_events[var1_name])
        elv2=ak.to_numpy(el_events[var2_name])
        elv3=ak.to_numpy(el_events[var3_name])
        
        pv1=ak.to_numpy(p_events[var1_name])
        pv2=ak.to_numpy(p_events[var2_name])
        pv3=ak.to_numpy(p_events[var3_name])

        

        el_hist_values, el_edges = np.histogramdd((elv1,elv2,elv3), bins= (var1_binning, var2_binning, var3_binning))        
        el_Events += el_hist_values
        
        p_hist_values, p_edges = np.histogramdd((pv1,pv2,pv3), bins= (var1_binning, var2_binning, var3_binning))        
        p_Events += p_hist_values
                

    # this process is done, save result
    np.savez(os.path.join(resultdir, f"results_{rank}.npz"), var1_binning=var1_binning, var2_binning=var2_binning, var3_binning=var3_binning, el_Events = el_Events, p_Events = p_Events)
    
def make_args(filename, treename, chunk_size, nranks, **kwargs):
    # this function creates the arguments for handle_file for each process
    for rank in range(nranks):
        yield (filename, treename, chunk_size, rank, nranks, kwargs)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)")
    parser.add_argument("filename", help="Path to root file to read tree from. Can also be path to a file with a list of root files or a pattern of root files (e.g. results/ExampleAnalysisTree*.root)")
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
    var1_binning = None
    var2_binning = None
    var3_binning = None
    el_Events = None
    p_Events = None
    for rank in range(args.nprocesses):
        filename = os.path.join(args.resultdir, f"results_{rank}.npz")
        with np.load(filename) as result_file:
            if var1_binning is None or var2_binning is None or var3_binning is None:
                var1_binning = result_file["var1_binning"]
                var2_binning = result_file["var2_binning"]
                var3_binning = result_file["var3_binning"]
            else:
                # make sure all processes used the same binning
                assert np.all(var1_binning == result_file["var1_binning"])
                assert np.all(var2_binning == result_file["var2_binning"])
                assert np.all(var3_binning == result_file["var3_binning"])
                
            if el_Events is None:
                el_Events = result_file["el_Events"]
            else:
                el_Events += result_file["el_Events"]
                
            if p_Events is None:
                p_Events = result_file["p_Events"]
            else:
                p_Events += result_file["p_Events"]    

    # now save merged result
    np.savez(os.path.join(args.resultdir, "results.npz"), var1_binning=var1_binning, var2_binning=var2_binning, var3_binning = var3_binning, el_Events=el_Events, p_Events=p_Events)

    el_num = np.zeros(len(var3_binning) -1)
    p_num = np.zeros(len(var3_binning) -1)
    
    x=(var1_binning[1:]+var1_binning[:-1])/2
    y=(var2_binning[1:]+var2_binning[:-1])/2
    
    for binn in range(len(var3_binning) -1):

        
        xv, yv = np.meshgrid(y, x)
        sel = yv <= 9 - 6.4 * xv 
        
        el_events = el_Events[: , : , binn] 
        el_num[binn] = el_events[sel].sum()
        
        p_events = p_Events[: , : , binn]    
        p_num[binn] = p_events[sel].sum()
        
    
    flux = el_num/ (el_num + p_num) 
    
    font_size = 15
    figure = plt.figure(figsize=(12, 6.15))
    plot = figure.subplots(1, 1)
    plot.set_title("electron flux ratio", fontsize = font_size)
    plot.set_xlabel("Kinetic Energy" + r"$(\rm Gev)$" ,fontsize = font_size)
    plot.set_ylabel(r"$\frac{e^-}{e^- + e^+}$", fontsize = font_size+5)
    
    plt.scatter((var3_binning[1:]+var3_binning[:-1])/2 , flux, marker='o', facecolor='none', edgecolor='r')
    plot.set_xscale("log")
    figure.savefig("electron_fluxratio.pdf" , dpi=250)
    plt.close(figure)
    
if __name__ == "__main__":
    main()        
        
