#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 14:43:55 2022

@author: yasaman
"""


import multiprocessing as mp
import os

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors


from ApplyCuts import *
from ElectronIdentificationCuts import *
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
    
    var2_name = "TotalEnergy3D"

    
    var1_binning = np.linspace(var1_min, var1_max, bin1_num +1)
    var2_binning = np.array([0.5,0.65,0.82,1.01,1.22,1.46,1.72,2.0,2.31,2.65,3.0,3.36,3.73,
                             4.12,4.54,5.0,5.49,6.0,6.54,7.1,7.69,8.3,8.95,9.62,10.32,11.04,
                             11.8,12.59,13.41,14.25,15.14,16.05,17.0,17.98,18.99,20.04,21.13,
                             22.25,23.42,24.62,25.9,27.25,28.68,30.21,31.82,33.53,35.36,37.31,
                             39.39,41.61,44.0,46.57,49.33,52.33,55.58,59.13,63.02,67.3,72.05,
                             77.37,83.36,90.19,98.08,107.3,118.4,132.1,148.8,169.9,197.7,237.2,
                             290.0,370.0,500.0,700.0,1000.0])
    
    bin2_num = len(var2_binning) -1
    eEvents = np.zeros((bin1_num, bin2_num))
    ccpEvents = np.zeros((bin1_num, bin2_num))
    
    
    for events in read_tree(filename, treename, chunk_size=chunk_size, rank=rank, nranks=nranks):
        
        events = ApplyPreselectionCuts(events)
        events = ApplySelectionCuts(events)
        events = CutEnergyOverRigidity(events)
    
        NegativRigidity = events.TrackerRigidity < 0  
        events = events[NegativRigidity]  
        
        #Electron sample 
        el_events = CutEcalChiSquareLateralNormalized(events)
        selection = el_events.EcalBDTv7_EnergyElectron_Smoothed > 0 
        el_events = el_events[selection]
        
        
        #Charged confused proton sample
        selection = events.EcalBDTv7_EnergyElectron_Smoothed < 0
        ccp_events = events[selection]
             
        
        ev1 = ak.to_numpy(el_events[var1_name])
        ev2 = ak.to_numpy(el_events[var2_name])
        
        ccpv1 = ak.to_numpy(ccp_events[var1_name])
        ccpv2 = ak.to_numpy(ccp_events[var2_name])


        el_hist_values, el_edges = np.histogramdd((ev1,ev2), bins= (var1_binning, var2_binning))        
        eEvents += el_hist_values
        
        ccp_hist_values, ccp_edges = np.histogramdd((ccpv1,ccpv2), bins= (var1_binning, var2_binning))        
        ccpEvents += ccp_hist_values

          
    # this process is done, save result
    np.savez(os.path.join(resultdir, f"results_{rank}.npz"), var1_binning=var1_binning,
             var2_binning=var2_binning, eEvents=eEvents, ccpEvents=ccpEvents)
    
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
    eEvents = None
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
                
            if eEvents is None:
                eEvents = result_file["eEvents"]
                ccpEvents = result_file["ccpEvents"]
                
            else:
                eEvents += result_file["eEvents"]
                ccpEvents += result_file["ccpEvents"]
        

    # now save merged result
    np.savez(os.path.join(args.resultdir, "results.npz"), var1_binning=var1_binning, var2_binning=var2_binning,  eEvents=eEvents, 
             ccpEvents=ccpEvents)

    
if __name__ == "__main__":
    main()