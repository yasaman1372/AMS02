#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:55:28 2023

@author: yasaman
"""


import multiprocessing as mp
import os
import numpy as np
from Cuts.ElectronTagCuts import *
from Cuts.ElectronIdentificationCuts import *
from Cuts.ElectronSelectionCuts import *
from Cuts.PreselectionCuts import *
from Cuts.ApplyCuts import *
from tools.roottree import read_tree

address='/home/op115134/Software/YasamanAnalysis/RootFiles/EnergydependentCuts/'
with uproot.open(address+'LeptonAnalysis_EcalBDTCutResults_Average.root') as file:
# print(list(file))
    cut = file['allTracksEcalBDTCutValueGraph']
    cutx = cut.members['fX']
    cuty = cut.members['fY']
    Ecal_BDT_cut = interpolate.interp1d(cutx,cuty,fill_value="extrapolate")

def handle_file(arg):
    # This function is called once for each process.
    # rank and nranks specifies which process out of how many processes this one is.
    # Based on that, read_tree knows which files to read.

    filename, treename, chunk_size, rank, nranks, kwargs = arg

    resultdir = kwargs["resultdir"]
    
    var1_name= 'TRD Estimator'
    var1_min = 0
    var1_max = 2
    bin1_num = 100
    var1_binning = np.linspace(var1_min, var1_max, bin1_num +1)
    
    var2_name = "EcalEnergyElectronNewMaximumShower" 

    var2_binning = np.array([0.5,0.65,0.82,1.01,1.22,1.46,1.72,2.0,2.31,2.65,3.0,3.36,3.73,
                             4.12,4.54,5.0,5.49,6.0,6.54,7.1,7.69,8.3,8.95,9.62,10.32,11.04,
                             11.8,12.59,13.41,14.25,15.14,16.05,17.0,17.98,18.99,20.04,21.13,
                             22.25,23.42,24.62,25.9,27.25,28.68,30.21,31.82,33.53,35.36,37.31,
                             39.39,41.61,44.0,46.57,49.33,52.33,55.58,59.13,63.02,67.3,72.05,
                             77.37,83.36,90.19,98.08,107.3,118.4,132.1,148.8,169.9,197.7,237.2,
                             290.0,370.0,500.0,700.0,1000.0])
    bin2_num = len(var2_binning) -1
    TCEvents = np.zeros((bin1_num,bin2_num))
    TCEvents_passed = np.zeros((bin1_num,bin2_num))
    
    var3_name= 'TofUpperCharge'
    var3_min = 0
    var3_max = 10
    bin3_num = 500
    var3_binning = np.linspace(var3_min, var3_max, bin3_num +1)
    Events = np.zeros((bin3_num,bin2_num))
    
    for events in read_tree(filename, treename, chunk_size=chunk_size, rank=rank, nranks=nranks):
        
        #Upper TOF charge
        TCevents = ApplyPreselectionCuts(events)
        TCevents = NegativeRigidityTag(TCevents)
        
        #selection cuts:
        TCevents = CutTrdActiveLayers(TCevents)
        TCevent = CutTrdNoHelium(TCevents)
        TCevents = CutTrackerPatternSortedByMDR(TCevents)
        TCevents = CutTrackerCharge(TCevents)
        Tcevents = CutTrackerChiSquareY(TCevents)
        
        #identification cuts:
        TCevents = CutEnergyOverRigidity(TCevents)
        TCevents = CutEcalChiSquareLateralNormalized(TCevents, "MC")
        TCevents = CutTrackerTrackEcalCogDeltaX(TCevents)
        TCevents = CutTrackerTrackEcalCogDeltaY(TCevents)   
        
        #other cuts:       
        #TCevents = GeomagneticCutoff(TCevents)
        EcalBDT = Ecal_BDT_cut(TCevents.EcalEnergyElectronNewMaximumShower)
        TCevents = TCevents[TCevents.EcalBDT_v7_EnergyD > EcalBDT ]
        # TCevents = EcalElectronTag(TCevents)
        # TCevents = TrdElectronTag(TCevents)
        # TCevents = TrdNoHeliumTag(TCevents)
        # TCevents = TofBetaTag(TCevents)
        # TCevents = TrackerChargeTag(TCevents)
        
        TCevents_passed = CutTofUpperCharge(TCevents)
        
        v1 = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(TCevents))
        v2 = ak.to_numpy(TCevents[var2_name])
        v3=ak.to_numpy(TCevents[var3_name])
        
        v1_passed = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(TCevents_passed))
        v2_passed = ak.to_numpy(TCevents_passed[var2_name])
        
        hist_passed, edges_passed = np.histogramdd((v1_passed,v2_passed), bins= (var1_binning,var2_binning), weights=ak.to_numpy(TCevents_passed.McEventWeightElectron))        
        TCEvents_passed += hist_passed    
        
        hist, edges = np.histogramdd((v1,v2), bins= (var1_binning,var2_binning),weights=ak.to_numpy(TCevents.McEventWeightElectron))         
        TCEvents += hist
        
        hist_values, hist_edges = np.histogramdd((v3,v2), bins= (var3_binning,var2_binning), weights=ak.to_numpy(TCevents.McEventWeightElectron))      
        Events += hist_values
        
        # this process is done, save result
    np.savez(os.path.join(resultdir, f"results_{rank}.npz"),TOF_charge_binning=var3_binning, TRD_Estimator_binning=var1_binning, Energy_binning = var2_binning, TCEvents = TCEvents, TCEvents_passed = TCEvents_passed, Events = Events)

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
        pool_args = make_args(args.filename, args.treename, args.chunk_size, args.nprocesses, resultdir=args.resultdir)
        # and execute handle_file for each process
        for _ in pool.imap_unordered(handle_file, pool_args):
            pass

    # now all processes are done, load the results and merge them
    Energy_binning  = None
    TRD_Estimator_binning = None
    TCEvents = None
    for rank in range(args.nprocesses):
        filename = os.path.join(args.resultdir, f"results_{rank}.npz")
        with np.load(filename) as result_file:
            if Energy_binning is None:
                Energy_binning = result_file["Energy_binning"]
                TRD_Estimator_binning = result_file["TRD_Estimator_binning"]
                TOF_charge_binning = result_file["TOF_charge_binning"]
            else:
                # make sure all processes used the same binning
                assert np.all(Energy_binning == result_file["Energy_binning"])
                #assert np.all(TRD_Estimator_binning == result_file["TRD_Estimator_binning"])
                             
            if TCEvents is None:
                
                TCEvents = result_file["TCEvents"]
                TCEvents_passed = result_file["TCEvents_passed"]
                Events = result_file["Events"]
                      
            else:

                TCEvents += result_file["TCEvents"]
                TCEvents_passed += result_file["TCEvents_passed"]
                Events += result_file["Events"]


    # now save merged result
    np.savez(os.path.join(args.resultdir, "Efficiency_corrections_uppertof_MC.npz"),TOF_charge_binning = TOF_charge_binning, TRD_Estimator_binning = TRD_Estimator_binning, Energy_binning = Energy_binning, TCEvents = TCEvents, TCEvents_passed = TCEvents_passed, Events = Events)   
    
        
if __name__ == "__main__":
    main()        
        
