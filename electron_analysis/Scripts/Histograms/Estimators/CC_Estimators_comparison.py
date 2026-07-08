#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 09:43:54 2023

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
Hist_branches = ["TotalEnergy3D","EcalBDT3D","CCBDT",'ElectronCCMVABDT',"TrackerTrackGBLMaxSpanRigidity","EcalEnergyElectronNewMaximumShower", "TrdPLikelihoodHybridHitsElectronECAL", 
                 "TrdPLikelihoodHybridHitsProtonECAL",
                  "EcalBDT_v7_EnergyD"]

Branches = Preselection_Branches + Selection_Cuts_Branches + Identification_Cuts_Branches + Hist_branches

def handle_file(arg):
    # This function is called once for each process.
    # rank and nranks specifies which process out of how many processes this one is.
    # Based on that, read_tree knows which files to read.

    filename, treename, chunk_size, rank, nranks, kwargs = arg
   # data_type="ISS"

    resultdir = kwargs["resultdir"]
    data_type = kwargs["data_type"]


    var1_name= 'TrdEstimator'
    var1_min = 0
    var1_max = 2
    bin1_num = 100
    L_TRD_binning = np.linspace(var1_min, var1_max, bin1_num +1)
        
    var2_name='ElectronCCMVABDT'
    var2_min = -1
    var2_max = 1
    bin2_num = 100
    CCMVA_binning = np.linspace(var2_min, var2_max, bin2_num +1)
    
    var3_name='CCBDT'
    var3_min = -1
    var3_max = 1
    bin3_num = 100
    CCBDT_binning = np.linspace(var3_min, var3_max, bin3_num +1)
  
    var4_name = "TotalEnergy3D" 
    #var3_binning = np.array([20,50])
    Energy_binning = np.array([0.5,0.65,0.82,1.01,1.22,1.46,1.72,2.0,2.31,2.65,3.0,3.36,3.73,
     4.12,4.54,5.0,5.49,6.0,6.54,7.1,7.69,8.3,8.95,9.62,10.32,11.04,
     11.8,12.59,13.41,14.25,15.14,16.05,17.0,17.98,18.99,20.04,21.13,
     22.25,23.42,24.62,25.9,27.25,28.68,30.21,31.82,33.53,35.36,37.31,
     39.39,41.61,44.0,46.57,49.33,52.33,55.58,59.13,63.02,67.3,72.05,
     77.37,83.36,90.19,98.08,107.3,118.4,132.1,148.8,169.9,197.7,237.2,
     290.0,370.0,500.0,700.0,1000.0])
    
    bin4_num = len(var4_binning) -1
    
    
    var5_name= 'Rigidity sign'
    Rigidity_binning = np.array([-1.5,0,1.5])
    bin5_num = len(var5_binning) -1
    
  
    Events_st_CCBDT = np.zeros((bin1_num, bin3_num, bin4_num, bin5_num))
    # Events_mt_CCBDT = np.zeros((bin1_num, bin3_num, bin4_num, bin5_num))
    # Events_all_CCBDT= np.zeros((bin1_num, bin3_num, bin4_num, bin5_num))
    
    
    Events_st_CCMVA = np.zeros((bin1_num, bin2_num, bin4_num, bin5_num))
    # Events_mt_CCMVA = np.zeros((bin1_num, bin2_num, bin4_num, bin5_num))
    # Events_all_CCMVA = np.zeros((bin1_num, bin2_num, bin4_num, bin5_num))
    
    for events in read_tree(filename, treename, chunk_size=chunk_size, rank=rank, nranks=nranks):
        
        events = ApplyPreselectionCuts(events) #preselection cuts are applied
        events = ApplySelectionCuts(events)
        events = ApplyIdentificationCuts(events)
        events = GeomagneticCutoff(events)
        events = events[events.EcalBDT3D > -0.75 ]
        events_st = events[events.TrackerNumberOfTracks == 1]
        # events_mt = events[events.TrackerNumberOfTracks > 1]
        # events_all = events[events.TrackerNumberOfTracks >= 1]        

        #single track events 
        L_TRD = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(events_st))
        CCMVA = ak.to_numpy(events_st[var2_name])
        CCBDT = ak.to_numpy(events_st[var3_name])
        E = ak.to_numpy(events_st[var4_name])
        R = np.sign(ak.to_numpy(events_st["TrackerTrackGBLMaxSpanRigidity"]))
        
        # #multi track events
        # v1_mt = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(events_mt))
        # v2_mt = ak.to_numpy(events_mt[var2_name])
        # v3_mt = ak.to_numpy(events_mt[var3_name])  
        # v4_mt = np.sign(ak.to_numpy(events_mt["TrackerTrackGBLMaxSpanRigidity"]))
        
        # #all track events
        # v1_all = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(events_all))
        # v2_all = ak.to_numpy(events_all[var2_name])
        # v3_all = ak.to_numpy(events_all[var3_name])  
        # v4_all = np.sign(ak.to_numpy(events_all["TrackerTrackGBLMaxSpanRigidity"]))

        hist_values_st_CCMVA, edges_st_CCMVA = np.histogramdd((L_TRD,CCMVA,E,R), bins= (L_TRD_binning, CCMVA_binning, Energy_binning, Rigidity_binning))        
        Events_st_CCMVA += hist_values_st_CCMVA
        
        hist_values_st_CCBDT, edges_st_CCBDT = np.histogramdd((L_TRD,CCBDT,E,R), bins= (L_TRD_binning, CCBDT_binning, Energy_binning, Rigidity_binning))        
        Events_st_CCBDT += hist_values_st_CCBDT
        
        
        # hist_values_mt, edges_mt = np.histogramdd((v1_mt,v2_mt,v3_mt,v4_mt), bins= (var1_binning, var2_binning, var3_binning, var4_binning))        
        # Events_mt += hist_values_mt
        
        
        # hist_values_all, edges_all = np.histogramdd((v1_all,v2_all,v3_all,v4_all), bins= (var1_binning, var2_binning, var3_binning, var4_binning))        
        # Events_all += hist_values_all
        
        
          
        

    # this process is done, save result
    np.savez(os.path.join(resultdir, f"results_{rank}.npz"), L_TRD_binning = L_TRD_binning , CCMVA_binning = CCMVA_binning, CCBDT_binning = CCBDT_binning, Energy_binning = Energy_binning, 
             Rigidity_binning=Rigidity_binning, Events_st_CCMVA = Events_st_CCMVA, Events_st_CCBDT=Events_st_CCBDT)
    
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
        pool_args = make_args(args.filename, args.treename, args.chunk_size, args.nprocesses, resultdir=args.resultdir, data_type=args.data_type)
        # and execute handle_file for each process
        for _ in pool.imap_unordered(handle_file, pool_args):
            pass

    # now all processes are done, load the results and merge them
    L_TRD_binning = None
    CCMVA_binning = None
    CCBDT_binning = None
    Energy_binning = None
    Rigidity_binning = None
    Events_st_CCBDT = None
    Events_all_CCMVA = None

    for rank in range(args.nprocesses):
        filename = os.path.join(args.resultdir, f"results_{rank}.npz")
        with np.load(filename) as result_file:
            if var1_binning is None or var2_binning is None or var3_binning is None:
                L_TRD_binning = result_file["L_TRD_binning "]
                CCMVA_binning = result_file["CCMVA_binning"]
                CCBDT_binning = result_file["CCBDT_binning"]
                Energy_binning = result_file["Energy_binning"]
                Rigidity_binning = result_file["Rigidity_binning"]
            else:
                # make sure all processes used the same binning
                assert np.all(L_TRD_binning == result_file["L_TRD_binning"])
                assert np.all(CCMVA_binning == result_file["CCMVA_binning"])
                assert np.all(CCBDT_binning == result_file["CCBDT_binning"])
                assert np.all(Energy_binning == result_file["Energy_binning"])
                assert np.all(Rigidity_binning == result_file["Rigidity_binning"])
                
            if Events_st_CCBDT is None or Events_all_CCMVA is None:
                Events_st_CCBDT = result_file["Events_st_CCBDT"]
                Events_st_CCMVA = result_file["Events_st_CCMVA"]
                # Events_all = result_file["Events_all"]
            else:
                Events_st_CCBDT += result_file["Events_st_CCBDT"]
                Events_st_CCMVA +=result_file["Events_st_CCMVA"]
                #Events_all += result_file["Events_all"]

    # now save merged result
    np.savez(os.path.join(args.resultdir, "results.npz"), L_TRD_binning=L_TRD_binning, CCMVA_binning = CCMVA_binning, CCBDT_binning = CCBDT_binning, Energy_binning=Energy_binning,
             Rigidity_binning = Rigidity_binning, Events_st_CCBDT = Events_st_CCBDT , Events_st_CCMVA = Events_st_CCMVA)
    
    for e in range(len(Energy_binning) -1):
        
        a = var3_binning[binn]
        b = var3_binning[binn+1]
        
        CCBDT_sig = np.zeros((len(CCBDT_binning)))
        CCMVA_sig = np.zeros((len(CCMVA_binning)))
        
        CCBDT_bkg = np.zeros((len(CCBDT_binning)))
        CCMVA_bkg = np.zeros((len(CCMVA_binning)))
        
        CCBDT_data_negative_st = Events_st_CCBDT[:,:,e,0].sum(axis=0)
        CCBDT_data_positive_st = Events_st_CCBDT[:,:,e,1].sum(axis=0)
        
        CCMVA_data_negative_st = Events_st_CCMVA[:,:,e,0].sum(axis=0)
        CCMVA_data_positive_st = Events_st_CCMVA[:,:,e,1].sum(axis=0)
        
        for c in range(len(CCMVA_binning)):
            
            CCBDT_sig_ratio = CCBDT_data_negative_st[:,:c].sum(axis=1)/CCBDT_data_negative_st[:,c:].sum(axis=1)
            CCBDT_bkg_ratio = CCBDT_data_positive_st[:,:c].sum(axis=1)/CCBDT_data_positive_st[:,c:].sum(axis=1)
            
            CCMVA_sig_ratio = CCMVA_data_negative_st[:,:c].sum(axis=1)/CCMVA_data_negative_st[:,c:].sum(axis=1)
            CCMVA_bkg_ratio = CCMVA_data_positive_st[:,:c].sum(axis=1)/CCMVA_data_positive_st[:,c:].sum(axis=1)
            
            CCBDT_sig[c] = CCBDT_sig_ratio
            CCBDT_bkg[c] = CCBDT_bkg_ratio
            
            CCMVA_sig[c] = CCMVA_sig_ratio
            CCMVA_bkg[c] = CCMVA_bkg_ratio
            

        
        figure = plt.figure(figsize=(12, 10))
        plot = figure.subplots(1, 1)
        plot.set_xlabel('sig ratio',fontsize = 26)
        plot.set_ylabel("bkg ratio", fontsize = 26)
        plot.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
        plot.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.minorticks_on()
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        for axis in ['top','bottom','left','right']:
            plot.spines[axis].set_linewidth(2)
        
        
        plt.plot(CCBDT_sig_ratio, CCBDT_bkg_ratio, label="CCBDT")
        plt.plot(CCMVA_sig_ratio, CCMVA_bkg_ratio, label="CCMVA")

        
    
        plot.plot([],[],' ',label=args.data_type + " pass 8")
        plot.plot([],[],' ',label="Energy range =["+ str(a) +", "+str(b)+"]"+" Gev")
        plt.legend(fontsize=12,frameon=True,labelcolor='k')
        
        
    
        figure.savefig("CC_estimator_performance_comparision"+str(e)+".pdf" , dpi=250)
     
        
if __name__ == "__main__":
    main()             
        
        
        
        
    
        
        
        
        
        
        
        