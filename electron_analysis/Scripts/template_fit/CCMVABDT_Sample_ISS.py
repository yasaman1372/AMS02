#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 16:21:04 2023

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

from Cuts.ElectronTagCuts import *
from Cuts.ElectronIdentificationCuts import *
from Cuts.ApplyCuts import *
from tools.roottree import read_tree

   

Preselection_Branches = Preselection_Cuts_Branch_List()
Selection_Cuts_Branches = Selection_Cuts_Branch_List()
Identification_Cuts_Branches = Identification_Cuts_Branch_List() 
Hist_branches = ["CCBDT","TotalEnergy3D","EcalBDT3D","ElectronCCMVABDT","Stoermer","Time","EcalBDT_v7_EnergyD","TrackerTrackGBLMaxSpanRigidity","EcalEnergyElectronNewMaximumShower", "TrdPLikelihoodHybridHitsElectronECAL", 
                 "TrdPLikelihoodHybridHitsProtonECAL", "TrackerNumberOfTracks", "EcalChiSquareLateral"]

Branches = Preselection_Branches + Selection_Cuts_Branches + Identification_Cuts_Branches + Hist_branches

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

    var1_name= 'CCBDT'
    var1_min = -1
    var1_max = 1
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
    
    
    eEvents_st = np.zeros((bin1_num, bin2_num))
    pEvents_st = np.zeros((bin1_num, bin2_num)) 
    ccpEvents_st = np.zeros((bin1_num, bin2_num))
    
    eEvents_mt = np.zeros((bin1_num, bin2_num))
    pEvents_mt = np.zeros((bin1_num, bin2_num)) 
    ccpEvents_mt = np.zeros((bin1_num, bin2_num))
    
       
    for events in read_tree(filename, treename, chunk_size=chunk_size, rank=rank, nranks=nranks, branches=Branches):
        
        events = ApplyPreselectionCuts(events) #preselection cuts are applied
        events = ApplySelectionCuts(events)
        events = CutEnergyOverRigidity(events)
        #events = ApplyIdentificationCuts(events,"ISS")
        events = GeomagneticCutoff(events)
        
        events_st = events[events.TrackerNumberOfTracks == 1] #single track events
        events_mt = events[events.TrackerNumberOfTracks > 1] #multi track events
 
        #electron template selection
        
        #single track
        el_events_st = events_st[events_st.TrackerTrackGBLMaxSpanRigidity < 0]
        el_events_st = CutEcalChiSquareLateralNormalized(el_events_st, "ISS")
        #EcalBDT = Ecal_BDT_cut(el_events.EcalEnergyElectronNewMaximumShower)
        #el_events_st = el_events_st[el_events_st.EcalBDT_v7_EnergyD > 0 ]
        el_events_st = el_events_st[el_events_st.EcalBDT3D > -0.75 ]
        el_events_st = el_events_st[TrdLRElecProt_Energy_HybridHits_TrdP(el_events_st) < 0.75]
        
        #multi track
        el_events_mt = events_mt[events_mt.TrackerTrackGBLMaxSpanRigidity < 0]
        el_events_mt = CutEcalChiSquareLateralNormalized(el_events_mt, "ISS")
        # el_events_mt = el_events_mt[el_events_mt.EcalBDT_v7_EnergyD > 0 ]
        el_events_mt = el_events_mt[el_events_mt.EcalBDT3D > -0.75 ]
        el_events_mt = el_events_mt[TrdLRElecProt_Energy_HybridHits_TrdP(el_events_mt) < 0.75]
        
        
        #proton sample
        
        #single track
        p_events_st = events_st[events_st.TrackerTrackGBLMaxSpanRigidity > 0]
        #p_events_st = p_events_st[p_events_st.EcalBDT_v7_EnergyD < 0 ]
        p_events_st = p_events_st[p_events_st.EcalBDT3D < -0.75 ]
        p_events_st = p_events_st[TrdLRElecProt_Energy_HybridHits_TrdP(p_events_st) > 0.8]
        
        
        #multi tracks
        p_events_mt = events_mt[events_mt.TrackerTrackGBLMaxSpanRigidity > 0]
        #p_events_mt = p_events_mt[p_events_mt.EcalBDT_v7_EnergyD < 0 ]
        p_events_mt = p_events_mt[p_events_mt.EcalBDT3D < -0.75 ]
        p_events_mt = p_events_mt[TrdLRElecProt_Energy_HybridHits_TrdP(p_events_mt) > 0.8]
        
        
        #charged confused poroton template selection
        
        #single track
        ccp_events_st = events_st[events_st.TrackerTrackGBLMaxSpanRigidity < 0]
        # ccp_events_st = ccp_events_st[ccp_events_st.EcalBDT_v7_EnergyD < 0 ]
        ccp_events_st = ccp_events_st[ccp_events_st.EcalBDT3D < - 0.75 ]
        ccp_events_st = ccp_events_st[TrdLRElecProt_Energy_HybridHits_TrdP(ccp_events_st) > 0.8]
        
        #multi track
        ccp_events_mt = events_mt[events_mt.TrackerTrackGBLMaxSpanRigidity < 0]
        # ccp_events_mt = ccp_events_mt[ccp_events_mt.EcalBDT_v7_EnergyD < 0 ]
        ccp_events_mt = ccp_events_mt[ccp_events_mt.EcalBDT3D < - 0.75 ]
        ccp_events_mt = ccp_events_mt[TrdLRElecProt_Energy_HybridHits_TrdP(ccp_events_mt) > 0.8]
        
        #single track       
        ev1_st = ak.to_numpy(el_events_st[var1_name])
        ev2_st = ak.to_numpy(el_events_st[var2_name])
        
        pv1_st = ak.to_numpy(p_events_st[var1_name])
        pv2_st = ak.to_numpy(p_events_st[var2_name])
        
        ccpv1_st = ak.to_numpy(ccp_events_st[var1_name])
        ccpv2_st = ak.to_numpy(ccp_events_st[var2_name])
        
        #multi track
        ev1_mt = ak.to_numpy(el_events_mt[var1_name])
        ev2_mt = ak.to_numpy(el_events_mt[var2_name])
        
        pv1_mt = ak.to_numpy(p_events_mt[var1_name])
        pv2_mt = ak.to_numpy(p_events_mt[var2_name])
        
        ccpv1_mt = ak.to_numpy(ccp_events_mt[var1_name])
        ccpv2_mt = ak.to_numpy(ccp_events_mt[var2_name])
        
        #single track 
        ehist_values_st, eedges_st = np.histogramdd((ev1_st,ev2_st), bins= (var1_binning, var2_binning))        
        eEvents_st += ehist_values_st
        
        phist_values_st, pedges_st = np.histogramdd((pv1_st,pv2_st), bins= (var1_binning, var2_binning))        
        pEvents_st += phist_values_st
        
        ccphist_values_st, ccpedges_st = np.histogramdd((ccpv1_st,ccpv2_st), bins= (var1_binning, var2_binning))        
        ccpEvents_st += ccphist_values_st
        
        #multi track
        ehist_values_mt, eedges_mt = np.histogramdd((ev1_mt,ev2_mt), bins= (var1_binning, var2_binning))        
        eEvents_mt += ehist_values_mt
        
        phist_values_mt, pedges_mt = np.histogramdd((pv1_mt,pv2_mt), bins= (var1_binning, var2_binning))        
        pEvents_mt += phist_values_mt
        
        ccphist_values_mt, ccpedges_mt = np.histogramdd((ccpv1_mt,ccpv2_mt), bins= (var1_binning, var2_binning))        
        ccpEvents_mt += ccphist_values_mt
        
    # this process is done, save result
    np.savez(os.path.join(resultdir, f"results_{rank}.npz"), var1_binning=var1_binning,
             var2_binning=var2_binning, eEvents_st=eEvents_st, pEvents_st=pEvents_st, ccpEvents_st=ccpEvents_st,
             eEvents_mt=eEvents_mt, pEvents_mt=pEvents_mt, ccpEvents_mt=ccpEvents_mt)
    
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
    var1_binning = None
    var2_binning = None
    eEvents_st = None
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
                
            if eEvents_st is None:
                
                eEvents_st = result_file["eEvents_st"]
                pEvents_st = result_file["pEvents_st"]
                ccpEvents_st = result_file["ccpEvents_st"]
                
                eEvents_mt = result_file["eEvents_mt"]
                pEvents_mt = result_file["pEvents_mt"]
                ccpEvents_mt = result_file["ccpEvents_mt"]
                
            else:
                
                eEvents_st += result_file["eEvents_st"]
                pEvents_st = result_file["pEvents_st"]
                ccpEvents_st += result_file["ccpEvents_st"]
                
                eEvents_mt += result_file["eEvents_mt"]
                pEvents_mt = result_file["pEvents_mt"]
                ccpEvents_mt += result_file["ccpEvents_mt"]

    # now save merged result
    np.savez(os.path.join(args.resultdir, "CCMVABDT_template_ISS.npz"), var1_binning=var1_binning, var2_binning=var2_binning,  eEvents_st=eEvents_st, pEvents_st=pEvents_st,
             ccpEvents_st=ccpEvents_st, eEvents_mt=eEvents_mt, pEvents_mt=pEvents_mt,
                      ccpEvents_mt=ccpEvents_mt)


                
    
if __name__ == "__main__":
    main()
