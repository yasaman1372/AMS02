#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 12:12:33 2022

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
from Cuts.ElectronIdentificationCuts import CutEcalChiSquareLateralNormalized
from Cuts.ApplyCuts import *
from tools.roottree import read_tree

   

Preselection_Branches = Preselection_Cuts_Branch_List()
Selection_Cuts_Branches = Selection_Cuts_Branch_List()
Identification_Cuts_Branches = Identification_Cuts_Branch_List() 
Hist_branches = ["EcalBDT3D","Stoermer","Time","EcalBDT_v7_EnergyD","TrackerTrackGBLMaxSpanRigidity","EcalEnergyElectronNewMaximumShower", "TrdPLikelihoodHybridHitsElectronECAL", 
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

    var1_name= 'TRD Estimator'
    var1_min = 0
    var1_max = 2
    bin1_num = 100
    
    var2_name = "EcalEnergyElectronNewMaximumShower" 

    
    var1_binning = np.linspace(var1_min, var1_max, bin1_num +1)
    var2_binning = np.array([0.5,0.65,0.82,1.01,1.22,1.46,1.72,2.0,2.31,2.65,3.0,3.36,3.73,
                             4.12,4.54,5.0,5.49,6.0,6.54,7.1,7.69,8.3,8.95,9.62,10.32,11.04,
                             11.8,12.59,13.41,14.25,15.14,16.05,17.0,17.98,18.99,20.04,21.13,
                             22.25,23.42,24.62,25.9,27.25,28.68,30.21,31.82,33.53,35.36,37.31,
                             39.39,41.61,44.0,46.57,49.33,52.33,55.58,59.13,63.02,67.3,72.05,
                             77.37,83.36,90.19,98.08,107.3,118.4,132.1,148.8,169.9,197.7,237.2,
                             290.0,370.0,500.0,700.0,1000.0])
    
    bin2_num = len(var2_binning) -1
    eEvents_st_passed = np.zeros((bin1_num, bin2_num))  
    eEvents_st_failed = np.zeros((bin1_num, bin2_num))

    
    pEvents_st = np.zeros((bin1_num, bin2_num))
    ccpEvents_st = np.zeros((bin1_num, bin2_num))
    
    
    eEvents_mt_passed = np.zeros((bin1_num, bin2_num))
    eEvents_mt_failed = np.zeros((bin1_num, bin2_num))
    
    pEvents_mt = np.zeros((bin1_num, bin2_num))
    ccpEvents_mt = np.zeros((bin1_num, bin2_num))
   
    
    
    for events in read_tree(filename, treename, chunk_size=chunk_size, rank=rank, nranks=nranks, branches=Branches):
        
        events = ApplyPreselectionCuts(events) 
        events = ApplySelectionCuts(events)
        events = CutEnergyOverRigidity(events)
        events = GeomagneticCutoff(events)
        events_st = events[events.TrackerNumberOfTracks == 1]
        events_mt = events[events.TrackerNumberOfTracks > 1]
        
 
        #electron template sample selection single track
        el_events_st = events_st[events_st.TrackerTrackGBLMaxSpanRigidity < 0]
        #el_events_st = CutEcalChiSquareLateralNormalized(el_events_st, "ISS")
        #EcalBDT = Ecal_BDT_cut(el_events_st.EcalEnergyElectronNewMaximumShower)
        #el_events_st_passed = el_events_st[el_events_st.EcalBDT_v7_EnergyD > EcalBDT ] #events that pass the ecal cut
        #el_events_st_failed = el_events_st[el_events_st.EcalBDT_v7_EnergyD <= EcalBDT ] #events that faile the ecal cut
        el_events_st_passed = el_events_st[el_events_st.EcalBDT3D > -0.75 ]
        el_events_st_failed = el_events_st[el_events_st.EcalBDT3D <= -0.75 ]
        
        
        
        #charged confused poroton sample template selection single track
        ccp_events_st = events_st[events_st.TrackerTrackGBLMaxSpanRigidity < 0]
        #EcalBDT = Ecal_BDT_cut(ccp_events_st.EcalEnergyElectronNewMaximumShower)
        #ccp_events_st = ccp_events_st[ccp_events_st.EcalBDT_v7_EnergyD < EcalBDT] #events that pass the Ecal cut
        ccp_events_st = ccp_events_st[ccp_events_st.EcalBDT3D < -0.75]
        
        

        
        #poroton template sample selection single track
        p_events_st = events_st[events_st.TrackerTrackGBLMaxSpanRigidity > 0]
        # EcalBDT = Ecal_BDT_cut(p_events_st.EcalEnergyElectronNewMaximumShower)
        # p_events_st = p_events_st[p_events_st.EcalBDT_v7_EnergyD < EcalBDT]
        p_events_st = p_events_st[p_events_st.EcalBDT3D < -0.75]
        
        
           
        #the single/multiple passed/failed eevents in the Ecal cut, are used the make the Energy/TRD estimator data for the histograms
        ev1_st_passed = TrdLRElecProt_Energy_HybridHits_TrdP(el_events_st_passed)
        ev2_st_passed = ak.to_numpy(el_events_st_passed[var2_name])
        
        ev1_st_failed = TrdLRElecProt_Energy_HybridHits_TrdP(el_events_st_failed)
        ev2_st_failed = ak.to_numpy(el_events_st_failed[var2_name])
        
        ccpv1_st = TrdLRElecProt_Energy_HybridHits_TrdP(ccp_events_st)
        ccpv2_st = ak.to_numpy(ccp_events_st[var2_name])
    
        
        pv1_st = TrdLRElecProt_Energy_HybridHits_TrdP(p_events_st)
        pv2_st =ak.to_numpy(p_events_st[var2_name])
        

        #the TRD estimatpr histograms are made
        ehist_values_st_passed, eedges_st_passed = np.histogramdd((ev1_st_passed,ev2_st_passed), bins= (var1_binning, var2_binning))        
        eEvents_st_passed += ehist_values_st_passed
        
        ehist_values_st_failed, eedges_st_failed = np.histogramdd((ev1_st_failed,ev2_st_failed), bins= (var1_binning, var2_binning))        
        eEvents_st_failed += ehist_values_st_failed
        
        ccphist_values_st, ccpedges_st = np.histogramdd((ccpv1_st,ccpv2_st), bins= (var1_binning, var2_binning))        
        ccpEvents_st += ccphist_values_st

        
        phist_values_st, pedges_st = np.histogramdd((pv1_st,pv2_st), bins= (var1_binning, var2_binning))        
        pEvents_st += phist_values_st
        
        
        #electron template sample selection multi track
        el_events_mt = events_mt[events_mt.TrackerTrackGBLMaxSpanRigidity < 0]
        #el_events_mt = CutEcalChiSquareLateralNormalized(el_events_mt, "ISS")
        el_events_mt_passed = el_events_mt[el_events_mt.EcalBDT3D > -0.75 ]
        el_events_mt_failed = el_events_mt[el_events_mt.EcalBDT3D <= -0.75 ]
        # EcalBDT = Ecal_BDT_cut(el_events_mt.EcalEnergyElectronNewMaximumShower)
        # el_events_mt_passed = el_events_mt[el_events_mt.EcalBDT_v7_EnergyD > EcalBDT ] #events that pass the ecal cut
        # el_events_mt_failed = el_events_mt[el_events_mt.EcalBDT_v7_EnergyD <= EcalBDT ] #events that fail the ecal cut
        
        
        
        #charged confused poroton sample template selection multi track
        ccp_events_mt = events_mt[events_mt.TrackerTrackGBLMaxSpanRigidity < 0]
        ccp_events_mt = ccp_events_mt[ccp_events_mt.EcalBDT3D < -0.75]
        # EcalBDT = Ecal_BDT_cut(ccp_events_mt.EcalEnergyElectronNewMaximumShower)
        # ccp_events_mt = ccp_events_mt[ccp_events_mt.EcalBDT_v7_EnergyD < EcalBDT] #events that pass the ecal cut
        
        #poroton sample template selection multi track
        p_events_mt = events_mt[events_mt.TrackerTrackGBLMaxSpanRigidity > 0]
        p_events_mt = p_events_mt[p_events_mt.EcalBDT3D < -0.75]
        # EcalBDT = Ecal_BDT_cut(p_events_mt.EcalEnergyElectronNewMaximumShower)
        # p_events_mt = p_events_mt[p_events_mt.EcalBDT_v7_EnergyD < EcalBDT]
        
               
        ev1_mt_passed = TrdLRElecProt_Energy_HybridHits_TrdP(el_events_mt_passed)
        ev2_mt_passed = ak.to_numpy(el_events_mt_passed[var2_name])
        
        ev1_mt_failed = TrdLRElecProt_Energy_HybridHits_TrdP(el_events_mt_failed)
        ev2_mt_failed = ak.to_numpy(el_events_mt_failed[var2_name])
        
        ccpv1_mt = TrdLRElecProt_Energy_HybridHits_TrdP(ccp_events_mt)
        ccpv2_mt = ak.to_numpy(ccp_events_mt[var2_name])
        
        
        pv1_mt = TrdLRElecProt_Energy_HybridHits_TrdP(p_events_mt)
        pv2_mt =ak.to_numpy(p_events_mt[var2_name])
        

        ehist_values_mt_passed, eedges_mt_passed = np.histogramdd((ev1_mt_passed,ev2_mt_passed), bins= (var1_binning, var2_binning))        
        eEvents_mt_passed += ehist_values_mt_passed
        
        ehist_values_mt_failed, eedges_mt_failed = np.histogramdd((ev1_mt_failed,ev2_mt_failed), bins= (var1_binning, var2_binning))        
        eEvents_mt_failed += ehist_values_mt_failed
        
        ccphist_values_mt, ccpedges_mt= np.histogramdd((ccpv1_mt,ccpv2_mt), bins= (var1_binning, var2_binning))        
        ccpEvents_mt += ccphist_values_mt
        
        phist_values_mt, pedges_mt = np.histogramdd((pv1_mt,pv2_mt), bins= (var1_binning, var2_binning))        
        pEvents_mt += phist_values_mt
        
        
    # this process is done, save result
    np.savez(os.path.join(resultdir, f"results_{rank}.npz"), var1_binning=var1_binning,
             var2_binning=var2_binning, eEvents_st_passed = eEvents_st_passed, eEvents_st_failed = eEvents_st_failed,
             ccpEvents_st = ccpEvents_st, pEvents_st=pEvents_st,
             eEvents_mt_passed = eEvents_mt_passed, eEvents_mt_failed = eEvents_mt_failed, ccpEvents_mt = ccpEvents_mt, 
             pEvents_mt=pEvents_mt )
    
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
    eEvents_st_passed = None
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
                
            if eEvents_st_passed is None:
                eEvents_st_passed = result_file["eEvents_st_passed"]
                eEvents_st_failed = result_file["eEvents_st_failed"]
                
                ccpEvents_st = result_file["ccpEvents_st"]          
                pEvents_st = result_file["pEvents_st"]
                
                eEvents_mt_passed = result_file["eEvents_mt_passed"]
                eEvents_mt_failed = result_file["eEvents_mt_failed"]
                
                ccpEvents_mt = result_file["ccpEvents_mt"]
                pEvents_mt = result_file["pEvents_mt"]
            else:
                eEvents_st_passed += result_file["eEvents_st_passed"]
                eEvents_st_failed += result_file["eEvents_st_failed"] 
                ccpEvents_st += result_file["ccpEvents_st"]
                pEvents_st += result_file["pEvents_st"]
                
                eEvents_mt_passed += result_file["eEvents_mt_passed"]
                eEvents_mt_failed += result_file["eEvents_mt_failed"]
                ccpEvents_mt += result_file["ccpEvents_mt"]
                pEvents_mt += result_file["pEvents_mt"]

    # now save merged result
    np.savez(os.path.join(args.resultdir, "TRD_templatefit_data.npz"), var1_binning=var1_binning, var2_binning=var2_binning,  eEvents_st_passed = eEvents_st_passed, 
            eEvents_st_failed = eEvents_st_failed, ccpEvents_st = ccpEvents_st, pEvents_st = pEvents_st,
            eEvents_mt_passed = eEvents_mt_passed, eEvents_mt_failed = eEvents_mt_failed ,ccpEvents_mt = ccpEvents_mt, 
            pEvents_mt = pEvents_mt )


                
    
if __name__ == "__main__":
    main()
