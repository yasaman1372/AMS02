#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:12:38 2022

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
Hist_branches = ["TotalEnergy3D",'EcalBDT3D','EcalBDTSmoothed_v7_EnergyD',"TrackerTrackGBLMaxSpanRigidity","EcalEnergyElectronNewMaximumShower", "TrdPLikelihoodHybridHitsElectronECAL", 
                 "TrdPLikelihoodHybridHitsProtonECAL"]

Branches = Preselection_Branches + Selection_Cuts_Branches + Identification_Cuts_Branches + Hist_branches


def handle_file(arg):
    # This function is called once for each process.
    # rank and nranks specifies which process out of how many processes this one is.
    # Based on that, read_tree knows which files to read.

    filename, treename, chunk_size, rank, nranks, kwargs = arg
    data_type= kwargs["data_type"]
    resultdir = kwargs["resultdir"]

    var1_name= 'TrdEstimator'
    var1_min = 0
    var1_max = 2
    bin1_num = 100
    var1_binning = np.linspace(var1_min, var1_max, bin1_num +1)

    var2_name='EcalBDT3D'
    var2_min = -1
    var2_max = 1
    bin2_num = 100
    var2_binning = np.linspace(var2_min, var2_max, bin2_num +1)
       

    #var2_name='EcalBDTSmoothed_v7_EnergyD'
    var2_name='EcalBDT3D'
    trans_var2_min = -20
    trans_var2_max = 20
    bin2_num = 100
    trans_var2_binning = np.linspace(trans_var2_min, trans_var2_max, bin2_num +1)
  
    var3_name = "TotalEnergy3D" 
    var3_binning = np.array([20,50])
    bin3_num = len(var3_binning) -1
  
    Negative_Events = np.zeros((bin1_num, bin2_num, bin3_num))
    Positive_Events = np.zeros((bin1_num, bin2_num, bin3_num))
    
    Negative_Events_trans = np.zeros((bin1_num, bin2_num, bin3_num))
    Positive_Events_trans = np.zeros((bin1_num, bin2_num, bin3_num))
    
    for events in read_tree(filename, treename, chunk_size=chunk_size, rank=rank, nranks=nranks):
        
       # print(np.max(ak.to_numpy(events)),flush=True)
        events = ApplyPreselectionCuts(events) #preselection cuts are applied
       # events = ApplySelectionCuts(events)
        #selection Cuts:
        events = CutTofBeta(events) #TOF velocity 
        events = CutTofUpperCharge(events) #Upper TOF charge
        events = CutTrdActiveLayers(events) #Enough Active Layers in TRD
        events = CutTrdNoHelium(events) #TRD Helium rejection
        events = CutTrackerPatternSortedByMDR(events) #Tracker hit pattern
        events = CutTrackerCharge(events) #Inner Tracker charge
        events = CutTrackerChiSquareY(events) #Tracker track goodness of fit in Y project
        events = TimeRange(events) 
        
        events = ApplyIdentificationCuts(events)
        events =  GeomagneticCutoff(events)
       # print(np.max(events),flush=True)

        Negative_events = events[events.TrackerTrackGBLMaxSpanRigidity < 0]
        Positive_events = events[events.TrackerTrackGBLMaxSpanRigidity > 0]
        
        nv1 =ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(Negative_events))
        nv2=ak.to_numpy(Negative_events[var2_name])
        trans_nv2 = np.log((1+nv2)/(1-nv2))
        nv3=ak.to_numpy(Negative_events[var3_name])

       # print(nv1,flush=True)
       # print(nv2,flush=True)
       # print(nv3,flush=True)


        Negative_hist_values, Negative_edges = np.histogramdd((nv1,nv2,nv3), bins= (var1_binning, var2_binning, var3_binning))        
        Negative_Events += Negative_hist_values
        
        Negative_hist_values_trans, Negative_edges_trans = np.histogramdd((nv1,trans_nv2,nv3), bins= (var1_binning, trans_var2_binning, var3_binning))
        Negative_Events_trans += Negative_hist_values_trans
        #print(Negative_Events,flush=True)

        pv1=ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(Positive_events))
        pv2=ak.to_numpy(Positive_events[var2_name])
        trans_pv2 = np.log((1+pv2)/(1-pv2))
        pv3=ak.to_numpy(Positive_events[var3_name])        

        Positive_hist_values, Positive_edges = np.histogramdd((pv1,pv2,pv3), bins= (var1_binning, var2_binning, var3_binning))        
        Positive_Events += Positive_hist_values

        Positive_hist_values_trans, Positive_edges_trans = np.histogramdd((pv1,trans_pv2,pv3), bins= (var1_binning, trans_var2_binning, var3_binning))
        Positive_Events_trans += Positive_hist_values_trans

        
        #print(Positive_Events,flush=True)

        #break
    #jjkknnhblk
    # this process is done, save result
    np.savez(os.path.join(resultdir, f"results_{rank}.npz"), trans_var2_binning = trans_var2_binning, var1_binning=var1_binning, var2_binning=var2_binning, var3_binning=var3_binning, Negative_Events = Negative_Events, Positive_Events = Positive_Events, trans_Negative_Events = Negative_Events_trans, trans_Positive_Events = Positive_Events_trans)
    
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
    var3_binning = None
    Negative_Events = None
    Positive_Events = None
    for rank in range(args.nprocesses):
        filename = os.path.join(args.resultdir, f"results_{rank}.npz")
        with np.load(filename) as result_file:
            if var1_binning is None or var2_binning is None or var3_binning is None:
                var1_binning = result_file["var1_binning"]
                var2_binning = result_file["var2_binning"]
                var3_binning = result_file["var3_binning"]
                trans_var2_binning = result_file["trans_var2_binning"]
            else:
                # make sure all processes used the same binning
                assert np.all(var1_binning == result_file["var1_binning"])
                assert np.all(var2_binning == result_file["var2_binning"])
                assert np.all(var3_binning == result_file["var3_binning"])
                assert np.all(trans_var2_binning == result_file["trans_var2_binning"])
                
                
            if Negative_Events is None or Positive_Events is None:
                Negative_Events = result_file["Negative_Events"]
                Positive_Events = result_file["Positive_Events"]
                trans_Negative_Events = result_file["trans_Negative_Events"]
                trans_Positive_Events = result_file["trans_Positive_Events"]
            else:
                Negative_Events += result_file["Negative_Events"]
                Positive_Events +=result_file["Positive_Events"]

                trans_Negative_Events += result_file["trans_Negative_Events"]
                trans_Positive_Events +=result_file["trans_Positive_Events"]
    # now save merged result
    np.savez(os.path.join(args.resultdir, "results.npz"), trans_var2_binning = trans_var2_binning, var1_binning=var1_binning, var2_binning=var2_binning, var3_binning = var3_binning, Negative_Events = Negative_Events, Positive_Events = Positive_Events)
    
    for binn in range(len(var3_binning) -1):
        
        figure = plt.figure(figsize=(12, 10))
        plot = figure.subplots(1, 1)
        plot.set_xlabel(r'$Z \times \Lambda_{\rm TRD}$',fontsize = 26)
        plot.set_ylabel(r'$\Lambda_{\rm Ecal}$', fontsize = 26)
        plot.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
        plot.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.minorticks_on()
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        for axis in ['top','bottom','left','right']:
            plot.spines[axis].set_linewidth(2)

        
        pevents= Positive_Events[: , : , binn]
        nevents= Negative_Events[: , : , binn]

        
        events=np.concatenate((nevents[::-1],pevents))
        events = np.ma.masked_where(events == 0, events).transpose()
        
        TRD_binn=np.concatenate(((-(var1_binning[1:]+var1_binning[:-1])/2)[::-1],(var1_binning[1:]+var1_binning[:-1])/2))
        ECAL_binn= (var2_binning[1:]+var2_binning[:-1])/2
        
        mesh = plot.pcolormesh(TRD_binn, ECAL_binn , events, norm=colors.LogNorm(), cmap =plt.cm.get_cmap('jet'))
        cbar = plt.colorbar(mesh, ax=plot, aspect=10)
        cbar.set_label("Events",fontsize=25)
        
        cbar.ax.tick_params(which="major",direction='in', length=7, width=1.5,labelsize=20)
        cbar.ax.tick_params(which="minor",direction='in')
        
   
        plot.plot([],[],' ',label="ISS data")
        plot.plot([],[],' ',label="Energy range = [20,50] Gev")
        plt.legend(fontsize=12,frameon=True,labelcolor='k')
        
        
    
        figure.savefig("2Dhistogram_3DEcal_TRD_Estimators.pdf" , dpi=250)

        
        figure = plt.figure(figsize=(12, 10))
        plot = figure.subplots(1, 1)
        plot.set_xlabel(r'$Z \times \Lambda_{\rm TRD}$',fontsize = 26)
        plot.set_ylabel(r'$\Lambda_{\rm Ecal}$', fontsize = 26)
        plot.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
        plot.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.minorticks_on()
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        for axis in ['top','bottom','left','right']:
            plot.spines[axis].set_linewidth(2)


        trans_pevents= trans_Positive_Events[: , : , binn]
        trans_nevents= trans_Negative_Events[: , : , binn]


        trans_events=np.concatenate((trans_nevents[::-1],trans_pevents))
        trans_events = np.ma.masked_where(trans_events == 0, trans_events).transpose()

        TRD_binn=np.concatenate(((-(var1_binning[1:]+var1_binning[:-1])/2)[::-1],(var1_binning[1:]+var1_binning[:-1])/2))
        ECAL_binn= (trans_var2_binning[1:]+trans_var2_binning[:-1])/2

        mesh = plot.pcolormesh(TRD_binn, ECAL_binn , trans_events, norm=colors.LogNorm(), cmap =plt.cm.get_cmap('jet'))
        cbar = plt.colorbar(mesh, ax=plot, aspect=10)
        cbar.set_label("Events",fontsize=25)

        cbar.ax.tick_params(which="major",direction='in', length=7, width=1.5,labelsize=20)
        cbar.ax.tick_params(which="minor",direction='in')

        plot.set_ylim(-20,10)
        plot.plot([],[],' ',label="ISS data")
        plot.plot([],[],' ',label="Energy range = [20,50] Gev")
        plt.legend(fontsize=15,frameon=True,labelcolor='k')



        figure.savefig("2Dhistogram_3DEcal_TRD_Estimators_trans.pdf" , dpi=250)



     
        
if __name__ == "__main__":
    main()    
   
