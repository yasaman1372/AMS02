#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:59:00 2023

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
    data_type="ISS"

    resultdir = kwargs["resultdir"]

    var1_name= 'TrdEstimator'
    var1_min = 0
    var1_max = 2
    bin1_num = 100
    var1_binning = np.linspace(var1_min, var1_max, bin1_num +1)
        
    var2_name='ElectronCCMVABDT'
    var2_min = -1
    var2_max = 1
    bin2_num = 100
    var2_binning = np.linspace(var2_min, var2_max, bin2_num +1)
    
    
  
    var3_name = "TotalEnergy3D" 
    #var3_binning = np.array([0.5,1000.0])
    var3_binning = np.array([0.5,0.65,0.82,1.01,1.22,1.46,1.72,2.0,2.31,2.65,3.0,3.36,3.73,
     4.12,4.54,5.0,5.49,6.0,6.54,7.1,7.69,8.3,8.95,9.62,10.32,11.04,
     11.8,12.59,13.41,14.25,15.14,16.05,17.0,17.98,18.99,20.04,21.13,
     22.25,23.42,24.62,25.9,27.25,28.68,30.21,31.82,33.53,35.36,37.31,
     39.39,41.61,44.0,46.57,49.33,52.33,55.58,59.13,63.02,67.3,72.05,
     77.37,83.36,90.19,98.08,107.3,118.4,132.1,148.8,169.9,197.7,237.2,
     290.0,370.0,500.0,700.0,1000.0])
    
    bin3_num = len(var3_binning) -1
    
    
    var4_name= 'Rigidity sign'
    var4_binning = np.array([-1.5,0,1.5])
    bin4_num = len(var4_binning) -1
    
  
    Events_st = np.zeros((bin1_num, bin2_num, bin3_num, bin4_num))
    Events_mt = np.zeros((bin1_num, bin2_num, bin3_num, bin4_num))
    Events_all = np.zeros((bin1_num, bin2_num, bin3_num, bin4_num))
    
    
    for events in read_tree(filename, treename, chunk_size=chunk_size, rank=rank, nranks=nranks):
        
        #events_all=events
        #print(events_all)
        events_all = ApplyPreselectionCuts(events) #preselection cuts are applied
        # events = ApplySelectionCuts(events)
        # events = ApplyIdentificationCuts(events,data_type)
        # events = GeomagneticCutoff(events)
        # EcalBDT = Ecal_BDT_cut(events.EcalEnergyElectronNewMaximumShower)
        # events = events[events.EcalBDT_v7_EnergyD > EcalBDT ]
        # events = events[events.EcalBDT3D > -0.75 ]
        # events_st = events[events.TrackerNumberOfTracks == 1]
        # events_mt = events[events.TrackerNumberOfTracks > 1]
        # events_all = events[events.TrackerNumberOfTracks >= 1]        

        # #single track events
        # v1_st = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(events_st))
        # v2_st = ak.to_numpy(events_st[var2_name])
        # v3_st = ak.to_numpy(events_st[var3_name])
        # v4_st = np.sign(ak.to_numpy(events_st["TrackerTrackGBLMaxSpanRigidity"]))
        
        # #multi track events
        # v1_mt = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(events_mt))
        # v2_mt = ak.to_numpy(events_mt[var2_name])
        # v3_mt = ak.to_numpy(events_mt[var3_name])  
        # v4_mt = np.sign(ak.to_numpy(events_mt["TrackerTrackGBLMaxSpanRigidity"]))
        
        #all track events
        v1_all = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(events_all))
        v2_all = ak.to_numpy(events_all[var2_name])
        v3_all = ak.to_numpy(events_all[var3_name])  
        v4_all = np.sign(ak.to_numpy(events_all["TrackerTrackGBLMaxSpanRigidity"]))

        # hist_values_st, edges_st = np.histogramdd((v1_st,v2_st,v3_st,v4_st), bins= (var1_binning, var2_binning, var3_binning, var4_binning),weights=ak.to_numpy(events_st.McEventWeightElectron) )        
        # Events_st += hist_values_st
        
        
        # hist_values_mt, edges_mt = np.histogramdd((v1_mt,v2_mt,v3_mt,v4_mt), bins= (var1_binning, var2_binning, var3_binning, var4_binning),weights=ak.to_numpy(events_mt.McEventWeightElectron))        
        # Events_mt += hist_values_mt
        
        
        hist_values_all, edges_all = np.histogramdd((v1_all,v2_all,v3_all,v4_all), bins= (var1_binning, var2_binning, var3_binning, var4_binning))        
        Events_all += hist_values_all
        
        
          
        

    # this process is done, save result
    np.savez(os.path.join(resultdir, f"results_{rank}.npz"), var1_binning=var1_binning, var2_binning=var2_binning, var3_binning=var3_binning, var4_binning=var4_binning, Events_all = Events_all)
    
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
    var3_binning = None
    var4_binning = None
    # Events_st = None
    # Events_mt = None
    Events_all = None
    for rank in range(args.nprocesses):
        filename = os.path.join(args.resultdir, f"results_{rank}.npz")
        with np.load(filename) as result_file:
            if var1_binning is None or var2_binning is None or var3_binning is None:
                var1_binning = result_file["var1_binning"]
                var2_binning = result_file["var2_binning"]
                var3_binning = result_file["var3_binning"]
                var4_binning = result_file["var4_binning"]
            else:
                # make sure all processes used the same binning
                assert np.all(var1_binning == result_file["var1_binning"])
                assert np.all(var2_binning == result_file["var2_binning"])
                assert np.all(var3_binning == result_file["var3_binning"])
                assert np.all(var4_binning == result_file["var4_binning"])
                
            if Events_all is None:
                # Events_st = result_file["Events_st"]
                # Events_mt = result_file["Events_mt"]
                Events_all = result_file["Events_all"]
            else:
                # Events_st += result_file["Events_st"]
                # Events_mt +=result_file["Events_mt"]
                Events_all += result_file["Events_all"]

    # now save merged result
    np.savez(os.path.join(args.resultdir, "results.npz"), var1_binning=var1_binning, var2_binning=var2_binning, var3_binning = var3_binning, var4_binning=var4_binning, Events_all = Events_all)
    
    for binn in range(len(var3_binning) -1):
        
        figure = plt.figure(figsize=(12, 10))
        plot = figure.subplots(1, 1)
        plot.set_xlabel(r'$Z \times \Lambda_{\rm TRD}$',fontsize = 26)
        plot.set_ylabel(r'$\Lambda_{\rm CC}$', fontsize = 26)
        plot.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
        plot.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.minorticks_on()
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        for axis in ['top','bottom','left','right']:
            plot.spines[axis].set_linewidth(2)

        data_negative_all = Events_all[:,:,binn,0]
        data_positive_all = Events_all[:,:,binn,1]
        signed_data_all = np.concatenate((data_negative_all[::-1,:], data_positive_all), axis=0)

       # events = signed_data_all
        events = np.ma.masked_where(signed_data_all <= 1e-10, signed_data_all).transpose()
        
        signed_trd_binning = np.concatenate((-var1_binning[:0:-1], var1_binning))
        
        # TRD_binn=np.concatenate(((-(var1_binning[1:]+var1_binning[:-1])/2)[::-1],(var1_binning[1:]+var1_binning[:-1])/2))
        CCMVA_binning= (var2_binning[1:]+var2_binning[:-1])/2
        
        mesh = plot.pcolormesh(0.5*(signed_trd_binning[1:]+signed_trd_binning[:-1]), CCMVA_binning , events,norm=colors.LogNorm(),cmap =plt.cm.get_cmap('jet'))
        cbar = plt.colorbar(mesh, ax=plot, aspect=10)
        cbar.set_label("Events",fontsize=25)
        
        cbar.ax.tick_params(which="major",direction='in', length=7, width=1.5,labelsize=20)
        cbar.ax.tick_params(which="minor",direction='in')
        
   
        plot.plot([],[],' ',label="MC electron data pass8")
        plot.plot([],[],' ',label="Energy range = [0.5,1000] Gev")
        plt.legend(fontsize=12,frameon=True,labelcolor='k')
        
        
    
        figure.savefig("2D_histogram_CCMVABDT_TRD_Estimators_single.pdf" , dpi=250)
        
        figure = plt.figure(figsize=(12, 10))
        plot = figure.subplots(1, 1)
        plot.set_ylabel("Events",fontsize = 26)
        plot.set_xlabel(r'$\Lambda_{\rm CC}$', fontsize = 26)
        plot.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
        plot.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.minorticks_on()
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        for axis in ['top','bottom','left','right']:
            plot.spines[axis].set_linewidth(2)

        data_negative_all = Events_all[:,:,binn,0].sum(axis=0)
        data_positive_all = Events_all[:,:,binn,1].sum(axis=0)
        
        #CCMVA_binning= (var2_binning[1:]+var2_binning[:-1])/2
        
        CCMVA_binning= var2_binning
        plot.step(np.concatenate(([CCMVA_binning[0]], CCMVA_binning)), np.concatenate(([0], data_negative_all, [0])), where="post",linewidth=2,label="R<0")
        plot.step(np.concatenate(([CCMVA_binning[0]], CCMVA_binning)), np.concatenate(([0], data_positive_all, [0])), where="post",linewidth=2,label="R>0")
        #plt.plot(CCMVA_binning, data_negative_all, label="R <0")
        #plt.plot(CCMVA_binning, data_positive_all, label="R >0")
        
    
        
        cbar.ax.tick_params(which="major",direction='in', length=7, width=1.5,labelsize=20)
        cbar.ax.tick_params(which="minor",direction='in')
        
   
        plot.plot([],[],' ',label="MC Electron data pass8")
        plot.plot([],[],' ',label="Energy range = [0.5,1000] Gev")
        plt.legend(fontsize=12,frameon=True,labelcolor='k')
        
        
    
        figure.savefig("2D_histogram_CCMVABDT_TRD_Estimators_projection.pdf" , dpi=250)
     
        
if __name__ == "__main__":
    main() 
