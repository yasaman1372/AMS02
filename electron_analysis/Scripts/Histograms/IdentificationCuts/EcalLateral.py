#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 15:25:20 2022

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

from scipy import interpolate
from Cuts.ElectronTagCuts import *
from Cuts.ApplyCuts import *
from tools.roottree import read_tree
from Cuts.ElectronIdentificationCuts import CutEcalChiSquareLateralNormalized 
from Cuts.ElectronIdentificationCuts import ComputeEcalChiSquareLateralNormalized

address= '/home/op115134/Software/YasamanAnalysis/RootFiles/'

with uproot.open(address + 'CutValue.root') as file:
	cut = file['CutValues']
#print(cut,flush=True)
cutx = cut.members['fX']
cuty = cut.members['fY']
F_cut = interpolate.interp1d(cutx,cuty,fill_value="extrapolate")

with uproot.open(address + 'LeptonAnalysis_EcalChiSquareLateralNormalizedCalibration.root') as file:
        
    meanMCgraf = file['meanMCGraphSmoothed']
    sigmaMCgraf = file ['sigmaMCGraphSmoothed']
        
    meanISSgraf = file['meanISSGraphSmoothed']
    sigmaISSgraf = file['sigmaISSGraphSmoothed']
    
   # print(meanMCgraf.all_members["fX"], flush=True)    
    meanMC_int = interpolate.interp1d(meanMCgraf.all_members["fX"], meanMCgraf.all_members["fY"])
    sigmaMC_int = interpolate.interp1d(sigmaMCgraf.all_members["fX"], sigmaMCgraf.all_members["fY"])
       
    meanISS_int = interpolate.interp1d(meanISSgraf.all_members["fX"], meanISSgraf.all_members["fY"])
    sigmaISS_int = interpolate.interp1d(sigmaISSgraf.all_members["fX"], sigmaISSgraf.all_members["fY"])
    

#def EcalChiSquareLateralNormalizedCorrected(nchi2, EcalEnergyElectron, data_type, meanMC, meanISS, sigmaMC, sigmaISS):
        
   # if data_type == "ISS":
       # return (nchi2 - meanISS) / sigmaISS
    
   # elif data_type == "MC":
       # return (nchi2 - meanMC)/sigmaMC
    
#def ComputeEcalChiSquareLateralNormalized(events, data_type):
    
   # chi2 = ak.to_numpy(events.EcalChiSquareLateral)
   # print("chi2",flush=True)
   # print(chi2,flush=True)
   # pr1_binningrint(len(events),flush=True)     
   # print(chi2.shape,flush=True)  
   # if chi2 == 0:
       # return 0.0
   # else:
   # energy = ak.to_numpy(events.EcalEnergyElectronNewMaximumShower)
   # x= np.clip(energy,15.01,399.99)
   
   # x=np.log10(x)
   # print(x,flush=True) 
   # meanISS = meanISS_int(x)
   # meanMC = meanMC_int(x)

   # sigmaISS = sigmaISS_int(x)
   # sigmaMC = sigmaMC_int(x)
    
        #parameters from gbatch as of 2015-09-04 (EcalPDF Version 2)
   # p0 = 0.281165
   # p1 = -0.0493095
   # p2 = 0.120408
   # p3 = -0.0181409
    
   # nchi2 = chi2 - (p0 + (p1*x) + (p2*x*x) + (p3*x*x*x))
   # print(nchi2.shape,chi2.shape,flush=True)
   # return nchi2*(chi2 !=0)
   # return EcalChiSquareLateralNormalizedCorrected(nchi2, energy, data_type, meanMC, meanISS, sigmaMC, sigmaISS)*(chi2 !=0)


Tag_Cuts = [NegativeRigidityTag, EcalElectronTag, ElectronUpperEnergyOverRigidityTag, ElectronTrdLikelihoodRatioOnlyTag]

Tag_Cut_Branches = ['TofBeta', 'TrackerTrackChoutkoMaxSpanRigidity', 'EcalShowerDirectionZ', 'EcalBDT_v7_EnergyD',
 'EcalEnergyDepositedMaximumShower', 'TrackerTrackChoutkoMaxSpanRigidity','TrdPLikelihoodHybridHitsElectronECAL',
 'TrdPLikelihoodHybridHitsProtonECAL']

Preselection_Branches = Preselection_Cuts_Branch_List()

Histogram_branches = ['McEventWeightElectron','EcalEnergyElectronNewMaximumShower', 'EcalChiSquareLateral']

Branches = Tag_Cut_Branches + Preselection_Branches + Histogram_branches
 

def handle_file(arg):
    # This function is called once for each process.
    # rank and nranks specifies which process out of how many processes this one is.
    # Based on that, read_tree knows which files to read.

    filename, treename, chunk_size, rank, nranks, kwargs = arg
    data_type = kwargs["data_type"]
    resultdir = kwargs["resultdir"]
    
    var1_name = "EcalEnergyElectronNewMaximumShower"
    var1_binning = np.array([0.5,0.65,0.82,1.01,1.22,1.46,1.72,2.0,2.31,2.65,3.0,3.36,3.73,
                              4.12,4.54,5.0,5.49,6.0,6.54,7.1,7.69,8.3,8.95,9.62,10.32,11.04,
                              11.8,12.59,13.41,14.25,15.14,16.05,17.0,17.98,18.99,20.04,21.13,
                              22.25,23.42,24.62,25.9,27.25,28.68,30.21,31.82,33.53,35.36,37.31,
                              39.39,41.61,44.0,46.57,49.33,52.33,55.58,59.13,63.02,67.3,72.05,
                              77.37,83.36,90.19,98.08,107.3,118.4,132.1,148.8,169.9,197.7,237.2,
                              290.0,370.0,500.0,700.0,1000.0])
    
    
    bin1_num = len(var1_binning) -1

    var2_name= 'EcalChiSquareLateral'
    var2_min = -3.0
    var2_max = 15
    bin2_num = 200
    var2_binning = np.linspace(var2_min, var2_max, bin2_num +1)
    
    Events = np.zeros((bin1_num, bin2_num))
    
    
    for events in read_tree(filename, treename, chunk_size=chunk_size, rank=rank, nranks=nranks, branches=Branches):
        events = ApplyPreselectionCuts(events) #preselection cuts are applied
       # events = CutEcalChiSquareLateralNormalized(events,data_type)
       # upper_cut = F_cut(events.EcalEnergyElectronNewMaximumShower)
       # EcalEnergyElectronNewMaximumShower = ak.to_numpy(events.EcalEnergyElectronNewMaximumShower )
       # selection = ComputeEcalChiSquareLateralNormalized(events, data_type) < upper_cut 
       # events = events[selection]


        # events = CutEcalChiSquareLateralNormalized(events,data_type)
       # for cut in Tag_Cuts:
           # events = cut(events) #Tag cuts are applied
        
        v1=ak.to_numpy(events[var1_name])
        #v2=ak.to_numpy(events[var2_name])
        v2 = ComputeEcalChiSquareLateralNormalized(events, data_type)      
       # print(v2,flush=True)       

        if data_type == 'MC': 
            hist_values, edges = np.histogramdd((v1,v2), bins= (var1_binning, var2_binning), weights=ak.to_numpy(events.McEventWeightElectron))
        
        elif data_type == 'ISS':
           # print("hist value ISS:") 
            hist_values, edges = np.histogramdd((v1,v2), bins= (var1_binning, var2_binning))
           # print(hist_values,flush=True)
                     
        Events += hist_values


    # this process is done, save result
    np.savez(os.path.join(resultdir, f"results_{rank}.npz"),var1_binning=var1_binning,var2_binning=var2_binning,Events=Events)
    
def make_args(filename, treename, chunk_size, nranks, **kwargs):
    # this function creates the arguments for handle_file for each process
    for rank in range(nranks):
        yield (filename, treename, chunk_size, rank, nranks, kwargs)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", default="MC", help="title of the input data (e.g. ISS)")
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
            if var1_binning is None or var2_binning is None :
                var1_binning = result_file["var1_binning"]
                var2_binning = result_file["var2_binning"]
               # upper_cut = result_file['upper_cut']
               # EcalEnergyElectronNewMaximumShower = result_file['EcalEnergyElectronNewMaximumShower']
                
            else:
                # make sure all processes used the same binning
                assert np.all(var1_binning == result_file["var1_binning"])
                assert np.all(var2_binning == result_file["var2_binning"])
                
            if Events is None:
                Events = result_file["Events"]
            else:
                Events += result_file["Events"]

    # now save merged result
   # print(Events,flush=True)
    np.savez(os.path.join(args.resultdir, "results.npz"),var2_binning=var2_binning, var1_binning=var1_binning, Events=Events)
   # print(Events.sum())

    
    figure = plt.figure(figsize=(12, 10))
    plot = figure.subplots(1, 1)
    plot.set_xlabel("Ecal Energy/ GeV",fontsize = 26)
    plot.set_ylabel(" Ecal Lateral shower ", fontsize = 26)
    plot.set_xscale("log")
    plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=20)
    plot.tick_params(axis='both', which="minor",direction='in',length=7, width=1.5, labelsize=20)
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plot.set_ylim(-2,15)
    plot.set_xlim(7,max((var1_binning[1:]+var1_binning[:-1])/2))
    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)
        
        
    hist_cut = F_cut((var1_binning[1:]+var1_binning[:-1])/2)
   # hcut = np.meshgrid((var2_binning[1:]+var2_binning[:-1])/2, cuty_new)[1]
   # xbin = (var1_binning[1:]+var1_binning[:-1])/2
   # ybin = (var2_binning[1:]+var2_binning[:-1])/2
    events = np.ma.masked_where(Events == 0, Events).transpose()
    scale=1/events.sum(axis=0)
    events = scale*events
   # print(np.shape(events))
   # cut_events = np.zeros_like(events)
   # for i in range(len(xbin)):
       # for j in range(len(ybin)):
           # if ybin[j] < hist_cut[i]:
               # cut_events[j,i]=events[j,i]
			
   # cut_events = np.ma.masked_where(cut_events == 0.0, cut_events)	
       
    mesh = plot.pcolormesh(var1_binning,var2_binning, events, norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap =plt.cm.get_cmap('jet'))
    cbar = plt.colorbar(mesh, ax=plot, aspect=10)
    cbar.set_label("Normalized Events",fontsize=24)
    cbar.ax.tick_params(which="major",direction='in', length=10, width=1.5,labelsize=20)
    cbar.ax.tick_params(which="minor",direction='in',length=7, width=1.5, labelsize=20)        
    
    plt.scatter(cutx,cuty,c="r")
    plt.scatter((var1_binning[1:]+var1_binning[:-1])/2,hist_cut,c="k")
   # plt.scatter(EcalEnergyElectronNewMaximumShower,upper_cut, c='m') 
   # plt.scatter(cutx,cuty,c="r")    
    figure.savefig("EcalLateralShower_cut.pdf" , dpi=250)
    plt.close(figure)
  
    
if __name__ == "__main__":
    main() 
