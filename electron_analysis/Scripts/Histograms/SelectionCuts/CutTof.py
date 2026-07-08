#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 11:22:11 2022

@author: yasaman
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 20:39:34 2022

@author: yasaman
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 14:07:43 2022

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


Branches= ['TofBeta', 'TrackerTrackChoutkoMaxSpanRigidity', 'EcalShowerDirectionZ', 'EcalBDT_v7_EnergyD',
 'EcalEnergyDepositedMaximumShower', 'TrackerTrackChoutkoMaxSpanRigidity','TrdPLikelihoodHybridHitsElectronECAL','TrdPLikelihoodHybridHitsProtonECAL','TrackerCharge','McEventWeightElectron',
'EcalCentreOfGravityX','EcalCentreOfGravityY', 'TrdTrackNumberOfSubLayersXZ','TrdTrackNumberOfSubLayersYZ','TrdTrackFirstSubLayerYZ','TrdTrackLastSubLayerYZ','TofNumberOfLayers',
'TofDeltaT','TofTrdMatchNorm','EcalEnergyElectronNew','EcalEnergyElectronNewMaximumShower','TrackerTrackIsNotInSolarArrayShadow','TrdTrackEcalCogAngleXZ', 'TrdTrackEcalCogAngleYZ',
'TrdTrackEcalCogDeltaX', 'TrdTrackEcalCogDeltaY'] 

def handle_file(arg):
    # This function is called once for each process.
    # rank and nranks specifies which process out of how many processes this one is.
    # Based on that, read_tree knows which files to read.

    filename, treename, chunk_size, rank, nranks, kwargs = arg

    resultdir = kwargs["resultdir"]
    data_type = kwargs["data_type"]

    var1_name= 'TofBeta'
    var1_min = 0
    var1_max = 2
    bin1_num = 200

    
    var1_binning = np.linspace(var1_min, var1_max, bin1_num +1)

    Events = np.zeros(bin1_num)
    Passed_Events = np.zeros(bin1_num)
    
    
    for events in read_tree(filename, treename, chunk_size=chunk_size, rank=rank, nranks=nranks, branches=Branches):
        events = ApplyPreselectionCuts(events) #preselection cuts are applied
        passed_events = CutTofBeta(events)

        # TagElectron = [NegativeRigidityTag, EcalElectronBdtTag, ElectronEnergyOverRigidityTag, ElectronTrdLikelihoodRatioOnlyTag, TrackerChargeTag]

        # for cut in TagElectron:
        #     events = cut(events) #Tag cuts are applied
        
        v1=ak.to_numpy(events[var1_name])


        if data_type =="ISS":
            passed_hist_values, passed_hist_edges = np.histogram(ak.to_numpy(passed_events[var1_name]), bins=var1_binning)
            hist_values, hist_edges = np.histogram(ak.to_numpy(events[var1_name]), bins=var1_binning)

        elif data_type == "MC":
            passed_hist_values, passed_hist_edges = np.histogram(ak.to_numpy(passed_events[var1_name]), bins=var1_binning,weights=ak.to_numpy(passed_events.McEventWeightElectron))
            hist_values, hist_edges = np.histogram(ak.to_numpy(events[var1_name]), bins=var1_binning, weights=ak.to_numpy(events.McEventWeightElectron))
        
        Passed_Events += passed_hist_values
        Events += hist_values

    # this process is done, save result
    np.savez(os.path.join(resultdir, f"results_{rank}.npz"), var1_binning=var1_binning,Events=Events, Passed_Events=Passed_Events)
    
def make_args(filename, treename, chunk_size, nranks, **kwargs):
    # this function creates the arguments for handle_file for each process
    for rank in range(nranks):
        yield (filename, treename, chunk_size, rank, nranks, kwargs)
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", default="", help = "ISS or MC")
    parser.add_argument("--title", default="MC", help="title of the input data (e.g. ISS)")
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
    var1_binning = None
    Events = None
    for rank in range(args.nprocesses):
        filename = os.path.join(args.resultdir, f"results_{rank}.npz")
        with np.load(filename) as result_file:
            if var1_binning is None :
                var1_binning = result_file["var1_binning"]
            else:
                # make sure all processes used the same binning
                assert np.all(var1_binning == result_file["var1_binning"])
    
                
            if Events is None:
                Events = result_file["Events"]
                Passed_Events = result_file["Passed_Events"]
            else:
                Events += result_file["Events"]
                Passed_Events += result_file["Passed_Events"]

    # now save merged result
    np.savez(os.path.join(args.resultdir, "results.npz"), var1_binning=var1_binning, Events=Events, Passed_Events=Passed_Events)
   
      
    #plot the histogram    
    #font_size = 15
    figure = plt.figure(figsize=(12, 10))
    plot = figure.subplots(1, 1)
    plot.set_xlabel("TOF"+" "+r'$\beta$',fontsize = 26,fontweight = 'bold')
    plot.set_ylabel("Normalaized Events", fontsize = 26,fontweight = 'bold')
    plot.set_yscale("log")
    plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
    plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plot.set_xlim(0.72,1.35)
    plot.set_ylim(10**-4,0.2)
    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)
        
        
    
    Events= Events/np.sum(Events)

    plot.step(np.concatenate(([var1_binning[0]], var1_binning)), np.concatenate(([0], Events, [0])), where="post",linewidth=2)
    plt.fill_between(var1_binning, np.concatenate(([0], Events)),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='b')
    # plot.plot([],[],' ',label="MC data")
    # plot.plot([],[],' ',label="lower cut = 0.8")
    # plot.plot([],[],' ',label="upper cut= 1.25")
    # plt.legend(bbox_to_anchor=(0.48, 0.71),fontsize=17,frameon=False)
    #plt.fill_between(pl)
    
    plt.axvline(x=0.8,linestyle=':',color='k',linewidth=5)
    plt.axvline(x= 1.24,linestyle=':',color='k',linewidth=5)
    
    
    

    
        
        
    figure.savefig("TOFhistogram_MC_pass6_Preselection.pdf" , dpi=250)
    plt.close(figure)
  
    
if __name__ == "__main__":
    main() 
