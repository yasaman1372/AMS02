#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 17:26:44 2022

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
#from ElectronSelectionCuts import *
from tools.roottree import read_tree

   

Preselection_Branches = Preselection_Cuts_Branch_List()
Selection_Cuts_Branches = Selection_Cuts_Branch_List()
Identification_Cuts_Branch =Identification_Cuts_Branch_List()

Branches = Preselection_Branches + Selection_Cuts_Branches + Identification_Cuts_Branch
 

def handle_file(arg):
    # This function is called once for each process.
    # rank and nranks specifies which process out of how many processes this one is.
    # Based on that, read_tree knows which files to read.

    filename, treename, chunk_size, rank, nranks, kwargs = arg
    data_type = 'MC'
    resultdir = kwargs["resultdir"]
    
    
    
    for events in read_tree(filename, treename, chunk_size=chunk_size, rank=rank, nranks=nranks, branches=Branches):
        events = ApplyPreselectionCuts(events) #preselection cuts are applied
        events = ApplySelectionCuts(events)
        events = ApplyIdentificationCuts(events,data_type)        

    
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
        pool_args = make_args(args.filename, args.treename, args.chunk_size, args.nprocesses, resultdir=args.resultdir)
        # and execute handle_file for each process
        for _ in pool.imap_unordered(handle_file, pool_args):
            pass

    # now all processes are done, load the results and merge them
  
    
if __name__ == "__main__":
    main()
