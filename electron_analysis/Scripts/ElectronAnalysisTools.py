#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 13:31:54 2022

@author: yasaman
"""
import awkward as ak
import numpy as np
from enum import Enum
from scipy import interpolate
import uproot

class BoundaryType(Enum):
    open = 0
    close = 1
    none = 2
   
class SingleCut_Energydependent:
    def __init__(self, cut_function, branch, boundary, cutvalues_func):
        self.cut_function = cut_function
        self.branch = branch
        self.boundary = boundary
        self.cutvalues_func = cutvalues_func
        
    def apply(self, events):
        return self.cut_function(events, self.branch, self.boundary, self.cutvalues)        
        
           
class SingleCut:
    def __init__(self, cut_function, branch, boundary, args):
        self.cut_function = cut_function
        self.branch = branch
        self.boundary = boundary
        self.args = args
        
    def apply(self, events):
        return self.cut_function(events, self.branch, self.boundary, **self.args)

    
class MultipleCuts:
    def __init__(self,cuts,combine_func):
        self.cuts = cuts
        self.combine_func = combine_func
        
    def apply(self, events):
        result =[cut.apply(events) for cut in self.cuts]
        return self.combine_func(result,axis=0)
 
def make_cutvalues_from_Rootfile(rootfile,filename):
    with uproot.open(rootfile) as file:
        cutfile = file[filename]
        x = cutfile.members['fX']
        y = cutfile.members['fY']
        cut = interpolate.interp1d(x,y,fill_value="extrapolate") 
        return cut

def energydependent_lowercut(events, branch, boundary, cut_values_func): ##raise error if cut value is None
    lowercut = cut_values_func(events[branch["energy_branch"]])
    if  boundary == BoundaryType.open:
        selection = events[branch["main_branch"]] < lowercut
    else:
        selection = events[branch["main_branch"]] <= lowercut
    return selection  

def energydependent_uppercut(events, branch, boundary, cut_values): ##raise error if cut value is None
    uppercut = cut_values(events[branch["energy_branch"]])
    if  boundary == BoundaryType.open:
        selection = events[branch["main_branch"]] < uppercut
    else:
        selection = events[branch["main_branch"]] <= uppercut
    return selection

 
def uppercut_abs(events, branch, boundary, **args):
    
    if  boundary == BoundaryType.open:
        selection = np.abs(ak.to_numpy(branch)) < args["max"]        
    else:
        selection = np.abs(ak.to_numpy(branch)) <= args["max"]         
    return selection




def lowercut(events, branch, boundary, **args):
    
    if boundary == BoundaryType.open:
        selection = events[branch] > args["min"]           
    else:
        selection = events[branch] >= args["min"]       
    return selection
 


   
def uppercut(events, branch, boundary, maxx=None, **args):
    
    if boundary == BoundaryType.open:
        selection = events[branch] < args["max"]           
    else:
        selection = events[branch] <= args["max"]       
    return selection

    


def absolute_difference_uppercut(events, branch, boundary, **args):
    
    if  boundary == BoundaryType.open:   
        selection = np.abs(ak.to_numpy(events[branch[0]]) - ak.to_numpy(events[branch[1]])) < args["max"]
    else:
        selection = np.abs(ak.to_numpy(events[branch[0]]) - ak.to_numpy(events[branch[1]])) <= args["max"]        
    return selection 

def boolian(events,branch,boundary,**args):
    return events[branch]


CUT_FUNCTIONS = {
    "uppercut_abs": uppercut_abs,
    "uppercut": uppercut,
    "lowercut": lowercut ,
    "absolute_difference_uppercut": absolute_difference_uppercut,
    "boolian": boolian,
    "Energydependent_lowercut": energydependent_lowercut,
    "Energydependent_uppercut": energydependent_uppercut
}


def load_cut(cut_config):
    
    if cut_config["type"]== "single_cut":
        return SingleCut(CUT_FUNCTIONS[cut_config["selection"]], cut_config["branch"],
                               BoundaryType[cut_config["boundary"]] ,cut_config["args"])
    
    if cut_config["type"]== "Energydependent_single_cut":       
        return SingleCut_Energydependent(CUT_FUNCTIONS[cut_config["selection"]], cut_config["branch"],
                               BoundaryType[cut_config["boundary"]] , make_cutvalues_from_Rootfile(cut_config["args"]["Rootfile"], cut_config["args"]["filename"]))
        
    if cut_config["type"] == "multipe_cuts_all":   
        return MultipleCuts([load_cut(inner_cut_config) for inner_cut_config in cut_config["cuts"]], np.all)
   
    if cut_config["type"] == "multipe_cuts_any":   
        return MultipleCuts([load_cut(inner_cut_config) for inner_cut_config in cut_config["cuts"]], np.any)


class Selection:
    
    def __init__(self, cuts):
        
        self.cuts = cuts
    
    @staticmethod
    def load(config):
        cuts=[]
        
        for cut_name, cut_config in config.items():
            print(cut_name)
            cuts.append(load_cut(cut_config))
        return Selection(cuts)
                    
    def apply(self, events):
        for cut in self.cuts:
            events = events[cut.apply(events)]
        return events    
                            
    
                
    
            
            
            
        
    
  
             
              
      

         
     

              
      
              
              
      
    