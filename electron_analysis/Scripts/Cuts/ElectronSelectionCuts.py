#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 00:11:40 2022

@author: yasaman
"""
from scipy import interpolate
import uproot
import awkward as ak
import numpy as np
from Cuts.ElectronTagCuts import *


address= '/home/op115134/Software/YasamanAnalysis/RootFiles/EnergydependentCuts/'

with uproot.open(address + 'TrdLRHeliElecLowerCut.root') as file:
        cut = file['TrdLRHeliElecLowerCut']
cutx = cut.members['fX']
cuty = cut.members['fY']
HeliumCut = interpolate.interp1d(cutx,cuty,fill_value="extrapolate")

def Selection_Cuts_Branch_List():
    branches=["Stoermer","TotalEnergy3D","Time","TrackerPattern", "TofBeta", "TofUpperCharge", "TrdPActiveLayersHybrid", "EcalEnergyElectronNewMaximumShower",
              "TrdPLikelihoodHybridHitsHelium", "TrdPLikelihoodHybridHitsElectron", "TrackerCharge", "TrackerTrackChoutkoMaxSpanChiSquareY"]
    return branches

TRACKER_PATTERN_MAP = np.array([0, 1, 3, 2, 5, 6, 4])

def ComputeTrackerPatternSortedByMDR(events):
    pattern = ak.to_numpy(events.TrackerPattern)
    return TRACKER_PATTERN_MAP[pattern + 1]
    #new_pattern = np.zeros_like(pattern, dtype=np.int16)
    #new_pattern[pattern == 0] = 1
    #new_pattern[pattern == 1] = 3
    #Events[Events==0]=1 # Layer 1 and 9, and maybe 2
    #Events[Events==1]=3 #Layer 1 and 2, but not 9
    #Events[Events==2]=2 #Layer 2 and 9, but not 1
    #Events[Events==3]=5 #Layer 1
    #Events[Events==4]=6 #Layer 2
    #Events[Events==5]=4 #Layer 9
    #Events[Events==-1]=0 #None of the above
    #return Events

def CutTofBeta(events):
    selection = (events.TofBeta >= 0.8) & (events.TofBeta <= 1.25)
    return events[selection]
    
def CutTofUpperCharge(events):
    selection = events.TofUpperCharge < 2.0
    return events[selection]

def CutTrdActiveLayers(events):
    selection = events.TrdPActiveLayersHybrid > 15
    return events[selection]

def CutTrdNoHelium(events):
    lower_cut = HeliumCut(ak.to_numpy(events.TotalEnergy3D))
    selection = TrdLRHeliElec_Rigidity_HybridHits_TrdP(events) > lower_cut
    return events[selection]

def CutTrackerPatternSortedByMDR(events):
    #selection = ComputeTrackerPatternSortedByMDR(events) == 1
    selection = (ComputeTrackerPatternSortedByMDR(events) >= 1) & (ComputeTrackerPatternSortedByMDR(events) <= 4)
    return events[selection]

def CutTrackerCharge(events):
    selection = (events.TrackerCharge > 0.5) & (events.TrackerCharge < 1.8)
    return events[selection]

def CutTrackerChiSquareY(events):
    selection = (events.TrackerTrackChoutkoMaxSpanChiSquareY > 0.01) & (events.TrackerTrackChoutkoMaxSpanChiSquareY < 25)
    return events[selection]

def TimeRange(events, mintime = 1305756000, maxtime = 1527458400):
    selection = (events.Time >= mintime) & (events.Time <= maxtime)
    return events[selection]

def GeomagneticCutoff(events,binn=np.array([0.5,0.65,0.82,1.01,1.22,1.46,1.72,2.0,2.31,2.65,3.0,3.36,3.73,
                               4.12,4.54,5.0,5.49,6.0,6.54,7.1,7.69,8.3,8.95,9.62,10.32,11.04,
                               11.8,12.59,13.41,14.25,15.14,16.05,17.0,17.98,18.99,20.04,21.13,
                               22.25,23.42,24.62,25.9,27.25,28.68,30.21,31.82,33.53,35.36,37.31,
                               39.39,41.61,44.0,46.57,49.33,52.33,55.58,59.13,63.02,67.3,72.05,
                               77.37,83.36,90.19,98.08,107.3,118.4,132.1,148.8,169.9,197.7,237.2,
                               290.0,370.0,500.0,700.0,1000.0])):
    
     index=np.maximum(np.digitize(ak.to_numpy(events.TotalEnergy3D),binn)-1,0)
    # index=np.maximum(np.digitize(ak.to_numpy(events.EcalEnergyDepositedMaximumShower),binn)-1,0)
     selection = binn[index]/events.Stoermer > 1.2 
     return events[selection]


# def GeomagneticCutoff(events,binn=np.array([0.5,0.65,0.82,1.01,1.22,1.46,1.72,2.0,2.31,2.65,3.0,3.36,3.73,
#                                4.12,4.54,5.0,5.49,6.0,6.54,7.1,7.69,8.3,8.95,9.62,10.32,11.04,
#                                11.8,12.59,13.41,14.25,15.14,16.05,17.0,17.98,18.99,20.04,21.13,
#                                22.25,23.42,24.62,25.9,27.25,28.68,30.21,31.82,33.53,35.36,37.31,
#                                39.39,41.61,44.0,46.57,49.33,52.33,55.58,59.13,63.02,67.3,72.05,
#                                77.37,83.36,90.19,98.08,107.3,118.4,132.1,148.8,169.9,197.7,237.2,
#                                290.0,370.0,500.0,700.0,1000.0])):
    
#      index=np.maximum(np.digitize(ak.to_numpy(events.EcalEnergyElectronNewMaximumShower),binn)-1,0)
#      selection = binn[index]/events.Stoermer > 1.2
#      return events[selection]

    
    
