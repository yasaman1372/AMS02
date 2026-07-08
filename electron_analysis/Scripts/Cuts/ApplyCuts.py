#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 14:55:34 2022

@author: yasaman
"""

from Cuts.PreselectionCuts import *
from Cuts.ElectronSelectionCuts import *
from Cuts.ElectronIdentificationCuts import *



def ApplyPreselectionCuts(events):
    #At least one Ecal Shower whithin fiducial volume
    events = CutEcalCentreOfGravityX(events)
    events = CutEcalCentreOfGravityY(events)
    
   # print("At least one Ecal Shower whithin fiducial volume",len(events))

    #At least one Useful TRD track
    events = CutTrdHasUsefulTrack(events)
    #events = CutEcalTrdTrackMachingChi2(events) 

   # print("At least one Useful TRD track", len(events))
    
    #Al least one useful TOF cluster combination 
    events = CutTofNumberOfLayers(events)
    events = CutTofTimeDifference(events)
    events = CutTofTrdMatching(events)
    events = CutTofIsDowngoing(events)

   # print("Al least one useful TOF cluster combination",len(events))
    
    #Highest Energetic Ecal Shower 
   # events = CutEcalShowerWithMaximumEnergyIsUsed(events)
    
    #Not in ISS solar array shadow
    events = CutIsNotInSolarArrayShadow(events)

   # print("Not in ISS solar array shadow", len(events))
    
    return events


def ApplySelectionCuts(events):
    events = CutTofBeta(events) #TOF velocity 
    events = CutTofUpperCharge(events) #Upper TOF charge
    events = CutTrdActiveLayers(events) #Enough Active Layers in TRD
    events = CutTrdNoHelium(events) #TRD Helium rejection
    events = CutTrackerPatternSortedByMDR(events) #Tracker hit pattern
    events = CutTrackerCharge(events) #Inner Tracker charge
    events = CutTrackerChiSquareY(events) #Tracker track goodness of fit in Y project
    events = TimeRange(events) 
    events =  GeomagneticCutoff(events) #Energy above Geomegnetic cutoff
    
    return events
 
   

def ApplyIdentificationCuts(events):
    events = CutEnergyOverRigidity(events) #Energy Rigidity matching
    #events = CutEcalChiSquareLateralNormalized(events,data_type) #Ecal lateral shower shape
    events = CutTrackerTrackEcalCogDeltaX(events) #Tracker Ecal matching in X projection
    events = CutTrackerTrackEcalCogDeltaY(events) #Tracker Ecal matching in Y projection
    
    return events
    
    
