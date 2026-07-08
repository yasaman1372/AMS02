#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 17:26:00 2022

@author: yasaman
"""

import awkward as ak
import numpy as np
from Cuts.ElectronTagCuts import *


def Preselection_Cuts_Branch_List():
    branches = ["TotalEnergy3D",'EcalCentreOfGravityX','EcalCentreOfGravityY', 'TrdTrackNumberOfSubLayersXZ',
    'TrdTrackNumberOfSubLayersYZ','TrdTrackFirstSubLayerYZ','TrdTrackLastSubLayerYZ','TofNumberOfLayers',
    'TofDeltaT','TofTrdMatchNorm','EcalEnergyElectronNew','EcalEnergyElectronNewMaximumShower','TrackerTrackIsNotInSolarArrayShadow',
    'TrdTrackEcalCogAngleXZ', 'TrdTrackEcalCogAngleYZ',
    'TrdTrackEcalCogDeltaX', 'TrdTrackEcalCogDeltaY', 'TofBeta', 'EcalEnergyDepositedMaximumShower']
    return branches


def TrdEcalMatchingAngularResolutionFunction(E,par):
	return par[0] + par[1]* np.maximum(E,1e-7)**par[2]

def TrdEcalMatchingSpatialResolutionFunction(E,par):
	return par[0]/(E + par[1]+ 1e-7) + par[2]

def ComputeTrdEcalChi2(events):
        E = events.EcalEnergyElectronNewMaximumShower
       # E = events.EcalEnergyDepositedMaximumShower
        trdTrackEcalCogAngleXZUncertainty = TrdEcalMatchingAngularResolutionFunction(E,par=[2.45281e-02, 7.88632e-01, -6.86901e-01])
        trdTrackEcalCogAngleYZUncertainty = TrdEcalMatchingAngularResolutionFunction(E, par =[1.67174e-02, 1.33480e+00, -6.09985e-01])
        trdTrackEcalCogDeltaXUncertainty = TrdEcalMatchingSpatialResolutionFunction(E, par = [4.74994e+00, -2.20374e-01, 4.95303e+00])
        trdTrackEcalCogDeltaYUncertainty = TrdEcalMatchingSpatialResolutionFunction(E , par = [1.82444e+01, -1.98260e-01, 5.07107e+00])

        return 0.25 *((events.TrdTrackEcalCogAngleXZ/trdTrackEcalCogAngleXZUncertainty)** 2 + (events.TrdTrackEcalCogAngleYZ/trdTrackEcalCogAngleYZUncertainty)**2 \
        + (events.TrdTrackEcalCogDeltaX/trdTrackEcalCogDeltaXUncertainty)**2 + (events.TrdTrackEcalCogDeltaY/trdTrackEcalCogDeltaYUncertainty)**2)   


#Ecal preselection

def CutEcalCentreOfGravityX(events):       
    selection = np.abs(ak.to_numpy(events.EcalCentreOfGravityX)) < 31.5
    return events[selection]

def CutEcalCentreOfGravityY(events):        
    selection = np.abs(ak.to_numpy(events.EcalCentreOfGravityY)) < 31.5
    return events[selection]

#TRD preselection

def CutTrdHasUsefulTrack(events):       
    selection = (events.TrdTrackNumberOfSubLayersXZ > 6) & (events.TrdTrackNumberOfSubLayersYZ > 4)\
        & (events.TrdTrackFirstSubLayerYZ > 20) & (events.TrdTrackLastSubLayerYZ < 10)
    return events[selection]

def CutEcalTrdTrackMachingChi2(events):
    TrdEcalChi2 = ComputeTrdEcalChi2(events)    
    selection= (TrdEcalChi2 >= 0.0) & (TrdEcalChi2 <= 6.0)
    return events[selection]

#TOF preselection

def CutTofNumberOfLayers(events):   
    selection = (events.TofNumberOfLayers > 2)
    return events[selection]

def CutTofTimeDifference(events):       
    selection = (events.TofDeltaT >= 0) & (events.TofDeltaT <= 20)
    return events[selection]

def CutTofTrdMatching(events):
    selection = (events.TofTrdMatchNorm >= 0) & (events.TofTrdMatchNorm <= 30)
    return events[selection]

def CutTofIsDowngoing(events):   
    selection = (events.TofBeta > 0)
    return events[selection]    
        
#Quality preselection
        
def CutEcalShowerWithMaximumEnergyIsUsed(events): ###     
    selection =  (np.abs(ak.to_numpy(events.EcalEnergyElectronNew) - ak.to_numpy(events.TotalEnergy3D)) <1e-5)  
    return events[selection]


def CutIsNotInSolarArrayShadow(events):    
    selection = events.TrackerTrackIsNotInSolarArrayShadow       
    return events[selection]   
    
    
    
    
    
    
    
    












    
