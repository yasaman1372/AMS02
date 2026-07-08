#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 10:21:35 2022

@author: yasaman
"""

import awkward as ak
import numpy as np


###################################
#Helper Tags
###################################
def TrdEcalMatchingAngularResolutionFunction(E,par):
	return par[0] + par[1]* np.maximum(E,1e-7)**par[2]

def TrdEcalMatchingSpatialResolutionFunction(E,par):
	return par[0]/(E + par[1]+ 1e-7) + par[2]

def ComputeTrdEcalChi2(events):
        E = events.TotalEnergy3D
       # E = events.EcalEnergyDepositedMaximumShower
        trdTrackEcalCogAngleXZUncertainty = TrdEcalMatchingAngularResolutionFunction(E,par=[2.45281e-02, 7.88632e-01, -6.86901e-01])
        trdTrackEcalCogAngleYZUncertainty = TrdEcalMatchingAngularResolutionFunction(E, par =[1.67174e-02, 1.33480e+00, -6.09985e-01])
        trdTrackEcalCogDeltaXUncertainty = TrdEcalMatchingSpatialResolutionFunction(E, par = [4.74994e+00, -2.20374e-01, 4.95303e+00])
        trdTrackEcalCogDeltaYUncertainty = TrdEcalMatchingSpatialResolutionFunction(E , par = [1.82444e+01, -1.98260e-01, 5.07107e+00])

        return 0.25 *((events.TrdTrackEcalCogAngleXZ/trdTrackEcalCogAngleXZUncertainty)** 2 + (events.TrdTrackEcalCogAngleYZ/trdTrackEcalCogAngleYZUncertainty)**2 \
        + (events.TrdTrackEcalCogDeltaX/trdTrackEcalCogDeltaXUncertainty)**2 + (events.TrdTrackEcalCogDeltaY/trdTrackEcalCogDeltaYUncertainty)**2)


#def ComputeTrackerPatternSortedByMDR(events):
   # Events= ak.to_numpy(events.TrackerPattern)
    #Events[Events==0]=1 # Layer 1 and 9, and maybe 2
    #Events[Events==1]=3 #Layer 1 and 2, but not 9
    #Events[Events==2]=2 #Layer 2 and 9, but not 1
    #Events[Events==3]=5 #Layer 1
    #Events[Events==4]=6 #Layer 2
    #Events[Events==5]=4 #Layer 9
    #Events[Events==-1]=0 #None of the above
    #return Events

def IsMC(events):
    return (events.McGeneratedMomentum > 0.0)

def IsISS(events):
    return ((events.McGeneratedMomentum ==0.0) & (events.BtNominalMomentum ==0))

def EnergyOverRigidity(events):
   # if version =="pass6":
       # return (ak.to_numpy(events.EcalEnergyDepositedMaximumShower)/(np.abs(ak.to_numpy(events.TrackerTrackChoutkoMaxSpanRigidity))+1e-7)) * ak.to_numpy(events.TrackerTrackChoutkoMaxSpanRigidity != 0.0)
   # elif version =="pass8":
    return (ak.to_numpy(events.TotalEnergy3D)/(np.abs(ak.to_numpy(events.TrackerTrackGBLMaxSpanRigidity))+1e-7)) * ak.to_numpy(events.TrackerTrackGBLMaxSpanRigidity != 0.0)
    #return np.nan_to_num ((ak.to_numpy(events.EcalEnergyElectronNewMaximumShower)/(np.abs(ak.to_numpy(events.TrackerTrackChoutkoMaxSpanRigidity))))*ak.to_numpy(events.TrackerTrackChoutkoMaxSpanRigidity != 0.0), nan=0)
   # return (ak.to_numpy(events.EcalEnergyDepositedMaximumShower)/(np.abs(ak.to_numpy(events.EcalEnergyDepositedMaximumShower))+1e-7)) * ak.to_numpy(events.TrackerTrackChoutkoMaxSpanRigidity != 0.0)
    #return (ak.to_numpy(events.TotalEnergy3D)/(np.abs(ak.to_numpy(events.TrackerTrackGBLMaxSpanRigidity))+1e-7)) * ak.to_numpy(events.TrackerTrackGBLMaxSpanRigidity != 0.0)
    ##### Niko uses EcalEnergyDepositedMaximumShower 
def CalculateLikelihoodRatio(liklihood1, liklihood2):
    #return np.nan_to_num( - np.log(liklihood1/(liklihood1 +liklihood2+ 1e-7))*(liklihood1 >0) - (liklihood1 <=0), nan = -1) 
    return np.nan_to_num( - np.log(liklihood1/(liklihood1 +liklihood2)), nan =-1)
        
def TrdLRHeliElec_Rigidity_HybridHits_TrdP_Raw(events):
    return CalculateLikelihoodRatio(ak.to_numpy(events.TrdPLikelihoodHybridHitsHelium), ak.to_numpy(events.TrdPLikelihoodHybridHitsElectron))

def TrdLRElecProt_Energy_HybridHits_TrdP_Raw(events):
    return CalculateLikelihoodRatio(ak.to_numpy(events.TrdPLikelihoodHybridHitsElectronECAL),ak.to_numpy(events.TrdPLikelihoodHybridHitsProtonECAL))
    
def TrdLRHeliElec_Rigidity_HybridHits_TrdP(events):
    result = TrdLRHeliElec_Rigidity_HybridHits_TrdP_Raw(events) ##### calbration factor should be added later for ISS data
    return result

def TrdLRElecProt_Energy_HybridHits_TrdP(events)   :
    result = TrdLRElecProt_Energy_HybridHits_TrdP_Raw(events) ##### calbration factor should be added later for ISS data
    return(result)        
                    
def TofNumberOfLayersTag(events):
    selection = events.TofNumberOfLayers > 2
    return events[selection]

def TofTimeDifferenceTag(events):
    selection = (events.TofDeltaT >= 0) & (events.TofDeltaT <= 20)
    return events[selection]

def TofTrdMatchingTag(events):
    selection = (events.TofTrdMatchNorm >= 0) & (events.TofTrdMatchNorm <= 30)
    return events[selection]

def TofBetaTag(events):
    selection = (events.TofBeta > 0.8) & (events.TofBeta <= 1.25)
    return events[selection]

def TofUpperChargeTag(events):
    selection = (events.TofUpperCharge > 0.5) & (events.TofUpperCharge < 1.8)
    return events[selection]

def EcalPreselectionTag(events):
    selection = (events.EcalCentreOfGravityX <32.0)\
        & (events.EcalCentreOfGravityY <32.0)
    return events[selection]   

def EcalElectronBdtTag(events):
    selection = (events.EcalShowerDirectionZ < 0)\
        & (events.EcalBDT_v7_EnergyD > 0.5)          
    return events[selection]

def EcalTrdPreselectionTag(events):
    events = EcalPreselectionTag(events)
    selection = (events.TrdTrackNumberOfSubLayersXZ > 6) & (events.TrdTrackNumberOfSubLayersYZ > 4) & (events.TrdTrackFirstSubLayerYZ > 20) & (events.TrdTrackLastSubLayerYZ < 10) & (ComputeTrdEcalChi2(events) >= 0.0) & (ComputeTrdEcalChi2(events) <= 6.0)
    return events[selection]  

def EcalTrdTofPreselectionTag(events):
    events = EcalTrdPreselectionTag(events)
    events = TofNumberOfLayersTag(events)
    events = TofTimeDifferenceTag(events)
    events = TofTrdMatchingTag(events)    
    return events

def EcalProtonBdtTag(events):
    selection = (events.EcalShowerDirectionZ < 0) & (events.EcalBDT_v7_EnergyD < -0.9)
    return events[selection]

def TrackerChargeTag(events):
    selection = (events.TrackerCharge > 0.5) & (events.TrackerCharge < 1.8)
    return events[selection]
        
def GoldenTrackerPatternTag(events):
    selection = (ComputeTrackerPatternSortedByMDR(events) >= 1) & (ComputeTrackerPatternSortedByMDR(events) <= 2)
    return events[selection]

def SingleTrackerTrackTag(events):
    selection = (events.TrackerNumberOfTracks == 1)
    return events[selection]

def TrackerChiSquareTag(events):
    selection = (events.TrackerTrackChoutkoMaxSpanChiSquareY  > 0.01) & (events.TrackerTrackChoutkoMaxSpanChiSquareY  < 25.0)
    return events[selection]

def TrackerInEcalAcceptanceTag(events):
    selection = (abs(ak.to_numpy(events.TrackerTrackAtEcalTopX)) < 32.0) \
       & (abs(ak.to_numpy(events.TrackerTrackAtEcalTopY)) < 32.0)\
       & (abs(ak.to_numpy(events.TrackerTrackAtEcalBottomX)) < 32.0)\
       & (abs(ak.to_numpy(events.TrackerTrackAtEcalBottomY)) < 32.0)
    return events[selection]

def TrdActiveLayersTag(events):
    selection = events.TrdPActiveLayersHybrid > 15
    return events[selection]
    
def TrdNoHeliumTag(events):
    selection = TrdLRHeliElec_Rigidity_HybridHits_TrdP(events) > 0.8
    return events[selection]

def TrdHasUsefulSegmentsInBothProjectionsTag(events):
    selection = (events.TrdMaxSubLayersXZ > 6) & (events.TrdMaxSubLayersYZ > 4)\
        & (events.TrdFirstSubLayerYZ > 20) & (events.TrdLastSubLayerYZ < 10)
    return events[selection]  
 
#def  TrdHasUsefulTrackTag(events):
    

def ElectronTrdLikelihoodRatioOnlyTag(events):
     TrdPLRElecProtECAL = TrdLRElecProt_Energy_HybridHits_TrdP(events)
     selection = TrdPLRElecProtECAL < 0.75
     return events[selection]
 
def ProtonTrdLikelihoodRatioOnlyTag(events):
    selection = events.TrdPLRElecProtECAL > 0.8
    return events[selection]    

def UnbiasedElectronEnergyOverRigidityTag(events):
    E = EnergyOverRigidity(events)
    selection = ((E>0)&(E>0.5))|(E<=0)
    return events[selection]     
    
def ElectronEnergyOverRigidityTag(events):
  selection = EnergyOverRigidity(events) > 0.5
  return events[selection]
    
def ElectronUpperEnergyOverRigidityTag(events):
    selection = EnergyOverRigidity(events) <10.0
    return events[selection]

def UnbiasedProtonEnergyOverRigidityTag(events):
    E = EnergyOverRigidity(events)
    selection = ((E > 0) & (E < 0.3)) | (E <= 0)
    return events[selection]

def NegativeRigidityTag(events):
    selection = events.TrackerTrackChoutkoMaxSpanRigidity < 0
    return events[selection]

def PositiveRigidityTag(events):
    selection = events.TrackerTrackChoutkoMaxSpanRigidity > 0
    return events[selection]    
    
##############################
# tags for preselection
##############################

def HasTrackerTag(events):    
    selection = events.TrackerTrackChoutkoMaxSpanRigidity != 0
    events = events[selection]
    events = GoldenTrackerPatternTag(events)
    events = SingleTrackerTrackTag(events)
    events = TrackerChargeTag(events)
    events = TrackerChiSquareTag(events)
    events = TrackerInEcalAcceptanceTag(events)
    return events

def HasTrdTag(events):
    events = TrdActiveLayersTag(events)
    events = TrdNoHeliumTag(events) 
    return events

def HasTofTag(events):
    events = TofUpperChargeTag(events)
    events = TofBetaTag(events)
    return events


def HasEcalTag(events):
    selection = events.EcalBDT_v7_EnergyD >= -1.0
    return events[selection]  

def TrdElectronTag(events):
    events = ElectronTrdLikelihoodRatioOnlyTag(events)   
    return events

def TrdProtonTag(events):
    events = ProtonTrdLikelihoodRatioOnlyTag(events)
    return events

def TofElectronTag(events):
    selection = abs(1.0 - 1/ak.to_numpy(events.TofBeta)) < 0.15
    return events[selection]

def TofProtonTag(events):
    selection =   (events.TofBeta > 0.5) & (events.TofBeta < 1.5)
    return events[selection]

# def EcalElectronTag(events):
#     events = UnbiasedProtonEnergyOverRigidityTag(events)
#     events = EcalProtonBdtTag(events)
#     return events

def EcalElectronTag(events):
    events = events[ EnergyOverRigidity(events) > 0.5]
    events = events[events.EcalShowerDirectionZ > 0]
    events = events [events.EcalBDT_v7_EnergyD > 0]  
    return events
    
def SoftEcalElectronTag(events):
    if events.EcalBDT_v7_EnergyD > -1.5:
        selection = events.EcalBDT_v7_EnergyD > -0.5
        return events[selection]
    else:
        return events

def SoftEcalProtonTag(events):
    if events.EcalBDT_v7_EnergyD > -1.5:
        selection = events.EcalBDT_v7_EnergyD < -0.7
        return events[selection]
    else:
        return events
    

#############################    
# Tags for selection
#############################

def TofSelectionTag(events):
    events = TofBetaTag(events)
    selection = events.TofUpperCharge < 2.0
    return events[selection]
    
def TrdSelectionTag(events):
    events = TrdActiveLayersTag(events)
    events = TrdNoHeliumTag(events)
    return events

###############
#Geomagnetic cut off
################

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
    
    
    
    
    

    
    
               
        
