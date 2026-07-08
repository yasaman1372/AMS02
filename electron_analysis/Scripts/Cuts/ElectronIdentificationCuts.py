#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 10:51:11 2022

@author: yasaman
"""
import uproot
import awkward as ak
import numpy as np
from scipy import interpolate
from Cuts.ElectronTagCuts import *

address= '/home/op115134/Software/YasamanAnalysis/RootFiles/'

with uproot.open(address + 'CutValue.root') as file:
        Lateralcut = file['CutValues']
Lateralcutx = Lateralcut.members['fX']
Lateralcuty = Lateralcut.members['fY']
Lateral_cut = interpolate.interp1d(Lateralcutx,Lateralcuty,fill_value="extrapolate")

with uproot.open(address + 'LeptonAnalysis_EcalChiSquareLateralNormalizedCalibration.root') as file:

    meanMCgraf = file['meanMCGraphSmoothed']
    sigmaMCgraf = file ['sigmaMCGraphSmoothed']

    meanISSgraf = file['meanISSGraphSmoothed']
    sigmaISSgraf = file['sigmaISSGraphSmoothed']

    meanMC_int = interpolate.interp1d(meanMCgraf.all_members["fX"], meanMCgraf.all_members["fY"],fill_value="extrapolate")
    sigmaMC_int = interpolate.interp1d(sigmaMCgraf.all_members["fX"], sigmaMCgraf.all_members["fY"],fill_value="extrapolate")

    meanISS_int = interpolate.interp1d(meanISSgraf.all_members["fX"], meanISSgraf.all_members["fY"],fill_value="extrapolate")
    sigmaISS_int = interpolate.interp1d(sigmaISSgraf.all_members["fX"], sigmaISSgraf.all_members["fY"],fill_value="extrapolate")
    
def EcalChiSquareLateralNormalizedCorrected(nchi2, EcalEnergyElectron, data_type, meanMC, meanISS, sigmaMC, sigmaISS):

    if data_type == "ISS":
        return (nchi2 - meanISS) / sigmaISS

    elif data_type == "MC":
        return (nchi2 - meanMC)/sigmaMC

def ComputeEcalChiSquareLateralNormalized(events, data_type):

    chi2 = ak.to_numpy(events.EcalChiSquareLateral)

    energy = ak.to_numpy(events.TotalEnergy3D)

    x= np.clip(energy,15.01,399.99)

    x=np.log10(x)

    meanISS = meanISS_int(energy)
    meanMC = meanMC_int(energy)

    sigmaISS = sigmaISS_int(energy)
    sigmaMC = sigmaMC_int(energy)

    #parameters from gbatch as of 2015-09-04 (EcalPDF Version 2)
    p0 = 0.281165
    p1 = -0.0493095
    p2 = 0.120408
    p3 = -0.0181409

    nchi2 = chi2 - (p0 + (p1*x) + (p2*x*x) + (p3*x*x*x))
   # return nchi2*(chi2 != 0.0)
   # print(nchi2.shape,chi2.shape,flush=True)
   # print(nchi2)
   
    result= EcalChiSquareLateralNormalizedCorrected(nchi2, energy, data_type, meanMC, meanISS, sigmaMC, sigmaISS)*(chi2 != 0.0)   
   # print(result,nchi2)
    return(result)
    
#address= '/home/op115134/Software/YasamanAnalysis/RootFiles/EnergydependentCuts/'
with uproot.open(address + 'EnergydependentCuts/TrackerEcalMatchingXUpperCut.root') as file:
    upper_cut = file['TrackerEcalMatchingXUpperCut']
    upper_cutx = upper_cut.members['fX']
    upper_cuty = upper_cut.members['fY']
    upperEX_cut = interpolate.interp1d(upper_cutx,upper_cuty,fill_value="extrapolate")

with uproot.open(address + 'EnergydependentCuts/TrackerEcalMatchingXLowerCut.root') as file:
    lower_cut = file['TrackerEcalMatchingXLowerCut']
    lower_cutx = lower_cut.members['fX']
    lower_cuty = lower_cut.members['fY']
    lowerEX_cut = interpolate.interp1d(lower_cutx,lower_cuty,fill_value="extrapolate")

with uproot.open(address + 'EnergydependentCuts/TrackerEcalMatchingYUpperCut.root') as file:
        cut = file['TrackerEcalMatchingYUpperCut']
        cutx = cut.members['fX']
        cuty = cut.members['fY']
        upperEY_cut = interpolate.interp1d(cutx,cuty,fill_value="extrapolate")

def Identification_Cuts_Branch_List():
    branches =['TrackerTrackGBLMaxSpanRigidity','EcalEnergyDepositedMaximumShower',"TotalEnergy3D", 'TrackerTrackChoutkoMaxSpanRigidity', 'EcalChiSquareLateral',
                'TrackerTrackEcalCogDeltaX','TrackerTrackEcalCogDeltaY', 'EcalEnergyDepositedMaximumShower']
    return branches

def CutEnergyOverRigidity(events):
    selection = (EnergyOverRigidity(events) > 0.5) & (EnergyOverRigidity(events) < 10.0)
    return events[selection]

def CutEcalChiSquareLateralNormalized(events, data_type):
    upper_cut = Lateral_cut(events.TotalEnergy3D)
    selection = (ComputeEcalChiSquareLateralNormalized(events, data_type) < upper_cut) | (events.TotalEnergy3D < 10)
   # print("ENERGY", flush=True)
   # print(events.EcalEnergyElectronNewMaximumShower, flush=True)
   # print("UPPER CUT", flush=True)
   # print(upper_cut*(events.EcalEnergyElectronNewMaximumShower <=10), flush=True)
    return events[selection]
    
def CutTrackerTrackEcalCogDeltaX(events):
    upper_cut = upperEX_cut(events.TotalEnergy3D)
    lower_cut = lowerEX_cut(events.TotalEnergy3D)
    selection = (events.TrackerTrackEcalCogDeltaX > lower_cut) & (events.TrackerTrackEcalCogDeltaX <upper_cut)
    return events[selection]
    
def CutTrackerTrackEcalCogDeltaY(events):
    upper_cut = upperEY_cut(events.TotalEnergy3D)
    selection = events.TrackerTrackEcalCogDeltaY < upper_cut
    return events[selection]
    
        
    
