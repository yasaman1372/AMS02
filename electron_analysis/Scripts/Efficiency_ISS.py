#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:31:24 2023

@author: yasaman
"""

import multiprocessing as mp
import os
import numpy as np
from Cuts.ElectronTagCuts import *
from Cuts.ElectronIdentificationCuts import *
from Cuts.ElectronSelectionCuts import *
from Cuts.PreselectionCuts import *
from Cuts.ApplyCuts import *
from tools.roottree import read_tree

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

    resultdir = kwargs["resultdir"]
    
    var1_name= 'TRD Estimator'
    var1_min = 0
    var1_max = 2
    bin1_num = 100
    var1_binning = np.linspace(var1_min, var1_max, bin1_num +1)
    
    var2_name = "EcalEnergyElectronNewMaximumShower" 

    var2_binning = np.array([0.5,0.65,0.82,1.01,1.22,1.46,1.72,2.0,2.31,2.65,3.0,3.36,3.73,
                             4.12,4.54,5.0,5.49,6.0,6.54,7.1,7.69,8.3,8.95,9.62,10.32,11.04,
                             11.8,12.59,13.41,14.25,15.14,16.05,17.0,17.98,18.99,20.04,21.13,
                             22.25,23.42,24.62,25.9,27.25,28.68,30.21,31.82,33.53,35.36,37.31,
                             39.39,41.61,44.0,46.57,49.33,52.33,55.58,59.13,63.02,67.3,72.05,
                             77.37,83.36,90.19,98.08,107.3,118.4,132.1,148.8,169.9,197.7,237.2,
                             290.0,370.0,500.0,700.0,1000.0])
    
    bin2_num = len(var2_binning) -1
    UTTEvents = np.zeros((bin1_num, bin2_num))
    UTCCEvents = np.zeros((bin1_num, bin2_num))
    TCEvents = np.zeros((bin1_num,bin2_num))
    TBEvents = np.zeros((bin1_num,bin2_num)) ##
    THEvents = np.zeros((bin1_num,bin2_num)) ##  
    TPEvents = np.zeros((bin1_num,bin2_num)) ## 
    TCHEvents = np.zeros((bin1_num,bin2_num)) ## 
    ECEvents = np.zeros((bin1_num,bin2_num)) ## 
    EALEvents = np.zeros((bin1_num,bin2_num))
    TTGEvents = np.zeros((bin1_num,bin2_num))
    ERMEvents = np.zeros((bin1_num,bin2_num))
    TEMXEvents = np.zeros((bin1_num,bin2_num)) ##
    TEMYEvents = np.zeros((bin1_num,bin2_num)) ##
    
    UTTEvents_passed = np.zeros((bin1_num,bin2_num))
    UTCCEvents_passed = np.zeros((bin1_num,bin2_num))
    TCEvents_passed = np.zeros((bin1_num,bin2_num))
    TBEvents_passed = np.zeros((bin1_num,bin2_num))
    THEvents_passed = np.zeros((bin1_num,bin2_num))
    TPEvents_passed = np.zeros((bin1_num,bin2_num))
    TCHEvents_passed = np.zeros((bin1_num,bin2_num))
    ECEvents_passed = np.zeros((bin1_num,bin2_num))
    EALEvents_passed = np.zeros((bin1_num,bin2_num))
    TTGEvents_passed = np.zeros((bin1_num,bin2_num))
    ERMEvents_passed = np.zeros((bin1_num,bin2_num))
    TEMXEvents_passed = np.zeros((bin1_num,bin2_num))
    TEMYEvents_passed = np.zeros((bin1_num,bin2_num))
    
    for events in read_tree(filename, treename, chunk_size=chunk_size, rank=rank, nranks=nranks):
        
        
        
        ######################## At least one useful TRD track#####################
        
        UTTevents = NegativeRigidityTag(events)
        UTTevents = HasTrackerTag(UTTevents)
        UTTevents = HasTofTag(UTTevents)
        UTTevents = HasEcalTag(UTTevents)
        UTTevents = EcalElectronTag(UTTevents)
        UTTevents = TofElectronTag(UTTevents)
        UTTevents = TrdHasUsefulSegmentsInBothProjectionsTag(UTTevents)
        UTTevents = EcalPreselectionTag(UTTevents)
        UTTevents_passed = CutTrdHasUsefulTrack(UTTevents)
        UTTevents_passed =  CutEcalTrdTrackMachingChi2(UTTevents_passed)
        
        v1 = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(UTTevents))
        v2 = ak.to_numpy(UTTevents[var2_name])
        
        v1_passed = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(UTTevents_passed))
        v2_passed = ak.to_numpy(UTTevents_passed[var2_name])
        
        
        hist_passed, edges_passed = np.histogramdd((v1_passed,v2_passed), bins=(var1_binning,var2_binning))      
        UTTEvents_passed += hist_passed   
        
        hist, edges = np.histogramdd((v1,v2), bins=(var1_binning,var2_binning))        
        UTTEvents += hist
        #######
        
        ############################ At least one useful TOF cluster combination ###################
        UTCCevents = NegativeRigidityTag(events)
        UTCCevents = HasTrackerTag(UTCCevents)
        UTCCevents = HasTrdTag(UTCCevents)
        UTCCevents = HasEcalTag(UTCCevents)
        UTCCevents = EcalElectronTag(UTCCevents) 
        UTCCevents = TofNumberOfLayersTag(UTCCevents)
        UTCCevents = TofTimeDifferenceTag(UTCCevents) 
        UTCCevents = EcalTrdPreselectionTag(UTCCevents)
        UTCCevents_passed = CutTofNumberOfLayers(UTCCevents)
        UTCCevents_passed = CutTofTimeDifference(UTCCevents_passed)
        UTCCevents_passed = CutTofTrdMatching(UTCCevents_passed)
        UTCCevents_passed = CutTofIsDowngoing(UTCCevents_passed)
        
        v1 = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(UTCCevents))
        v2 = ak.to_numpy(UTCCevents[var2_name])
        
        v1_passed = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(UTCCevents_passed))
        v2_passed = ak.to_numpy(UTCCevents_passed[var2_name])
        
        hist_passed, edges_passed = np.histogramdd((v1_passed,v2_passed), bins= (var1_binning,var2_binning))        
        UTCCEvents_passed += hist_passed    
        
        hist, edges = np.histogramdd((v1,v2), bins= (var1_binning,var2_binning))         
        UTCCEvents += hist
        
################################################### selection cuts#########################################        
        
        ################################ Upper TOF charge ######################
        
        # TCevents = ApplyPreselectionCuts(events)
        # TCevents = NegativeRigidityTag(TCevents)
        # TCevents = EcalElectronTag(TCevents)
        # TCevents = TrdElectronTag(TCevents)
        # TCevents = TrdNoHeliumTag(TCevents)
        # TCevents = TofBetaTag(TCevents)
        # TCevents = TrackerChargeTag(TCevents)
        # TCevents_passed = CutTofUpperCharge(TCevents)
        
        TCevents = ApplyPreselectionCuts(events)
        TCevents = NegativeRigidityTag(TCevents)
    
        
        #selection cuts:
        TCevents = CutTrdActiveLayers(TCevents)
        TCevent = CutTrdNoHelium(TCevents)
        TCevents = CutTrackerPatternSortedByMDR(TCevents)
        TCevents = CutTrackerCharge(TCevents)
        Tcevents = CutTrackerChiSquareY(TCevents)
        
        #identification cuts:
        TCevents = CutEnergyOverRigidity(TCevents)
        TCevents = CutEcalChiSquareLateralNormalized(TCevents, "ISS")
        TCevents = CutTrackerTrackEcalCogDeltaX(TCevents)
        TCevents = CutTrackerTrackEcalCogDeltaY(TCevents)
        
        #other cuts:        
        TCevents = GeomagneticCutoff(TCevents)
        EcalBDT = Ecal_BDT_cut(TCevents.EcalEnergyElectronNewMaximumShower)
        TCevents = TCevents[TCevents.EcalBDT_v7_EnergyD > EcalBDT ]
        
        TCevents_passed = CutTofUpperCharge(TCevents)
        
        v1 = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(TCevents))
        v2 = ak.to_numpy(TCevents[var2_name])
        
        v1_passed = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(TCevents_passed))
        v2_passed = ak.to_numpy(TCevents_passed[var2_name])
        
        hist_passed, edges_passed = np.histogramdd((v1_passed,v2_passed), bins= (var1_binning,var2_binning))        
        TCEvents_passed += hist_passed    
        
        hist, edges = np.histogramdd((v1,v2), bins= (var1_binning,var2_binning))         
        TCEvents += hist
 
        
        ############### Enough active layers in TRD ######################
        
        # EALevents = ApplyPreselectionCuts(events)
        # EALevents = NegativeRigidityTag(EALevents)
        # EALevents = HasTofTag(EALevents)
        # EALevents = EcalElectronTag(EALevents) 
        # EALevents = TrdElectronTag(EALevents)
        # EALevents = TrackerChargeTag(EALevents)
        # EALevents_passed = CutTrdActiveLayers(EALevents)
        
        
        EALevents = ApplyPreselectionCuts(events)
        EALevents = NegativeRigidityTag(EALevents)
        
        #selection cuts
        EALevents = CutTrackerPatternSortedByMDR(EALevents)
        EALevents = CutTrackerCharge(EALevents)
        EALevents = CutTrackerChiSquareY(EALevents)
        EALevents = CutTofBeta(EALevents)
        EALevents = CutTofUpperCharge(EALevents)
        
        #identification cuts
        EALevents = CutEnergyOverRigidity(EALevents)
        EALevents = CutEcalChiSquareLateralNormalized(EALevents, "ISS")
        EALevents = CutTrackerTrackEcalCogDeltaX(EALevents)
        EALevents = CutTrackerTrackEcalCogDeltaY(EALevents)
        
        #other cuts:        
        EALevents = GeomagneticCutoff(EALevents)
        EcalBDT = Ecal_BDT_cut(EALevents.EcalEnergyElectronNewMaximumShower)
        EALevents = EALevents[EALevents.EcalBDT_v7_EnergyD > EcalBDT ]
        
        EALevents_passed = CutTrdActiveLayers(EALevents)
                
        v1 = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(EALevents))
        v2 = ak.to_numpy(EALevents[var2_name])
        
        v1_passed = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(EALevents_passed))
        v2_passed = ak.to_numpy(EALevents_passed[var2_name])
        
        hist_passed, edges_passed = np.histogramdd((v1_passed,v2_passed), bins= (var1_binning,var2_binning))        
        EALEvents_passed += hist_passed    
        
        hist, edges = np.histogramdd((v1,v2), bins= (var1_binning,var2_binning))         
        EALEvents += hist
        
        
        ####################### TrdNoHelium ########################
        
        THevents = ApplyPreselectionCuts(events)
        THevents = NegativeRigidityTag(THevents)
        
        #identification cuts
        THevents = ApplyIdentificationCuts(THevents, data_type = "ISS")
        
        #selection cuts 
        THevents = CutTofBeta(THevents)
        THevents = CutTofUpperCharge(THevents)
        THevents = CutTrdActiveLayers(THevents)
        THevents = CutTrackerPatternSortedByMDR(THevents)
        THevents = CutTrackerCharge(THevents)
        THevents = CutTrackerChiSquareY(THevents)
        
        #other cuts
        THevents = GeomagneticCutoff(THevents)
        EcalBDT = Ecal_BDT_cut(THevents.EcalEnergyElectronNewMaximumShower)
        THevents = THevents[THevents.EcalBDT_v7_EnergyD > EcalBDT ]
        
        THevents_passed = CutTrdNoHelium(THevents)
        
        v1 = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(THevents))
        v2 = ak.to_numpy(THevents[var2_name])
        
        v1_passed = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(THevents_passed))
        v2_passed = ak.to_numpy(THevents_passed[var2_name])
        
        hist_passed, edges_passed = np.histogramdd((v1_passed,v2_passed), bins= (var1_binning,var2_binning))        
        THEvents_passed += hist_passed  
        
        hist, edges = np.histogramdd((v1,v2), bins= (var1_binning,var2_binning))         
        THEvents += hist
        


        ##################### CutTrackerPatternSortedByMDR ###################
        
        TPevents = ApplyPreselectionCuts(events)
        TPevents = NegativeRigidityTag(TPevents)
        
        #identification cuts
        TPevents = ApplyIdentificationCuts(TPevents, data_type = "ISS")
        
        #selection cuts 
        TPevents = CutTofBeta(TPevents)
        TPevents = CutTofUpperCharge(TPevents)
        TPevents = CutTrdActiveLayers(TPevents)
        TPevents = CutTrdNoHelium(TPevents)
        TPevents = CutTrackerCharge(TPevents)
        TPevents = CutTrackerChiSquareY(TPevents)
        
        #other cuts
        TPevents = GeomagneticCutoff(TPevents)
        EcalBDT = Ecal_BDT_cut(TPevents.EcalEnergyElectronNewMaximumShower)
        TPevents = TPevents[TPevents.EcalBDT_v7_EnergyD > EcalBDT ]
        
        TPevents_passed = CutTrackerPatternSortedByMDR(TPevents)
        
        v1 = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(THevents))
        v2 = ak.to_numpy(THevents[var2_name])
        
        v1_passed = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(TPevents))
        v2_passed = ak.to_numpy(TPevents[var2_name])
        
        hist_passed, edges_passed = np.histogramdd((v1_passed,v2_passed), bins= (var1_binning,var2_binning))        
        TPEvents_passed += hist_passed  
        
        hist, edges = np.histogramdd((v1,v2), bins= (var1_binning,var2_binning))         
        TPEvents += hist
          
        ################## CutTrackerCharge ################
        
        TCHevents = ApplyPreselectionCuts(events)
        TCHevents = NegativeRigidityTag(TCHevents)
        
        #selection cuts:
        TCHevents = CutTrackerPatternSortedByMDR(TCHevents)
        TCHevents = CutTofBeta(TCHevents)
        TCHevents = CutTofUpperCharge(TCHevents)  
        TCHevents = CutTrdActiveLayers(TCHevents)
        TCHevents = CutTrdNoHelium(TCHevents)
        TCHevents = CutTrackerChiSquareY(TCHevents)
        
        #identification cuts:
        TCHevents = ApplyIdentificationCuts(TCHevents, data_type = "ISS")  
        
        #other cuts:
        TCHevents = GeomagneticCutoff(TCHevents)    
        EcalBDT = Ecal_BDT_cut(TCHevents.EcalEnergyElectronNewMaximumShower)
        TCHevents = TCHevents[TCHevents.EcalBDT_v7_EnergyD > EcalBDT ]  
        
        TCHevents_passed = CutTrackerCharge(TCHevents)
        
        v1 = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(TCHevents))
        v2 = ak.to_numpy(TCHevents[var2_name])
        
        v1_passed = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(TCHevents_passed))
        v2_passed = ak.to_numpy(TCHevents_passed[var2_name])
        
        hist_passed, edges_passed = np.histogramdd((v1_passed,v2_passed), bins= (var1_binning,var2_binning))        
        TCHEvents_passed += hist_passed    
        
        hist, edges = np.histogramdd((v1,v2), bins= (var1_binning,var2_binning))         
        TCHEvents += hist
        

        ######################Tracker track goodness-of-fit in Y-projection###########################
        
        #Nikos tag cuts:
        
        # TTGevents = ApplyPreselectionCuts(events)
        # TTGevents = NegativeRigidityTag(TTGevents)
        # TTGevents = HasTofTag(TTGevents)
        # TTGevents = EcalElectronTag(TTGevents)
        # TTGevents = TrdActiveLayersTag(TTGevents)
        # TTGevents = TrackerChargeTag(TTGevents)
        # TTGevents = GoldenTrackerPatternTag(TTGevents)
        # TTGevents_passed = CutTrackerChiSquareY(TTGevents)
        
        TTGevents = ApplyPreselectionCuts(events)
        TTGevents = NegativeRigidityTag(TTGevents)
        
        #selection cuts:
        TTGevents = CutTrackerPatternSortedByMDR(TTGevents)
        TTGevents = CutTrackerCharge(TTGevents)
        TTGevents = CutTofBeta(TTGevents)
        TTGevents = CutTofUpperCharge(TTGevents)  
        TTGevents = CutTrdActiveLayers(TTGevents)
        TTGevents = CutTrdNoHelium(TTGevents)
        
        #identification cuts:
        TTGevents = CutEnergyOverRigidity(TTGevents)
        TTGevents = CutEcalChiSquareLateralNormalized(TTGevents, "ISS")
        TTGevents = CutTrackerTrackEcalCogDeltaX(TTGevents)
        
        #other cuts
        TTGevents = GeomagneticCutoff(TTGevents)
        EcalBDT = Ecal_BDT_cut(TTGevents.EcalEnergyElectronNewMaximumShower)
        TTGevents = TTGevents[TTGevents.EcalBDT_v7_EnergyD > EcalBDT ]
        
        TTGevents_passed = CutTrackerChiSquareY(TTGevents)
         
        
        v1 = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(TTGevents))
        v2 = ak.to_numpy(TTGevents[var2_name])
        
        v1_passed = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(TTGevents_passed))
        v2_passed = ak.to_numpy(TTGevents_passed[var2_name])
        
        hist_passed, edges_passed = np.histogramdd((v1_passed,v2_passed), bins= (var1_binning,var2_binning))        
        TTGEvents_passed += hist_passed    
        
        hist, edges = np.histogramdd((v1,v2), bins= (var1_binning,var2_binning))         
        TTGEvents += hist
        
############################################################identification cuts #####################################################
        
        #################################Energy rigidity matching####################
        
        # ERMevents = ApplyPreselectionCuts(events)
        # ERMevents = ApplySelectionCuts(ERMevents)
        # ERMevents = NegativeRigidityTag(ERMevents)
        # ERMevents = EcalElectronTag(ERMevents)
        # ERMevents = TrdElectronTag(ERMevents)
        # ERMevents_passed = CutEnergyOverRigidity(ERMevents)
        
        ERMevents = ApplyPreselectionCuts(events)
        ERMevents = ApplySelectionCuts(ERMevents)
        ERMevents = NegativeRigidityTag(ERMevents)
        
        #identification cuts
        ERMevents = CutEcalChiSquareLateralNormalized(ERMevents, "ISS")
        ERMevents = CutTrackerTrackEcalCogDeltaX(ERMevents)
        ERMevents = CutTrackerTrackEcalCogDeltaY(ERMevents)
        
        #other cuts
        ERMevents = GeomagneticCutoff(ERMevents)
        EcalBDT = Ecal_BDT_cut(ERMevents.EcalEnergyElectronNewMaximumShower)
        ERMevents = ERMevents[ERMevents.EcalBDT_v7_EnergyD > EcalBDT ]
        
        ERMevents_passed = CutEnergyOverRigidity(ERMevents)
        
        
        v1 = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(ERMevents))
        v2 = ak.to_numpy(ERMevents[var2_name])
        
        v1_passed = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(ERMevents_passed))
        v2_passed = ak.to_numpy(ERMevents_passed[var2_name])
        
        hist_passed, edges_passed = np.histogramdd((v1_passed,v2_passed), bins= (var1_binning,var2_binning))        
        ERMEvents_passed += hist_passed    
        
        hist, edges = np.histogramdd((v1,v2), bins= (var1_binning,var2_binning))         
        ERMEvents += hist
        
        ####################### CutEcalChiSquareLateralNormalized #######################
        
        ECevents = ApplyPreselectionCuts(events)
        ECevents = ApplySelectionCuts(ECevents)
        ECevents = NegativeRigidityTag(ECevents)
            
        #identification cuts
        ECevents = CutEnergyOverRigidity(ECevents)
        ECevents = CutTrackerTrackEcalCogDeltaX(ECevents)
        ECevents = CutTrackerTrackEcalCogDeltaY(ECevents)
            
        #other cuts
        EcalBDT = Ecal_BDT_cut(ECevents.EcalEnergyElectronNewMaximumShower)
        ECevents = ECevents[ECevents.EcalBDT_v7_EnergyD > EcalBDT ]
            
        ECevents_passed = CutEcalChiSquareLateralNormalized(ECevents, "ISS")
            
        v1 = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(ECevents))
        v2 = ak.to_numpy(ECevents[var2_name])
            
        v1_passed = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(ECevents_passed))
        v2_passed = ak.to_numpy(ECevents_passed[var2_name])
            
        hist_passed, edges_passed = np.histogramdd((v1_passed,v2_passed), bins= (var1_binning,var2_binning))        
        ECEvents_passed += hist_passed    
            
        hist, edges = np.histogramdd((v1,v2), bins= (var1_binning,var2_binning))         
        ECEvents += hist
            
            
        
        #########################Tracker ECAL matching in X-projection###########################
        
        # TEMevents = ApplyPreselectionCuts(events)
        # TEMevents = ApplySelectionCuts(TEMevents)
        # TEMevents = NegativeRigidityTag(TEMevents)
        # TEMevents = EcalElectronTag(TEMevents) 
        # TEMevents = TrdElectronTag(TEMevents)
        # TEMevents = ElectronEnergyOverRigidityTag(TEMevents)
        # TEMevents_passed = CutTrackerTrackEcalCogDeltaX(TEMevents)
        
        TEMXevents = ApplyPreselectionCuts(events)
        TEMXevents = ApplySelectionCuts(TEMXevents)
        TEMXevents = NegativeRigidityTag(TEMXevents)
        
        #identification cuts:
        TEMXevents = CutEnergyOverRigidity(TEMXevents)
        TEMXevents = CutEcalChiSquareLateralNormalized(TEMXevents, "ISS")
        TEMXevents = CutTrackerTrackEcalCogDeltaY(TEMXevents)
        
        #other cuts
        TEMXevents = GeomagneticCutoff(TEMXevents)
        EcalBDT = Ecal_BDT_cut(TEMXevents.EcalEnergyElectronNewMaximumShower)
        TEMXevents = TEMXevents[TEMXevents.EcalBDT_v7_EnergyD > EcalBDT ]
        
        TEMXevents_passed = CutTrackerTrackEcalCogDeltaX(TEMXevents)
        
        v1 = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(TEMXevents))
        v2 = ak.to_numpy(TEMXevents[var2_name])
        
        v1_passed = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(TEMXevents_passed))
        v2_passed = ak.to_numpy(TEMXevents_passed[var2_name])
        
        hist_passed, edges_passed = np.histogramdd((v1_passed,v2_passed), bins= (var1_binning,var2_binning))        
        TEMXEvents_passed += hist_passed    
        
        hist, edges = np.histogramdd((v1,v2), bins= (var1_binning,var2_binning))         
        TEMXEvents += hist
        
    ############################# Tracker ECAL matching in Y-projection ####################
                
        TEMYevents = ApplyPreselectionCuts(events)
        TEMYevents = ApplySelectionCuts(TEMYevents)
        TEMYevents = NegativeRigidityTag(TEMYevents)
        
        #identification cuts:
        TEMYevents = CutEnergyOverRigidity(TEMYevents)
        TEMYevents = CutEcalChiSquareLateralNormalized(TEMYevents, "ISS")
        TEMYevents = CutTrackerTrackEcalCogDeltaX(TEMYevents)
        
        #other cuts
        TEMYevents = GeomagneticCutoff(TEMYevents)
        EcalBDT = Ecal_BDT_cut(TEMYevents.EcalEnergyElectronNewMaximumShower)
        TEMYevents = TEMYevents[TEMYevents.EcalBDT_v7_EnergyD > EcalBDT ]
        
        TEMYevents_passed = CutTrackerTrackEcalCogDeltaY(TEMYevents)
        
        v1 = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(TEMYevents))
        v2 = ak.to_numpy(TEMYevents[var2_name])
        
        v1_passed = ak.to_numpy(TrdLRElecProt_Energy_HybridHits_TrdP(TEMYevents_passed))
        v2_passed = ak.to_numpy(TEMYevents_passed[var2_name])
        
        hist_passed, edges_passed = np.histogramdd((v1_passed,v2_passed), bins= (var1_binning,var2_binning))        
        TEMYEvents_passed += hist_passed    
        
        hist, edges = np.histogramdd((v1,v2), bins= (var1_binning,var2_binning))         
        TEMYEvents += hist        
                
        
    
    # this process is done, save result
    #np.savez(os.path.join(resultdir, f"results_{rank}.npz"), TRD_Estimator_binning=var1_binning, Energy_binning = var2_binning, UTTEvents = UTTEvents,UTTEvents_passed = UTTEvents_passed, UTCCEvents = UTCCEvents,UTCCEvents_passed = UTCCEvents_passed, TCEvents = TCEvents,TCEvents_passed = TCEvents_passed, EALEvents = EALEvents, EALEvents_passed = EALEvents_passed, TTGEvents = TTGEvents,TTGEvents_passed = TTGEvents_passed, ERMEvents = ERMEvents, ERMEvents_passed = ERMEvents_passed,TEMEvents = TEMEvents,TEMEvents_passed = TEMEvents_passed)
    
    # this process is done, save result
    np.savez(os.path.join(resultdir, f"results_{rank}.npz"), 
             TRD_Estimator_binning=var1_binning, Energy_binning = var2_binning, 
             UTTEvents = UTTEvents,UTTEvents_passed = UTTEvents_passed, 
             UTCCEvents = UTCCEvents,UTCCEvents_passed = UTCCEvents_passed, 
             TCEvents = TCEvents,TCEvents_passed = TCEvents_passed,
             TBEvents = TBEvents,TBEvents_passed = TBEvents_passed,
             THEvents = THEvents,THEvents_passed = THEvents_passed,
             TPEvents = TPEvents,TPEvents_passed = TPEvents_passed,
             TCHEvents = TCHEvents,TCHEvents_passed = TCHEvents_passed,
             ECEvents = ECEvents,ECEvents_passed = ECEvents_passed,
             EALEvents = EALEvents, EALEvents_passed = EALEvents_passed, 
             TTGEvents = TTGEvents,TTGEvents_passed = TTGEvents_passed, 
             ERMEvents = ERMEvents, ERMEvents_passed = ERMEvents_passed,
             TEMYEvents = TEMYEvents,TEMYEvents_passed = TEMYEvents_passed,
             TEMXEvents = TEMXEvents,TEMXEvents_passed = TEMXEvents_passed)
    
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
    Energy_binning  = None
    TRD_Estimator_binning = None
    UTTEvents = None
    for rank in range(args.nprocesses):
        filename = os.path.join(args.resultdir, f"results_{rank}.npz")
        with np.load(filename) as result_file:
            if Energy_binning is None:
                Energy_binning = result_file["Energy_binning"]
                TRD_Estimator_binning = result_file["TRD_Estimator_binning"]
            else:
                # make sure all processes used the same binning
                assert np.all(Energy_binning == result_file["Energy_binning"])
                #assert np.all(TRD_Estimator_binning == result_file["TRD_Estimator_binning"])
             
                
            if UTTEvents is None:
                UTTEvents = result_file["UTTEvents"]
                UTCCEvents = result_file["UTCCEvents"]
                TCEvents = result_file["TCEvents"]
                TBEvents = result_file["TBEvents"]
                THEvents = result_file["THEvents"]
                TPEvents = result_file["TPEvents"]
                TCHEvents = result_file["TCHEvents"]
                ECEvents = result_file["ECEvents"]
                EALEvents = result_file["EALEvents"]
                TTGEvents = result_file["TTGEvents"]
                ERMEvents = result_file["ERMEvents"]
                TEMXEvents = result_file["TEMXEvents"]
                TEMYEvents = result_file["TEMYEvents"]

                UTTEvents_passed = result_file["UTTEvents_passed"]
                UTCCEvents_passed = result_file["UTCCEvents_passed"]
                TCEvents_passed = result_file["TCEvents_passed"]
                TBEvents_passed = result_file["TBEvents_passed"]
                THEvents_passed = result_file["THEvents_passed"]
                TPEvents_passed = result_file["TPEvents_passed"]
                TCHEvents_passed = result_file["TCHEvents_passed"]
                ECEvents_passed = result_file["ECEvents_passed"]
                EALEvents_passed = result_file["EALEvents_passed"]
                TTGEvents_passed = result_file["TTGEvents_passed"]
                ERMEvents_passed = result_file["ERMEvents_passed"]
                TEMXEvents_passed = result_file["TEMXEvents_passed"]
                TEMYEvents_passed = result_file["TEMYEvents_passed"]
                
      
            else:
                UTTEvents += result_file["UTTEvents"]
                UTCCEvents += result_file["UTCCEvents"]
                TCEvents += result_file["TCEvents"]
                TBEvents += result_file["TBEvents"]
                THEvents += result_file["THEvents"]
                TPEvents += result_file["TPEvents"]
                TCHEvents += result_file["TCHEvents"]
                ECEvents += result_file["ECEvents"]
                EALEvents += result_file["EALEvents"]
                TTGEvents += result_file["TTGEvents"]
                ERMEvents += result_file["ERMEvents"]
                TEMXEvents += result_file["TEMXEvents"]
                TEMYEvents += result_file["TEMYEvents"]

                UTTEvents_passed += result_file["UTTEvents_passed"]
                UTCCEvents_passed += result_file["UTCCEvents_passed"]
                TCEvents_passed += result_file["TCEvents_passed"]
                TBEvents_passed += result_file["TBEvents_passed"]
                THEvents_passed += result_file["THEvents_passed"]
                TPEvents_passed += result_file["TPEvents_passed"]
                TCHEvents_passed += result_file["TCHEvents_passed"]
                ECEvents_passed += result_file["ECEvents_passed"]
                EALEvents_passed += result_file["EALEvents_passed"]
                TTGEvents_passed += result_file["TTGEvents_passed"]
                ERMEvents_passed += result_file["ERMEvents_passed"]
                TEMXEvents_passed += result_file["TEMXEvents_passed"]
                TEMYEvents_passed += result_file["TEMYEvents_passed"]




    # now save merged result
    #np.savez(os.path.join(args.resultdir, "Efficiency_corrections_ISS.npz"), TRD_Estimator_binning = TRD_Estimator_binning, Energy_binning = Energy_binning, UTTEvents = UTTEvents,UTTEvents_passed = UTTEvents_passed, UTCCEvents = UTCCEvents,UTCCEvents_passed = UTCCEvents_passed, TCEvents = TCEvents,TCEvents_passed = TCEvents_passed, EALEvents = EALEvents, EALEvents_passed = EALEvents_passed, TTGEvents = TTGEvents,TTGEvents_passed = TTGEvents_passed, ERMEvents = ERMEvents, ERMEvents_passed = ERMEvents_passed,TEMEvents = TEMEvents,TEMEvents_passed = TEMEvents_passed)   
    np.savez(os.path.join(args.resultdir, "Efficiency_corrections_MC.npz"), 
             TRD_Estimator_binning = TRD_Estimator_binning, Energy_binning = Energy_binning,
             UTTEvents = UTTEvents, UTTEvents_passed = UTTEvents_passed, 
             UTCCEvents = UTCCEvents, UTCCEvents_passed = UTCCEvents_passed, 
             TCEvents = TCEvents, TCEvents_passed = TCEvents_passed,
             TBEvents = TBEvents, TBEvents_passed = TBEvents_passed,
             THEvents = THEvents,THEvents_passed = THEvents_passed,
             TPEvents = TPEvents,TPEvents_passed = TPEvents_passed,
             TCHEvents = TCHEvents,TCHEvents_passed = TCHEvents_passed,
             ECEvents = ECEvents,ECEvents_passed = ECEvents_passed,
             EALEvents = EALEvents, EALEvents_passed = EALEvents_passed, 
             TTGEvents = TTGEvents,TTGEvents_passed = TTGEvents_passed, 
             ERMEvents = ERMEvents, ERMEvents_passed = ERMEvents_passed,
             TEMYEvents = TEMYEvents,TEMYEvents_passed = TEMYEvents_passed,
             TEMXEvents = TEMXEvents,TEMXEvents_passed = TEMXEvents_passed)   
    
    
        
if __name__ == "__main__":
    main()
        
