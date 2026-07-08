#ifndef LeptonAnalysisTree_hh
#define LeptonAnalysisTree_hh

#include <cassert>
#include <cmath>

#include "RunHeader.h"
#include "TreeInterface.hh"
#include "TrdLikelihoodRatioCalibration.hh"
#include "AnalysisSettings.hh"

namespace Analysis {
  class Event;
  class TrdSegment;
}

namespace Mva {
  class MvaImplementation;
}

namespace IO {
  class FileManager;
}

namespace Utilities {
  class ObjectManager;
}

class LeptonAnalysisTree : public IO::TreeInterface {
public:
  LeptonAnalysisTree();

  // Event information
  IO::TreeBranch<UInt_t>   Run                { "Run",                  0 };
  IO::TreeBranch<UInt_t>   Event              { "Event",                0 };
  IO::TreeBranch<UInt_t>   Time               { "Time",                 0 };
  IO::TreeBranch<UShort_t> AMSVersion         { "AMSVersion",           0 };
  IO::TreeBranch<UChar_t>  TriggerFlags       { "TriggerFlags",         0 };
 // IO::TreeBranch<UInt_t> Time                         { "Time"                 ,  0   };
 // IO::TreeBranch<Int_t> EventNumber                   { "EventNumber"          , -1   };
 // IO::TreeBranch<Double_t> Weight                     { "Weight"               , -1   }; 
 
  // Beam-test information
  IO::TreeBranch<UChar_t>  BtParticleId               { "BtParticleId",               IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  BtNominalMomentum          { "BtNominalMomentum",                                       0.0 };

  // Monte-Carlo generator information
  IO::TreeBranch<UChar_t>  McParticleId               { "McParticleId",               IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  McGeneratedMomentum        { "McGeneratedMomentum",                                     0.0 };
  IO::TreeBranch<Double_t> McEventWeightElectron      { "McEventWeightElectron",                                   0.0 };
  IO::TreeBranch<Double_t> McEventWeightPositron      { "McEventWeightPositron",                                   0.0 };
  IO::TreeBranch<Float_t>  McGeneratedPositionX       { "McGeneratedPositionX",                                 -999.0 };
  IO::TreeBranch<Float_t>  McGeneratedPositionY       { "McGeneratedPositionY",                                 -999.0 };
  IO::TreeBranch<Float_t>  McGeneratedPositionZ       { "McGeneratedPositionZ",                                 -999.0 };
  IO::TreeBranch<Float_t>  McGeneratedDirectionTheta  { "McGeneratedDirectionTheta",                            -999.0 };
  IO::TreeBranch<Float_t>  McGeneratedDirectionPhi    { "McGeneratedDirectionPhi",                              -999.0 };

  // TOF information
  IO::TreeBranch<UChar_t> TofNumberOfLayers { "TofNumberOfLayers",    0 };
  IO::TreeBranch<Float_t> TofBeta           { "TofBeta",         -999.0 };
  IO::TreeBranch<Float_t> TofDeltaT         { "TofDeltaT",       -999.0 };
  IO::TreeBranch<Float_t> TofLowerCharge    { "TofLowerCharge",     0.0 };
  IO::TreeBranch<Float_t> TofUpperCharge    { "TofUpperCharge",     0.0 };
  IO::TreeBranch<Float_t> TofTrdMatchNorm   { "TofTrdMatchNorm", -999.0 };

  // Tracker information
  IO::TreeBranch<UChar_t>              TrackerNumberOfTracks                                 { "TrackerNumberOfTracks",                                   0 };
  IO::TreeBranch<UChar_t>              TrackerNumberOfVertices                               { "TrackerNumberOfVertices",                                 0 };
  IO::TreeBranch<Float_t>              TrackerMinDeltaXAtTRDCenterAllTracksToReference       { "TrackerMinDeltaXAtTRDCenterAllTracksToReference",   9999.9f };
  IO::TreeBranch<Float_t>              TrackerMinDeltaYAtTRDCenterAllTracksToReference       { "TrackerMinDeltaYAtTRDCenterAllTracksToReference",   9999.9f };
  IO::TreeBranch<Float_t>              TrackerMinDeltaXAtLayer4AllTracksToReference          { "TrackerMinDeltaXAtLayer4AllTracksToReference",      9999.9f };
  IO::TreeBranch<Float_t>              TrackerMinDeltaYAtLayer4AllTracksToReference          { "TrackerMinDeltaYAtLayer4AllTracksToReference",      9999.9f };
  IO::TreeBranch<Float_t>              TrackerMinDeltaXAtLayer9AllTracksToReference          { "TrackerMinDeltaXAtLayer9AllTracksToReference",      9999.9f };
  IO::TreeBranch<Float_t>              TrackerMinDeltaYAtLayer9AllTracksToReference          { "TrackerMinDeltaYAtLayer9AllTracksToReference",      9999.9f };
  IO::TreeBranch<Float_t>              TrackerChargeError                                    { "TrackerChargeError",                                   0.0f };
  IO::TreeBranch<Float_t>              TrackerCharge                                         { "TrackerCharge",                                        0.0f };
  IO::TreeBranch<std::vector<Float_t>> TrackerCharges                                        { "TrackerCharges",                      IO::TreeVectorSize(9) };
  IO::TreeBranch<std::vector<Float_t>> TrackerClusterSignalRatios                            { "TrackerClusterSignalRatios",          IO::TreeVectorSize(9) };
  IO::TreeBranch<std::vector<Float_t>> TrackerClusterDistances                               { "TrackerClusterDistances",             IO::TreeVectorSize(9) };
  IO::TreeBranch<std::vector<Float_t>> TrackerRigiditiesWithoutHitInLayer                    { "TrackerRigiditiesWithoutHitInLayer",  IO::TreeVectorSize(9) };
  IO::TreeBranch<Float_t>              TrackerSumOfUnusedRigidities                          { "TrackerSumOfUnusedRigidities",                         0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackKalmanMaxSpanRigidity                     { "TrackerTrackKalmanMaxSpanRigidity",                    0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackKalmanMaxSpanChiSquareX                   { "TrackerTrackKalmanMaxSpanChiSquareX",                  0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackKalmanMaxSpanChiSquareY                   { "TrackerTrackKalmanMaxSpanChiSquareY",                  0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackKalmanMaxSpanRigidityRelError             { "TrackerTrackKalmanMaxSpanRigidityRelError",            0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackGBLMaxSpanRigidity                        { "TrackerTrackGBLMaxSpanRigidity",                       0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackGBLMaxSpanChiSquareX                      { "TrackerTrackGBLMaxSpanChiSquareX",                     0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackGBLMaxSpanChiSquareY                      { "TrackerTrackGBLMaxSpanChiSquareY",                     0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackGBLMaxSpanRigidityRelError                { "TrackerTrackGBLMaxSpanRigidityRelError",               0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackChoutkoMaxSpanRigidity                    { "TrackerTrackChoutkoMaxSpanRigidity",                   0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackChoutkoMaxSpanChiSquareX                  { "TrackerTrackChoutkoMaxSpanChiSquareX",                 0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackChoutkoMaxSpanChiSquareY                  { "TrackerTrackChoutkoMaxSpanChiSquareY",                 0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackChoutkoMaxSpanRigidityRelError            { "TrackerTrackChoutkoMaxSpanRigidityRelError",           0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackChoutkoMaxSpanElecMassRigidity            { "TrackerTrackChoutkoMaxSpanElecMassRigidity",           0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackChoutkoMaxSpanElecMassChiSquareX          { "TrackerTrackChoutkoMaxSpanElecMassChiSquareX",         0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackChoutkoMaxSpanElecMassChiSquareY          { "TrackerTrackChoutkoMaxSpanElecMassChiSquareY",         0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackChoutkoMaxSpanElecMassRigidityRelError    { "TrackerTrackChoutkoMaxSpanElecMassRigidityRelError",   0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackChoutkoInnerOnlyRigidity                  { "TrackerTrackChoutkoInnerOnlyRigidity",                 0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackChoutkoInnerOnlyChiSquareY                { "TrackerTrackChoutkoInnerOnlyChiSquareY",               0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackChoutkoInnerOnlyRigidityRelError          { "TrackerTrackChoutkoInnerOnlyRigidityRelError",         0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackChoutkoUpperHalfRigidity                  { "TrackerTrackChoutkoUpperHalfRigidity",                 0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackChoutkoUpperHalfChiSquareY                { "TrackerTrackChoutkoUpperHalfChiSquareY",               0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackChoutkoUpperHalfRigidityRelError          { "TrackerTrackChoutkoUpperHalfRigidityRelError",         0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackChoutkoLowerHalfRigidity                  { "TrackerTrackChoutkoLowerHalfRigidity",                 0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackChoutkoLowerHalfChiSquareY                { "TrackerTrackChoutkoLowerHalfChiSquareY",               0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackChoutkoLowerHalfRigidityRelError          { "TrackerTrackChoutkoLowerHalfRigidityRelError",         0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackChikanianMaxSpanRigidity                  { "TrackerTrackChikanianMaxSpanRigidity",                 0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackChikanianMaxSpanChiSquareY                { "TrackerTrackChikanianMaxSpanChiSquareY",               0.0f };
  IO::TreeBranch<Float_t>              TrackerTrackChikanianMaxSpanRigidityRelError          { "TrackerTrackChikanianMaxSpanRigidityRelError",         0.0f };
  IO::TreeBranch<Char_t>               TrackerPattern                                        { "TrackerPattern",                                         -1 };
  IO::TreeBranch<UShort_t>             TrackerHitPatternY                                    { "TrackerHitPatternY",                                      0 };
  IO::TreeBranch<UShort_t>             TrackerHitPatternXY                                   { "TrackerHitPatternXY",                                     0 };
  IO::TreeBranch<Short_t>              TrackerNumberOfLayersInnerWithYHit                    { "TrackerNumberOfLayersInnerWithYHit",                     -1 };
  IO::TreeBranch<Float_t>              TrackerTrackEcalCogDeltaX                             { "TrackerTrackEcalCogDeltaX",                          -999.0 };
  IO::TreeBranch<Float_t>              TrackerTrackEcalCogDeltaY                             { "TrackerTrackEcalCogDeltaY",                          -999.0 };
  IO::TreeBranch<Float_t>              TrackerTrackAtEcalTopX                                { "TrackerTrackAtEcalTopX",                             -999.0 };
  IO::TreeBranch<Float_t>              TrackerTrackAtEcalTopY                                { "TrackerTrackAtEcalTopY",                             -999.0 };
  IO::TreeBranch<Float_t>              TrackerTrackAtEcalBottomX                             { "TrackerTrackAtEcalBottomX",                          -999.0 };
  IO::TreeBranch<Float_t>              TrackerTrackAtEcalBottomY                             { "TrackerTrackAtEcalBottomY",                          -999.0 };
  IO::TreeBranch<Bool_t>               TrackerTrackIsNotInSolarArrayShadow                   { "TrackerTrackIsNotInSolarArrayShadow",                  true };
  IO::TreeBranch<Float_t>              CCBDT                                                 { "CCBDT",                                              -999.0 };




  // Ecal information
  IO::TreeBranch<UChar_t> EcalNumberOfShowers             { "EcalNumberOfShowers",               0 };
  IO::TreeBranch<UInt_t>  EcalShowerStatus                { "EcalShowerStatus",                  0 };
  IO::TreeBranch<Float_t> EnergyFractionInLastTwoLayers   { "EnergyFractionInLastTwoLayers",   0.0 };
  IO::TreeBranch<Float_t> EcalEnergyDeposited             { "EcalEnergyDeposited",             0.0 };
  IO::TreeBranch<Float_t> EcalEnergyDepositedRaw          { "EcalEnergyDepositedRaw",          0.0 };
  IO::TreeBranch<Float_t> EcalEnergyElectron              { "EcalEnergyElectron",              0.0 };
  IO::TreeBranch<Float_t> EcalEnergyElectronMaximumShower { "EcalEnergyElectronMaximumShower", 0.0 };
  IO::TreeBranch<Float_t> EcalEnergyElectronNew           { "EcalEnergyElectronNew",           0.0 };
  IO::TreeBranch<Float_t> EcalEnergyElectronNewMaximumShower { "EcalEnergyElectronNewMaximumShower", 0.0 };
  IO::TreeBranch<Float_t> EcalEnergyDepositedMaximumShower   { "EcalEnergyDepositedMaximumShower",   0.0 };
  IO::TreeBranch<Float_t> EcalEnergyDepositedAllShowers   { "EcalEnergyDepositedAllShowers",   0.0 };
  IO::TreeBranch<Float_t> EcalChiSquareLongitudinal       { "EcalChiSquareLongitudinal",      -1.0 };
  IO::TreeBranch<Float_t> EcalChiSquareLateral            { "EcalChiSquareLateral",         -999.0 };
  IO::TreeBranch<Float_t> EcalChiSquareCellRatio          { "EcalChiSquareCellRatio",       -999.0 };
  IO::TreeBranch<Float_t> EcalShowerMaximum               { "EcalShowerMaximum",              -1.0 };
  IO::TreeBranch<Float_t> EcalESE_v3                      { "EcalESE_v3",                     -2.0 };
  IO::TreeBranch<Float_t> EcalBDT_v7                      { "EcalBDT_v7",                     -2.0 };
  IO::TreeBranch<Float_t> EcalBDTSmoothed_v7              { "EcalBDTSmoothed_v7",             -2.0 };
  IO::TreeBranch<Float_t> EcalBDTChiSquare_v5             { "EcalBDTChiSquare_v5",            -2.0 };
  IO::TreeBranch<Float_t> EcalBDT_v7_EnergyD              { "EcalBDT_v7_EnergyD",             -2.0 };
  IO::TreeBranch<Float_t> EcalBDTSmoothed_v7_EnergyD      { "EcalBDTSmoothed_v7_EnergyD",     -2.0 };
  IO::TreeBranch<Float_t> EcalBDTChiSquare_v5_EnergyD     { "EcalBDTChiSquare_v5_EnergyD",    -2.0 };
//  IO::TreeBranch<Float_t> BDTEstimator3D                  { "BDTEstimator3D",                 -999.0 };
  IO::TreeBranch<Float_t> EcalBDT3D                       { "EcalBDT3D",                       0.0 };
  IO::TreeBranch<Float_t> TotalEnergy3D                   { "TotalEnergy3D",                   0.0 };
//  IO::TreeBranch<Float_t> EcalBDT                         { "EcalBDT",                         0.0 };
  IO::TreeBranch<Float_t> EcaLIntegralLikelihood3D        { "EcaLIntegralLikelihood3D",        0.0 };
  IO::TreeBranch<Float_t> EcalCentreOfGravityX            { "EcalCentreOfGravityX",         -999.0 };
  IO::TreeBranch<Float_t> EcalCentreOfGravityY            { "EcalCentreOfGravityY",         -999.0 };
  IO::TreeBranch<Float_t> EcalCentreOfGravityZ            { "EcalCentreOfGravityZ",         -999.0 };
  IO::TreeBranch<Float_t> EcalShowerDirectionZ            { "EcalShowerDirectionZ",         -999.0 };
  IO::TreeBranch<Float_t> ElectronCCMVABDT                { "ElectronCCMVABDT",               -2.0 };
 // IO::TreeBranch<Float_t> EcalBDTv7_EnergyElectron_Smoothed     { "EcalBDTv7_EnergyElectron_Smoothed" ,  0  };
 // IO::TreeBranch<Float_t> EcalBDTv7_EnergyD                     { "EcalBDTv7_EnergyD"                 ,  0  };
 // IO::TreeBranch<Float_t> EcalBDTv7_EnergyD_Smoothed            { "EcalBDTv7_EnergyD_Smoothed"        ,  0  };
 
 // TRD information
  IO::TreeBranch<UChar_t> TrdPActiveLayersTracker               { "TrdPActiveLayersTracker",                  0 };
  IO::TreeBranch<UChar_t> TrdPActiveLayersHybrid                { "TrdPActiveLayersHybrid",                   0 };
  IO::TreeBranch<UChar_t> TrdPActiveLayersStandalone            { "TrdPActiveLayersStandalone",               0 };
  IO::TreeBranch<UChar_t> TrdMaxSubLayersXZ                     { "TrdMaxSubLayersXZ",                        0 };
  IO::TreeBranch<UChar_t> TrdMaxSubLayersYZ                     { "TrdMaxSubLayersYZ",                        0 };
  IO::TreeBranch<UChar_t> TrdFirstSubLayerYZ                    { "TrdFirstSubLayerYZ",                       0 };
  IO::TreeBranch<UChar_t> TrdLastSubLayerYZ                     { "TrdLastSubLayerYZ",                        0 };
  IO::TreeBranch<UChar_t> TrdTrackNumberOfSubLayersXZ           { "TrdTrackNumberOfSubLayersXZ",              0 };
  IO::TreeBranch<UChar_t> TrdTrackNumberOfSubLayersYZ           { "TrdTrackNumberOfSubLayersYZ",              0 };
  IO::TreeBranch<UChar_t> TrdTrackFirstSubLayerYZ               { "TrdTrackFirstSubLayerYZ",                  0 };
  IO::TreeBranch<UChar_t> TrdTrackLastSubLayerYZ                { "TrdTrackLastSubLayerYZ",                   0 };
  IO::TreeBranch<UShort_t> TrdNumberOfHits                      { "TrdNumberOfHits",                          0 };
  IO::TreeBranch<Float_t> TrdPLikelihoodTrackerHitsElectron     { "TrdPLikelihoodTrackerHitsElectron",     -1.0 };
  IO::TreeBranch<Float_t> TrdPLikelihoodTrackerHitsProton       { "TrdPLikelihoodTrackerHitsProton",       -1.0 };
  IO::TreeBranch<Float_t> TrdPLikelihoodTrackerHitsHelium       { "TrdPLikelihoodTrackerHitsHelium",       -1.0 };
  IO::TreeBranch<Float_t> TrdPLikelihoodTrackerHitsElectronECAL { "TrdPLikelihoodTrackerHitsElectronECAL", -1.0 };
  IO::TreeBranch<Float_t> TrdPLikelihoodTrackerHitsProtonECAL   { "TrdPLikelihoodTrackerHitsProtonECAL",   -1.0 };
  IO::TreeBranch<Float_t> TrdPLikelihoodTrackerHitsHeliumECAL   { "TrdPLikelihoodTrackerHitsHeliumECAL",   -1.0 };
  IO::TreeBranch<Float_t> TrdPLikelihoodHybridHitsElectron      { "TrdPLikelihoodHybridHitsElectron",      -1.0 };
  IO::TreeBranch<Float_t> TrdPLikelihoodHybridHitsProton        { "TrdPLikelihoodHybridHitsProton",        -1.0 };
  IO::TreeBranch<Float_t> TrdPLikelihoodHybridHitsHelium        { "TrdPLikelihoodHybridHitsHelium",        -1.0 };
  IO::TreeBranch<Float_t> TrdPLikelihoodHybridHitsElectronECAL  { "TrdPLikelihoodHybridHitsElectronECAL",  -1.0 };
  IO::TreeBranch<Float_t> TrdPLikelihoodHybridHitsProtonECAL    { "TrdPLikelihoodHybridHitsProtonECAL",    -1.0 };
  IO::TreeBranch<Float_t> TrdPLikelihoodHybridHitsHeliumECAL    { "TrdPLikelihoodHybridHitsHeliumECAL",    -1.0 };
  IO::TreeBranch<Float_t> TrdPLikelihoodStandaloneHitsElectron      { "TrdPLikelihoodStandaloneHitsElectron",      -1.0 };
  IO::TreeBranch<Float_t> TrdPLikelihoodStandaloneHitsProton        { "TrdPLikelihoodStandaloneHitsProton",        -1.0 };
  IO::TreeBranch<Float_t> TrdPLikelihoodStandaloneHitsHelium        { "TrdPLikelihoodStandaloneHitsHelium",        -1.0 };
  IO::TreeBranch<Float_t> TrdPLikelihoodStandaloneHitsElectronECAL  { "TrdPLikelihoodStandaloneHitsElectronECAL",  -1.0 };
  IO::TreeBranch<Float_t> TrdPLikelihoodStandaloneHitsProtonECAL    { "TrdPLikelihoodStandaloneHitsProtonECAL",    -1.0 };
  IO::TreeBranch<Float_t> TrdPLikelihoodStandaloneHitsHeliumECAL    { "TrdPLikelihoodStandaloneHitsHeliumECAL",    -1.0 };
  IO::TreeBranch<Float_t> TrdTrackEcalCogDeltaX                 { "TrdTrackEcalCogDeltaX",               -999.0 };
  IO::TreeBranch<Float_t> TrdTrackEcalCogDeltaY                 { "TrdTrackEcalCogDeltaY",               -999.0 };
  IO::TreeBranch<Float_t> TrdTrackEcalCogAngleXZ                { "TrdTrackEcalCogAngleXZ",              -999.0 };
  IO::TreeBranch<Float_t> TrdTrackEcalCogAngleYZ                { "TrdTrackEcalCogAngleYZ",              -999.0 };
 
 //Geomagnetic Cut off
  IO::TreeBranch<Float_t> Stoermer                              { "Stoermer"             ,  0   };
  IO::TreeBranch<Float_t> IGRFMaxCutOff                         { "IGRFMaxCutOff"        ,  0   };
 
  IO::TreeBranch<Double_t> CalculateElectronProtonLikelihood    { "CalculateElectronProtonLikelihood" ,  0  };
  IO::TreeBranch<Double_t> CalculateHeliumElectronLikelihood    { "CalculateHeliumElectronLikelihood" ,  0  };
  IO::TreeBranch<Double_t> CalculateHeliumProtonLikelihood      { "CalculateHeliumProtonLikelihood"        ,  0  };
 // IO::TreeBranch<Float_t> EcaLIntegralLikelihood3D              { "EcaLIntegralLikelihood3D"          ,  0  };
 // IO::TreeBranch<Double_t> MCmomentum                           { "MCmomentum"                        ,  0  };
 // IO::TreeBranch<Double_t> MCparticleid                         { "MCparticleid"                       , 0  };

  // In-memory branches
  IO::InMemoryTreeBranch<Double_t> McEventWeight               { "McEventWeight",                                             0.0 };
  IO::InMemoryTreeBranch<Double_t> McEventWeightWithCutOff     { "McEventWeightWithCutOff",                                   0.0 };
  IO::InMemoryTreeBranch<Double_t> McEventWeightElectronWithCutOff { "McEventWeightElectronWithCutOff",                       0.0 };
  IO::InMemoryTreeBranch<Double_t> McEventWeightPositronWithCutOff { "McEventWeightPositronWithCutOff",                       0.0 };
  IO::InMemoryTreeBranch<Float_t> TrackerRigidity              { "TrackerRigidity",                                           0.0 };
  IO::InMemoryTreeBranch<Float_t> TrackerChiSquareX            { "TrackerChiSquareX",                                         0.0 };
  IO::InMemoryTreeBranch<Float_t> TrackerChiSquareY            { "TrackerChiSquareY",                                         0.0 };
  IO::InMemoryTreeBranch<Float_t> TrackerRelativeSagittaError  { "TrackerRelativeSagittaError",                               0.0 };
  IO::InMemoryTreeBranch<Float_t> TrdPLRElecProt               { "TrdPLRElecProt",                                           -1.0 };
  IO::InMemoryTreeBranch<Float_t> TrdPLRElecProtECAL           { "TrdPLRElecProtECAL",                                       -1.0 };
  IO::InMemoryTreeBranch<Float_t> TrdPLRElecProtStandaloneECAL { "TrdPLRElecProtStandaloneECAL",                             -1.0 };
  IO::InMemoryTreeBranch<Float_t> TrdPLRHeliElec               { "TrdPLRHeliElec",                                           -1.0 };
  IO::InMemoryTreeBranch<Float_t> EcalBDTBest                  { "EcalBDTBest",                                              -2.0 };
  IO::InMemoryTreeBranch<Float_t> EcalBDTBestSmoothed          { "EcalBDTBestSmoothed",                                      -2.0 };
  IO::InMemoryTreeBranch<Float_t> EcalEnergyBest               { "EcalEnergyBest",                                            0.0 };
  IO::InMemoryTreeBranch<Float_t> EcalEnergyBestMaximumShower  { "EcalEnergyBestMaximumShower",                               0.0 };
  IO::InMemoryTreeBranch<Float_t> EnergyOverRigidity           { "EnergyOverRigidity",                                          0 };
  IO::InMemoryTreeBranch<Float_t> EcalShowerMaximumTransformed { "EcalShowerMaximumTransformed",                                0 };
  IO::InMemoryTreeBranch<Char_t>  TrackerPatternSortedByMDR    { "TrackerPatternSortedByMDR",    IO::ValueLimitMode::HighestValue };
  IO::InMemoryTreeBranch<Float_t> TrdEcalChi2                  { "TrdEcalChi2",                                                 0 };
  IO::InMemoryTreeBranch<UChar_t> TrdPActiveLayers             { "TrdPActiveLayers",                                            0 };
  IO::InMemoryTreeBranch<Float_t> TrackerTrackThetaAtEcalUpper { "TrackerTrackThetaAtEcalUpper", IO::ValueLimitMode::HighestValue };
  IO::InMemoryTreeBranch<Float_t> EcalChiSquareLateralNormalized { "EcalChiSquareLateralNormalized",                       -999.0 };

  // Custom accessors
  bool IsMC() const { return McGeneratedMomentum() > 0.0f; }
  bool IsISS() const { return McGeneratedMomentum() == 0.0f && BtNominalMomentum() == 0.0f; }
  bool IsMcPrimaryGeneratorInAcceptance() const;

  void SetReEvaluateElectronCCMVABDT(bool value) { fReEvaluateElectronCCMVABDT = value; }
  bool ReEvaluateElectronCCMVABDT() const { return fReEvaluateElectronCCMVABDT; }
  Mva::MvaImplementation* ElectronChargeConfusionMva() const { return fElectronChargeConfusionMva; }

private:
  virtual void Fill(const Analysis::Event&);
  virtual void UpdateInMemoryBranches();
  virtual IO::TreeInterface* Create() const { return new LeptonAnalysisTree; }
  virtual const IO::TreeBranchBase<UInt_t>* CurrentEventNumber() const { return &Event; }
  virtual const IO::TreeBranchBase<UInt_t>* CurrentRunNumber() const { return &Run; }
  virtual const IO::TreeBranchBase<UInt_t>* CurrentEventTime() const { return &Time; }
  virtual const IO::TreeBranchBase<Double_t>* CurrentWeight() const { return &McEventWeightWithCutOff; }
  virtual const IO::TreeBranchBase<UChar_t>* CurrentTriggerFlags() const { return &TriggerFlags; }

public:
  float CalculateLikelihoodRatio(float likelihood1, float likelihood2) const { return likelihood1 > 0 ? -std::log(likelihood1 / (likelihood1 + likelihood2)) : -1.0f; }

  /* Uncalibrated TrdP likelihood ratios */
  float TrdLRElecProt_Energy_TrdHits_TrdP_Raw() const       { return CalculateLikelihoodRatio(TrdPLikelihoodStandaloneHitsElectronECAL(), TrdPLikelihoodStandaloneHitsProtonECAL()); }
  float TrdLRElecProt_Energy_HybridHits_TrdP_Raw() const    { return CalculateLikelihoodRatio(TrdPLikelihoodHybridHitsElectronECAL(),     TrdPLikelihoodHybridHitsProtonECAL()); }
  float TrdLRElecProt_Rigidity_HybridHits_TrdP_Raw() const  { return CalculateLikelihoodRatio(TrdPLikelihoodHybridHitsElectron(),         TrdPLikelihoodHybridHitsProton()); }
  float TrdLRHeliElec_Rigidity_HybridHits_TrdP_Raw() const  { return CalculateLikelihoodRatio(TrdPLikelihoodHybridHitsHelium(),           TrdPLikelihoodHybridHitsElectron()); }

  /* Calibrated TrdP likelihood ratios. */
  float TrdLRElecProt_Energy_TrdHits_TrdP() const {

    float result = TrdLRElecProt_Energy_TrdHits_TrdP_Raw();
    return IsMC() ? result * TrdLRElecProt_Energy_TrdHits_TrdP_CalibrationFactor(EcalEnergyBestMaximumShower()) : result;
  }

  float TrdLRElecProt_Energy_HybridHits_TrdP() const {

    float result = TrdLRElecProt_Energy_HybridHits_TrdP_Raw();
    return IsMC() ? result * TrdLRElecProt_Energy_HybridHits_TrdP_CalibrationFactor(EcalEnergyBestMaximumShower()) : result;
  }

  float TrdLRElecProt_Rigidity_HybridHits_TrdP() const {

    float result = TrdLRElecProt_Rigidity_HybridHits_TrdP_Raw();
    return IsMC() ? result * TrdLRElecProt_Rigidity_HybridHits_TrdP_CalibrationFactor(EcalEnergyBestMaximumShower()) : result;
  }

  float TrdLRHeliElec_Rigidity_HybridHits_TrdP() const {

    float result = TrdLRHeliElec_Rigidity_HybridHits_TrdP_Raw();
    return IsMC() ? result * TrdLRHeliElec_Rigidity_HybridHits_TrdP_CalibrationFactor(EcalEnergyBestMaximumShower()) : result;
  }

  Float_t ComputeEcalChiSquareLateralNormalized(bool uncorrected = false) const;

private:
  Char_t ComputeTrackerPatternSortedByMDR() const {

    switch (TrackerPattern()) {
    case  0: return 1; // Layer 1 and 9, and maybe 2
    case  1: return 3; // Layer 1 and 2, but not 9
    case  2: return 2; // Layer 2 and 9, but not 1
    case  3: return 5; // Layer 1
    case  4: return 6; // Layer 2
    case  5: return 4; // Layer 9
    case -1: return 0; // None of the above.
    }

    assert(false);
    return 0;
  }

  Double_t ComputeMcEventWeight() const;
  Double_t ComputeMcEventWeightWithCutOff() const;
  Double_t ComputeMcEventWeightElectronWithCutOff() const;
  Double_t ComputeMcEventWeightPositronWithCutOff() const;
  Float_t ComputeEcalShowerMaximumTransformed() const;
  Float_t ComputeTrdEcalChi2() const;
  Float_t ComputeTrackerTrackThetaAtEcalUpper() const;

  Mva::MvaImplementation* fElectronChargeConfusionMva;
  bool fReEvaluateElectronCCMVABDT;

public:
  bool HasPhysicsTrigger() const { return (TriggerFlags() & 0x3e) != 0; }

  friend class LeptonAnalysisTreeDraw;
};

#endif


