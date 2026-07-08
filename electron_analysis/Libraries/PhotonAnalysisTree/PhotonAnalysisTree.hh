#ifndef ExampleAnalysisTree_hh
#define ExampleAnalysisTree_hh

#include "TreeInterface.hh"
#include "AnalysisEvent.hh"
#include <vector>

class ExampleAnalysisTree : public IO::TreeInterface {

public:
  ExampleAnalysisTree();

  void Fill(const Analysis::Event& event);

private:
  void RegisterBranches();
  void UpdateInMemoryBranches();

  // General
  IO::TreeBranch<UInt_t>   EventNumber                                      { "EventNumber",                                      IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<UInt_t>   RunNumber                                        { "RunNumber",                                        IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<UInt_t>   EventTime                                        { "EventTime",                                        IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  CPUTime                                          { "CPUTime",                                          IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  RealTime                                         { "RealTime",                                         IO::ValueLimitMode::HighestValue };

  // Event header information
  IO::TreeBranch<Double_t> UTCTime                                          { "UTCTime",                                          IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  ISSYawInRadians                                  { "ISSYawInRadians",                                  IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  ISSPitchInRadians                                { "ISSPitchInRadians",                                IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  ISSRollInRadians                                 { "ISSRollInRadians",                                 IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  ISSLatitudeInRadians                             { "ISSLatitudeInRadians",                             IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  ISSLongitudeInRadians                            { "ISSLongitudeInRadians",                            IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  ISSDistanceFromEarthCenterInKm                   { "ISSDistanceFromEarthCenterInKm",                   IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  ISSVelocityInKmPerSecond                         { "ISSVelocityInKmPerSecond",                         IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  ISSVelocityLatitudeInRadians                     { "ISSVelocityLatitudeInRadians",                     IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  ISSVelocityLongitudeInRadians                    { "ISSVelocityLongitudeInRadians",                    IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  ISSAltitudeInKm                                  { "ISSAltitudeInKm",                                  IO::ValueLimitMode::HighestValue };

  // Global quantities
  IO::TreeBranch<UChar_t>  NumberOfAccHits                                  { "NumberOfAccHits",                                  IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<UChar_t>  NumberOfRichRings                                { "NumberOfRichRings",                                IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<UChar_t>  NumberOfTracks                                   { "NumberOfTracks",                                   IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<UChar_t>  NumberOfTrdHits                                  { "NumberOfTrdHits",                                  IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<UChar_t>  NumberOfTofHits                                  { "NumberOfTofHits",                                  IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<UChar_t>  NumberOfEcalShower                               { "NumberOfEcalShower",                               IO::ValueLimitMode::HighestValue };

  // Trigger
  IO::TreeBranch<UChar_t>  TriggerFlags                                     { "TriggerFlags",                                     IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<UShort_t> TriggerSubFlags                                  { "TriggerSubFlags",                                  IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  TriggerLiveTime                                  { "TriggerLiveTime",                                  IO::ValueLimitMode::HighestValue };

  // Tracker
  IO::TreeBranch<Float_t>  Rigidity                                         { "Rigidity",                                         IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  UpperTofCharge                                   { "UpperTofCharge",                                   IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  LowerTofCharge                                   { "LowerTofCharge",                                   IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  InnerTrackerCharge                               { "InnerTrackerCharge",                               IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<std::vector<Float_t>> TrackerCharges                       { "TrackerCharges",                                   IO::TreeVectorSize(9) };
  IO::TreeBranch<Float_t>  TrackerTrackPositionAtEcalX                      { "TrackerTrackPositionAtEcalX",                      IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  TrackerTrackPositionAtEcalY                      { "TrackerTrackPositionAtEcalY",                      IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<UChar_t>  TrackerNumberOfTracks                            { "TrackerNumberOfTracks",                            IO::ValueLimitMode::HighestValue };

  // ECAL
  IO::TreeBranch<Float_t>  EcalBDT3D                                        { "EcalBDT3D",                                        IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  TotalEnergy3D                                    { "TotalEnergy3D",                                    IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  EcaLIntegralLikelihood3D                         { "EcaLIntegralLikelihood3D",                         IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  EcalEnergy                                       { "EcalEnergy",                                       IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  EcalShowerPositionX                              { "EcalShowerPositionX",                              IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  EcalShowerPositionY                              { "EcalShowerPositionY",                              IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  EcalShowerPositionZ                              { "EcalShowerPositionZ",                              IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  ChiSquareEcalAxisLateralMethod                   { "ChiSquareEcalAxisLateralMethod",                   IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  EcalShowerDirectionZ                             { "EcalShowerDirectionZ",                             IO::ValueLimitMode::HighestValue };

  // TRD
  IO::TreeBranch<Int_t>    TrdMaxSubLayersXZ                                { "TrdMaxSubLayersXZ",                                IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Int_t>    TrdMaxSubLayersYZ                                { "TrdMaxSubLayersYZ",                                IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Int_t>    TrdMaxFirstSubLayerYZ                            { "TrdMaxFirstSubLayerYZ",                            IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Int_t>    TrdMinLastSubLayerYZ                             { "TrdMinLastSubLayerYZ",                             IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Int_t>    TrdSubLayersXZ                                   { "TrdSubLayersXZ",                                   IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Int_t>    TrdSubLayersYZ                                   { "TrdSubLayersYZ",                                   IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Int_t>    TrdFirstSubLayerYZ                               { "TrdFirstSubLayerYZ",                               IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Int_t>    TrdLastSubLayerYZ                                { "TrdLastSubLayerYZ",                                IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<UChar_t>  NumberOfTrdActiveLayers                          { "NumberOfTrdActiveLayers",                          IO::ValueLimitMode::HighestValue };

  // TOF
  IO::TreeBranch<UChar_t>  TofNumberOfLayers                                { "TofNumberOfLayers",                                IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  TofUpperCharge                                   { "TofUpperCharge",                                   IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  TofLowerCharge                                   { "TofLowerCharge",                                   IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  TofTrdMatchNorm                                  { "TofTrdMatchNorm",                                  IO::ValueLimitMode::HighestValue };

  // Monte Carlo
  IO::TreeBranch<Int_t>    MCparticleid                                     { "MCparticleid",                                     IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t>  MCmomentum                                       { "MCmomentum",                                       IO::ValueLimitMode::HighestValue };
};

#endif
