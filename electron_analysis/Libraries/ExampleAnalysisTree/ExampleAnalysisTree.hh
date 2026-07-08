::#ifndef ExampleAnalysisTree_hh
#define ExampleAnalysisTree_hh

#include "TreeInterface.hh"

class ExampleAnalysisTree : public IO::TreeInterface {
public:
  ExampleAnalysisTree();
  IO::TreeBranch<UInt_t> Time                         { "Time"                 ,  0   };
  IO::TreeBranch<Int_t> EventNumber                   { "EventNumber"          , -1   };
  IO::TreeBranch<Double_t> Weight                     { "Weight"               , -1   };
  IO::TreeBranch<Float_t> EcalEnergyElectron          { "EcalEnergyElectron"   ,  0.0 };
  IO::TreeBranch<Float_t> UpperTofCharge              { "UpperTofCharge"       ,  IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<Float_t> LowerTofCharge              { "LowerTofCharge"       ,  IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<UChar_t> TrackerNumberOfTracks       { "TrackerNumberOfTracks",  0   };
  IO::TreeBranch<Float_t> TrackerChiSquareY           { "TrackerChiSquareY"    ,  0   };
  IO::TreeBranch<Float_t> InnerTrackerCharge          { "InnerTrackerCharge"   ,  IO::ValueLimitMode::HighestValue };
  IO::TreeBranch<std::vector<Float_t>> TrackerCharges { "TrackerCharges"       ,  IO::TreeVectorSize(9) };
  IO::TreeBranch<Float_t> Rigidity                    { "Rigidity"             ,  0   };
  IO::TreeBranch<Float_t> NumberOfEcalShower          { "NumberofECALShowers"  , -1   };
  IO::TreeBranch<Float_t> NumberOfTofHits             { "NumberOfTofHits"      , -1   };
  IO::TreeBranch<Float_t> NumberOfTrackerTracks       { "NumberOfTrackerTracks", -1   };
  IO::TreeBranch<Float_t> BetaTof                     { "BetaTof"              ,  0   }; 
  IO::TreeBranch<Double_t> NumberOfTrdActiveLayers    { "NumberOfTrdActiveLayers",-1  };
  IO::TreeBranch<Float_t> EcalEnergy                  { "EcalEnergy"           ,  0   };
  IO::TreeBranch<Float_t> Chi2TrackerX                { "Chi2TrackerX"         ,  0   };
  IO::TreeBranch<Float_t> Chi2TrackerY                { "Chi2TrackerY"         ,  0   };
  IO::TreeBranch<Float_t> Stoermer                    { "Stoermer"             ,  0   };
  IO::TreeBranch<Float_t> IGRFMaxCutOff               { "IGRFMaxCutOff"        ,  0   };
  IO::TreeBranch<Float_t> EcalShowerPositionX         { "EcalShowerPositionX"  ,  0   };
  IO::TreeBranch<Float_t> EcalShowerPositionY         { "EcalShowerPositionY"  ,  0   }; 
  IO::TreeBranch<Float_t> EcalShowerPositionZ         { "EcalShowerPositionZ"  ,  0   };
  IO::TreeBranch<Float_t> EcalBDT3D                   { "EcalBDT3D"            ,  0   };
  IO::TreeBranch<Float_t> EcalBDT                     { "EcalBDT"              ,  0   };
  IO::TreeBranch<Double_t> TofUpperCharge             { "TofUpperCharge"       ,  0   };
  IO::TreeBranch<Double_t> TofLowerCharge             { "TofLoweCharge"        ,  0   };
  IO::TreeBranch<UInt_t> TofNumberOfLayers            { "TofNumberOfLayers"    ,  0   };
  IO::TreeBranch<Double_t> TofBeta                    { "TofBeta"              ,  0   };
  IO::TreeBranch<Int_t> TofDeltaT                     { "TofDeltaT"            ,  0   };
  IO::TreeBranch<Double_t> TofTrdMatchNorm            { "TofTrdMatchNorm"      ,  0   };
  IO::TreeBranch<Double_t> TrdTrackEcalCogDeltaX      { "TrdTrackEcalCogDeltaX",  0   };
  IO::TreeBranch<Double_t> TrdTrackEcalCogDeltaY      { "TrdTrackEcalCogDeltaY",  0   };
  IO::TreeBranch<Double_t> TrdTrackEcalCogAngleXZ     { "TrdTrackEcalCogAngleXZ", 0   };
  IO::TreeBranch<Double_t> TrdTrackEcalCogAngleYZ     { "TrdTrackEcalCogAngleYZ", 0   };
  IO::TreeBranch<Double_t> EcalShowerDirectionZ       { "EcalShowerDirectionZ"  , 0   };
  IO::TreeBranch<Float_t> TotalEnergy3D               { "TotalEnergy3D"         , 0   };
  IO::TreeBranch<Int_t>  TrdMaxSubLayersXZ            { "TrdMaxSubLayersXZ"     , 0   }; 
  IO::TreeBranch<Int_t>  TrdMaxSubLayersYZ            { "TrdMaxSubLayersYZ"     , 0   };
  IO::TreeBranch<Int_t>  TrdMaxFirstSubLayerYZ        { "TrdMaxFirstSubLayerYZ" , 0   };
  IO::TreeBranch<Int_t>  TrdMinLastSubLayerYZ         { "TrdMinLastSubLayerYZ"  , 0   };
  IO::TreeBranch<Int_t>  TrdSubLayersXZ               { "TrdSubLayersXZ"        , 0   };
  IO::TreeBranch<Int_t>  TrdSubLayersYZ               { "TrdSubLayersYZ"        , 0   };
  IO::TreeBranch<Int_t>  TrdFirstSubLayerYZ           { "TrdFirstSubLayerYZ"    , 0   };
  IO::TreeBranch<Int_t>  TrdLastSubLayerYZ            { "TrdLastSubLayerYZ"     , 0   };
  IO::TreeBranch<Float_t> TrackerTrackPositionAtEcalX { "TrackerTrackPositionAtEcalX", 0 };
  IO::TreeBranch<Float_t> TrackerTrackPositionAtEcalY { "TrackerTrackPositionAtEcalY", 0 };
  IO::TreeBranch<Double_t> CalculateElectronProtonLikelihood    { "CalculateElectronProtonLikelihood" ,  0  };
  IO::TreeBranch<Double_t> CalculateHeliumElectronLikelihood    { "CalculateHeliumElectronLikelihood" ,  0  };
  IO::TreeBranch<Double_t> CalculateHeliumProtonLikelihood	{ "CalculateHeliumProtonLikelihood"        ,  0  };
  IO::TreeBranch<Float_t> EcaLIntegralLikelihood3D              { "EcaLIntegralLikelihood3D"          ,  0  };
  IO::TreeBranch<Float_t> EcalBDTv7_EnergyElectron_Smoothed     { "EcalBDTv7_EnergyElectron_Smoothed" ,  0  };
  IO::TreeBranch<Float_t> EcalBDTv7_EnergyD                     { "EcalBDTv7_EnergyD"                 ,  0  };
  IO::TreeBranch<Float_t> EcalBDTv7_EnergyD_Smoothed            { "EcalBDTv7_EnergyD_Smoothed"        ,  0  };
  IO::TreeBranch<Short_t> TrackerLayerPatternClassification     { "TrackerLayerPatternClassification" ,  0  };
  IO::TreeBranch<Float_t> ChiSquareEcalAxisLateralMethod        { "ChiSquareEcalAxisLateralMethod"    ,  0  };
  IO::TreeBranch<Double_t> MCmomentum                           { "MCmomentum"                        ,  0  };
  IO::TreeBranch<Double_t> MCparticleid                         { "MCparticleid"                       , 0  };
  IO::InMemoryTreeBranch<Float_t> EoverAbsR           { "EoverAbsR"            , -1   };


private:
  virtual void Fill(const Analysis::Event&);
  virtual void UpdateInMemoryBranches();
  virtual IO::TreeInterface* Create() const { return new ExampleAnalysisTree; }
  virtual const IO::TreeBranch<UInt_t>* CurrentEventTime() const { return &Time; }
  virtual const IO::TreeBranch<Double_t>* CurrentWeight() const { return &Weight; }
  virtual const IO::TreeBranch<UChar_t>* CurrentTriggerFlags() const { return nullptr; }
};

#endif
