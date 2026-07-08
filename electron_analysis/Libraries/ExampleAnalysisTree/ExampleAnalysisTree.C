#:include "AnalysisEvent.hh"
#include "ExampleAnalysisTree.hh"
#include "TrackerTrack.h"
#include <cmath>
#include "TrdTrackEcalMatching.hh"
#include "TrdTrackTofMatching.hh"
#include <ECAL.h>
#include <Event.h>
#define INFO_OUT_TAG "ExampleAnalysisTree"
#include "debugging.hh"

ExampleAnalysisTree::ExampleAnalysisTree()
  : IO::TreeInterface("ExampleAnalysisTree", "Example analysis tree") {

  RegisterBranches();
}

void ExampleAnalysisTree::Fill(const Analysis::Event& event) {

  Time = event.TimeStamp().GetSec();
  EventNumber = event.EventNumber();
  Weight = event.Weight();
  TrackerNumberOfTracks = event.NumberOfTrackerTracks();
  NumberOfEcalShower=event.NumberOfEcalShower();
  NumberOfTofHits=event.NumberOfTofHits();
  NumberOfTrackerTracks=event.NumberOfTrackerTracks();
  // new ECAL 3D BDT Estimator
  const AC::ECAL& ecal = event.RawEvent()->ECAL();
  EcalBDT3D = ecal.BDTEstimator3D();
  TotalEnergy3D = ecal.TotalEnergy3D();

  // other options for the 3D estimator (slightly different spectrum, same information):

  EcaLIntegralLikelihood3D = ecal.IntegralLikelihoodEstimator3D();
  if(event.IsMC()) {
  MCparticleid = event.McParticleId();
  MCmomentum = event.McMomentum();
 } 

//TRD segments

  int trdMaxSubLayersXZ = 0;
  for (unsigned int i = 0; i < event.TrdSegmentsXZ().size(); ++i)
    trdMaxSubLayersXZ = std::max(trdMaxSubLayersXZ, event.TrdSegmentsXZ()[i].NumberOfSublayersInSegment());
  TrdMaxSubLayersXZ = (trdMaxSubLayersXZ);

  int trdMaxSubLayersYZ = 0;
  int trdMaxFirstSubLayerYZ = 0;
  int trdMinLastSubLayerYZ = 20;
  for (unsigned int i = 0; i < event.TrdSegmentsYZ().size(); ++i) {
    trdMaxSubLayersYZ = std::max(trdMaxSubLayersYZ, event.TrdSegmentsYZ()[i].NumberOfSublayersInSegment());
    trdMaxFirstSubLayerYZ = std::max(trdMaxFirstSubLayerYZ, event.TrdSegmentsYZ()[i].FirstSublayerInSegment());
    trdMinLastSubLayerYZ = std::min(trdMinLastSubLayerYZ, event.TrdSegmentsYZ()[i].LastSublayerInSegment());
  }

  TrdMaxSubLayersYZ = (trdMaxSubLayersYZ);
  TrdMaxFirstSubLayerYZ = (trdMaxFirstSubLayerYZ);
  TrdMinLastSubLayerYZ = (trdMinLastSubLayerYZ);
 
  if (event.IsISS()) {
     IGRFMaxCutOff=event.IGRFMaxCutOff(RTI::CutOffMode::CutOff25PN);
     Stoermer= event.GeomagneticMaxCutOff(RTI::CutOffMode::CutOff25PN); 
 }
 
  if (const Analysis::Particle* primaryParticle = event.PrimaryParticle()) {
    EcalEnergyElectron = primaryParticle->EcalEnergyElectron();
    Rigidity = primaryParticle->Rigidity();
    UpperTofCharge = primaryParticle->UpperTofCharge();
    LowerTofCharge = primaryParticle->LowerTofCharge();
   // InnerTrackerCharge = primaryParticle->TrackerCharge();
    TrackerChiSquareY = primaryParticle->Chi2TrackerY();
    BetaTof = primaryParticle->BetaTof();
    NumberOfTrdActiveLayers = primaryParticle->NumberOfTrdActiveLayers();  
    EcalEnergy = primaryParticle->EcalEnergy();    
    Chi2TrackerX  = primaryParticle->Chi2TrackerX();
    Chi2TrackerY  = primaryParticle->Chi2TrackerY();
    CalculateElectronProtonLikelihood = primaryParticle->CalculateElectronProtonLikelihood();
    CalculateHeliumElectronLikelihood = primaryParticle->CalculateHeliumElectronLikelihood();
    CalculateHeliumProtonLikelihood   = primaryParticle->CalculateHeliumProtonLikelihood();
    TrackerLayerPatternClassification = primaryParticle->TrackerLayerPatternClassification();  
  
    // ECAL BDT Estimators
    EcalBDT = primaryParticle->EcalEstimator(AC::ECALShower::BDTv7_EnergyElectron);
    EcalBDTv7_EnergyElectron_Smoothed  = primaryParticle->EcalEstimator(AC::ECALShower::BDTv7_EnergyElectron_Smoothed);
    EcalBDTv7_EnergyD = primaryParticle->EcalEstimator(AC::ECALShower::BDTv7_EnergyD);
    EcalBDTv7_EnergyD_Smoothed = primaryParticle->EcalEstimator(AC::ECALShower::BDTv7_EnergyD_Smoothed);

    
     
  if (const auto& shower = primaryParticle->EcalShower()) {
    EcalShowerPositionX = shower->X();
    EcalShowerPositionY = shower->Y();
    EcalShowerPositionZ = shower->Z();
    ChiSquareEcalAxisLateralMethod = shower->ChiSquareEcalAxisLateralMethod();
   }  

   if (auto spline = primaryParticle->GetSplineTrack()) {
    auto trackPositionAtEcal = spline->PositionAtZ(EcalShowerPositionZ());
    TrackerTrackPositionAtEcalX = trackPositionAtEcal.X();
    TrackerTrackPositionAtEcalY = trackPositionAtEcal.Y();
   }

    if (auto TrackerTrack= primaryParticle -> TrackerTrack()) {
    InnerTrackerCharge = TrackerTrack-> ChargeYiJia().Charge();
    }


    // TOF
    TofNumberOfLayers = event.TofNumberOfLayers();
    if (primaryParticle && primaryParticle->HasTofBeta() && primaryParticle->HasTrdTrack()) {
    Analysis::TrdTrackTofMatching& match = Analysis::TrdTrackTofMatching::SharedInstance();
    match.ApplyMatching(event, primaryParticle->GetTrdTrack());
    TofLowerCharge = match.LowerCharge();
    TofUpperCharge = match.UpperCharge();
    TofBeta = match.Beta();
    TofDeltaT = match.TimeDifference();
    TofTrdMatchNorm = match.MatchingNorm();
  }
   
  //Ecal
  
     if (primaryParticle->HasEcalShower() && primaryParticle->HasTrdTrack()) {
     Analysis::TrdTrackEcalMatching& match = Analysis::TrdTrackEcalMatching::SharedInstance();
   //  double rigidity = TrackerTrackChoutkoMaxSpanRigidity();
     double chargeTimesMomentum = 0.0;
   //  if (rigidity <= 0)
   //    chargeTimesMomentum = -std::abs(EcalEnergyElectronNewMaximumShower());
   //  else
   //    chargeTimesMomentum = std::abs(EcalEnergyElectronNewMaximumShower());

     match.ApplyMatching(event, *primaryParticle->EcalShower(), *primaryParticle->GetTrdTrack(), chargeTimesMomentum);
     EcalShowerDirectionZ = match.ShowerDirection().Z();
     TrdTrackEcalCogDeltaX = match.DeltaX();
     TrdTrackEcalCogDeltaY = match.DeltaY();
     TrdTrackEcalCogAngleXZ = match.AngleXZ();
     TrdTrackEcalCogAngleYZ = match.AngleYZ();
  }

 //TRD

   if (primaryParticle->HasTrdTrack()){
   auto segmentXZ = primaryParticle->GetTrdTrack()->SegmentXZ();
   auto segmentYZ = primaryParticle->GetTrdTrack()->SegmentXZ();
   TrdSubLayersXZ = segmentXZ->NumberOfSublayersInSegment();
   TrdSubLayersYZ = segmentYZ->NumberOfSublayersInSegment();
   TrdFirstSubLayerYZ = segmentYZ->FirstSublayerInSegment();
   TrdLastSubLayerYZ = segmentYZ->LastSublayerInSegment();
}
   
 // Example on how to fill a vector.
    assert(TrackerCharges().empty());
    TrackerCharges().emplace_back(primaryParticle->TrackerChargeFor(Analysis::Particle::TrkLay1));
    TrackerCharges().emplace_back(primaryParticle->TrackerChargeFor(Analysis::Particle::TrkLay2));
    TrackerCharges().emplace_back(primaryParticle->TrackerChargeFor(Analysis::Particle::TrkLay3));
    TrackerCharges().emplace_back(primaryParticle->TrackerChargeFor(Analysis::Particle::TrkLay4));
    TrackerCharges().emplace_back(primaryParticle->TrackerChargeFor(Analysis::Particle::TrkLay5));
    TrackerCharges().emplace_back(primaryParticle->TrackerChargeFor(Analysis::Particle::TrkLay6));
    TrackerCharges().emplace_back(primaryParticle->TrackerChargeFor(Analysis::Particle::TrkLay7));
    TrackerCharges().emplace_back(primaryParticle->TrackerChargeFor(Analysis::Particle::TrkLay8));
    TrackerCharges().emplace_back(primaryParticle->TrackerChargeFor(Analysis::Particle::TrkLay9));
  }
}

void ExampleAnalysisTree::UpdateInMemoryBranches() {

  if (Rigidity() != 0)
    EoverAbsR = EcalEnergyElectron() / std::abs(Rigidity());
}
