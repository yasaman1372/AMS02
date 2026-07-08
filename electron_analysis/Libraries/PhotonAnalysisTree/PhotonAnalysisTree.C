#include "AnalysisEvent.hh"
#include "ExampleAnalysisTree.hh"
#include "TrackerTrack.h"
#include <cmath>
#define INFO_OUT_TAG "ExampleAnalysisTree"
#include "debugging.hh"

ExampleAnalysisTree::ExampleAnalysisTree()
  : IO::TreeInterface("ExampleAnalysisTree", "Example analysis tree") {

  RegisterBranches();
}

void ExampleAnalysisTree::Fill(const Analysis::Event& event) {

  // General event information
  Time = event.TimeStamp().GetSec();
  EventNumber = event.EventNumber();
  Weight = event.Weight();

  // Event header information
  const AC::EventHeader& eventHeader = event.EventHeader();
  UTCTime = eventHeader.UTCTimeStamp();
  ISSYawInRadians = eventHeader.ISSYawInRadians();
  ISSPitchInRadians = eventHeader.ISSPitchInRadians();
  ISSRollInRadians = eventHeader.ISSRollInRadians();
  ISSLatitudeInRadians = eventHeader.ISSLatitudeInRadians();
  ISSLongitudeInRadians = eventHeader.ISSLongitudeInRadians();
  ISSDistanceFromEarthCenterInKm = eventHeader.ISSDistanceFromEarthCenterInKm();
  ISSVelocityR = eventHeader.ISSVelocityR();
  ISSVelocityTheta = eventHeader.ISSVelocityTheta();
  ISSVelocityPhi = eventHeader.ISSVelocityPhi();
  ISSVelocityLatitudeInRadians = eventHeader.ISSVelocityLatitudeInRadians();
  ISSVelocityLongitudeInRadians = eventHeader.ISSVelocityLongitudeInRadians();
  ISSAltitudeInKm = eventHeader.ISSAltitudeInKm();

  // Subdetector information
  NumberOfAccHits = event.NumberOfAccHits();
  NumberOfRichRings = event.NumberOfRichRings();
  NumberOfTracks = event.NumberOfTracks();
  NumberOfTrdHits = event.NumberOfTrdHits();
  NumberOfTofHits = event.NumberOfTofHits();
  NumberOfEcalShower = event.NumberOfEcalShower();
  TrackerNumberOfTracks = event.NumberOfTrackerTracks();

  // ECAL information
  const AC::ECAL& ecal = event.RawEvent()->ECAL();
  EcalBDT3D = ecal.BDTEstimator3D();
  TotalEnergy3D = ecal.TotalEnergy3D();
  EcaLIntegralLikelihood3D = ecal.IntegralLikelihoodEstimator3D();

  // TRD segments
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
     IGRFMaxCutOff = event.IGRFMaxCutOff(RTI::CutOffMode::CutOff25PN);
     Stoermer = event.GeomagneticMaxCutOff(RTI::CutOffMode::CutOff25PN); 
  }
 
  if (const Analysis::Particle* primaryParticle = event.PrimaryParticle()) {
    Rigidity = primaryParticle->Rigidity();
    UpperTofCharge = primaryParticle->UpperTofCharge();
    LowerTofCharge = primaryParticle->LowerTofCharge();
    InnerTrackerCharge = primaryParticle->TrackerCharge();
    NumberOfTrdActiveLayers = primaryParticle->NumberOfTrdActiveLayers();  
    EcalEnergy = primaryParticle->EcalEnergy();    
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
      EcalShowerDirectionZ = shower->ShowerAxis().Z();
    }  

    if (auto spline = primaryParticle->GetSplineTrack()) {
      auto trackPositionAtEcal = spline->PositionAtZ(EcalShowerPositionZ());
      TrackerTrackPositionAtEcalX = trackPositionAtEcal.X();
      TrackerTrackPositionAtEcalY = trackPositionAtEcal.Y();
    }

    if (auto TrackerTrack = primaryParticle->TrackerTrack()) {
      InnerTrackerCharge = TrackerTrack->ChargeYiJia().Charge();
    }

    // TOF
    TofNumberOfLayers = primaryParticle->TofNumberOfLayers();
    if (primaryParticle && primaryParticle->HasTofBeta() && primaryParticle->HasTrdTrack()) {
      Analysis::TrdTrackTofMatching& match = Analysis::TrdTrackTofMatching::SharedInstance();
      match.ApplyMatching(event, primaryParticle->GetTrdTrack());
      TofUpperCharge = match.UpperCharge();
      TofLowerCharge = match.LowerCharge();
      TofTrdMatchNorm = match.MatchingNorm();
    }
 
    // TRD
    if (primaryParticle->HasTrdTrack()) {
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
  if (event.IsMC()) {
    MCparticleid = event.McParticleId();
    MCmomentum = event.McMomentum();
  }
}

void ExampleAnalysisTree::UpdateInMemoryBranches() {
  // Implement any in-memory branch updates if necessary
}
