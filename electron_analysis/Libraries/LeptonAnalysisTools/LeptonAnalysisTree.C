#include "LeptonAnalysisTree.hh"

#include "AMSGeometry.h"
#include "AcceptanceManager.hh"
#include "AnalysisEvent.hh"
#include "BeamtestSettings.hh"
#include "CutFactory.hh"
#include "DetectorManager.hh"
#include "EcalChiSquareLateralCalibration.hh"
#include "ElectronChargeConfusionMvaTree.hh"
#include "Event.h"
#include "EventFactory.hh"
#include "Clamping.hh"
#include "CoordinateSystems.hh"
#include "ECALShower.h"
#include "EcalLongitudinalShowerFit.hh"
#include "FileManagerController.hh"
#include "McSpectrumScaler.hh"
#include "MeasuringTimeParameterization.hh"
#include "MvaImplementation.hh"
#include "MvaInterface.hh"
#include "FileManager.hh"
#include "KinematicVariable.hh"
#include "SlowControlLookup.hh"
#include "TrackerTrack.h"
#include "TrdLikelihoodCalculation.hh"
#include "TrdParametrizedPDFs.hh"
#include "TrdSegment.hh"
#include "TrdTrackEcalMatching.hh"
#include "TrdTrackTofMatching.hh"

#include <TFile.h>
#include <TGraphAsymmErrors.h>

#define INFO_OUT_TAG "LeptonAnalysisTree"
#include "debugging.hh"

LeptonAnalysisTree::LeptonAnalysisTree()
  : IO::TreeInterface("LeptonAnalysisTree", "Lepton analysis tree")
  , fElectronChargeConfusionMva(nullptr)
  , fReEvaluateElectronCCMVABDT(false) {

  RegisterBranches();

  // Load electron charge-confusion MVA
  const auto* mvaInterfaceElectronCC = Mva::MvaInterface::CreateMvaInterfaceByName("Mva::ElectronChargeConfusionMvaInterface");
  fElectronChargeConfusionMva = mvaInterfaceElectronCC->CreateMvaImplementation();
}

void LeptonAnalysisTree::Fill(const Analysis::Event& event) {

  // new ECAL 3D BDT Estimator
 // const AC::ECAL& ecal = event.RawEvent()->ECAL();
 // EcalBDT3D = ecal.BDTEstimator3D();
 // TotalEnergy3D = ecal.TotalEnergy3D();

  const Analysis::Particle* particle = event.PrimaryParticle();
 // CalculateElectronProtonLikelihood = primaryParticle->CalculateElectronProtonLikelihood();
 // CalculateHeliumElectronLikelihood = primaryParticle->CalculateHeliumElectronLikelihood();
 // CalculateHeliumProtonLikelihood   = primaryParticle->CalculateHeliumProtonLikelihood();
 
 // General
  Run = clampTo<UInt_t>(event.Run());
 // Weight = event.Weight();
  Event = clampTo<UInt_t>(event.EventNumber());
  Time = clampTo<UInt_t>(event.TimeStamp().GetSec());
  AMSVersion = clampTo<UShort_t>(event.RawEvent()->RunHeader()->AMSVersion());
  TriggerFlags = clampTo<UChar_t>(event.TriggerFlags());
  
  //Geomagnetic cut off
  if (event.IsISS()) {
    IGRFMaxCutOff=event.IGRFMaxCutOff(RTI::CutOffMode::CutOff25PN);
    Stoermer= event.GeomagneticMaxCutOff(RTI::CutOffMode::CutOff25PN);
 }

  // Beam-test
  if (event.IsBeamTest()) {
    const std::vector<Utilities::BeamDescription>& beamDescriptions = Utilities::BeamtestSettings::Self().BeamDescriptionsForEvent(event);
    if (beamDescriptions.empty())
      FATAL_OUT_WITH_EVENT << "No beam test description available for this run. Aborting!" << std::endl;

    const Utilities::BeamDescription& beamDescription = beamDescriptions.front();
    BtParticleId = safeClampTo<UChar_t>(ParticleId::Id(Utilities::BeamtestSettings::Self().CherenkovParticleId(event)), 255);
    BtNominalMomentum = beamDescription.momentum;
  }

  // MC
  McParticleId = safeClampTo<UChar_t>(event.McParticleId(), 255);
  McGeneratedMomentum = event.McMomentum();
  if (event.IsMC()) {
    const AC::MCEventGenerator* primaryGenerator = event.RawEvent()->MC().PrimaryEventGenerator();
    assert(primaryGenerator);
    McGeneratedDirectionTheta = primaryGenerator->Theta();
    McGeneratedDirectionPhi = primaryGenerator->Phi();
    McGeneratedPositionX = primaryGenerator->X0();
    McGeneratedPositionY = primaryGenerator->Y0();
    McGeneratedPositionZ = primaryGenerator->Z0();

    auto* eventFactory = Analysis::EventFactory::Self();
    assert(eventFactory);
    assert(eventFactory->McSpectrumScaler());

    ParticleId::Species mcSpecies = ParticleId::ToSpecies(McParticleId());
    assert(mcSpecies == ParticleId::Electron || mcSpecies == ParticleId::Positron || mcSpecies == ParticleId::Proton);

    McEventWeightElectron = eventFactory->McSpectrumScaler()->GetWeight(mcSpecies, ParticleId::Electron, event.McMomentum(), Utilities::KinematicVariable::Energy);
    McEventWeightPositron = eventFactory->McSpectrumScaler()->GetWeight(mcSpecies, ParticleId::Positron, event.McMomentum(), Utilities::KinematicVariable::Energy);
  }

  // TOF
  TofNumberOfLayers = clampTo<UChar_t>(event.TofNumberOfLayers());
  if (particle && particle->HasTofBeta()) {
    Analysis::TrdTrackTofMatching& match = Analysis::TrdTrackTofMatching::SharedInstance();
    match.ApplyMatching(event, particle->GetTrdTrack());
    TofLowerCharge = match.LowerCharge();
    TofUpperCharge = match.UpperCharge();
    TofBeta = match.Beta();
    TofDeltaT = match.TimeDifference();
    TofTrdMatchNorm = match.MatchingNorm();
  }

  // Tracker and Coordinates
  TrackerNumberOfTracks = clampTo<UChar_t>(event.NumberOfTrackerTracks());

  const int refitPattern = AC::PGMA + AC::RebuildFromTDV;
  const int refitPatternifneeded = AC::AlignV6 + AC::RefitIfNeeded;
  const int refitPatternV6 = AC::AlignV6 + AC::RebuildFromTDV;
  const AC::Tracker& tracker = event.RawEvent()->Tracker();
 // CCBDT = tracker.CCBDT();

  float xDeltaTRDCenterMin = 9999.9f;
  float yDeltaTRDCenterMin = 9999.9f;

  float xDeltaLayer4Min = 9999.9f;
  float yDeltaLayer4Min = 9999.9f;

  float xDeltaLayer9Min = 9999.9f;
  float yDeltaLayer9Min = 9999.9f;

  const AC::TrackerTrack* associatedTrackerTrack = nullptr;
  if (particle && particle->HasTrackerTrack()) {
    associatedTrackerTrack = particle->TrackerTrack();
    assert(associatedTrackerTrack);
  }

  double sumOfUnusedRigidities = 0.0;
  for (const auto& trackerTrack : tracker.Tracks()) {
    if (&trackerTrack == associatedTrackerTrack || !associatedTrackerTrack)
      continue;

    int choutkoMaxSpanIndex = trackerTrack.GetFitFuzzy(AC::Choutko, AC::All, refitPattern, AC::DefaultMass);
    int alcarazMaxSpanIndex = trackerTrack.GetFitFuzzy(AC::Alcaraz, AC::All, refitPattern, AC::DefaultMass);
    int useTrackFitIndex = choutkoMaxSpanIndex >= 0 ? choutkoMaxSpanIndex : alcarazMaxSpanIndex;
    if (useTrackFitIndex >= 0) {
      auto& trackFit = trackerTrack.TrackFits().at(useTrackFitIndex);
      sumOfUnusedRigidities += std::abs(trackFit.Rigidity());
    }

    if (!associatedTrackerTrack->HasTrackFitCoordinates() || !trackerTrack.HasTrackFitCoordinates())
      continue;

    const AC::TrackerTrackCoordinates& referenceCoordinates = associatedTrackerTrack->TrackFitCoordinates();
    const AC::TrackerTrackCoordinates& coordinates = trackerTrack.TrackFitCoordinates();

    float xDeltaTRDCenter = std::abs(referenceCoordinates.XTRDCenter() - coordinates.XTRDCenter());
    if (xDeltaTRDCenter < xDeltaTRDCenterMin)
      xDeltaTRDCenterMin = xDeltaTRDCenter;

    float yDeltaTRDCenter = std::abs(referenceCoordinates.YTRDCenter() - coordinates.YTRDCenter());
    if (yDeltaTRDCenter < yDeltaTRDCenterMin)
      yDeltaTRDCenterMin = yDeltaTRDCenter;

    float xDeltaLayer4 = std::abs(referenceCoordinates.XLayer4() - coordinates.XLayer4());
    if (xDeltaLayer4 < xDeltaLayer4Min)
      xDeltaLayer4Min = xDeltaLayer4;

    float yDeltaLayer4 = std::abs(referenceCoordinates.YLayer4() - coordinates.YLayer4());
    if (yDeltaLayer4 < yDeltaLayer4Min)
      yDeltaLayer4Min = yDeltaLayer4;

    float xDeltaLayer9 = std::abs(referenceCoordinates.XLayer9() - coordinates.XLayer9());
    if (xDeltaLayer9 < xDeltaLayer9Min)
      xDeltaLayer9Min = xDeltaLayer9;

    float yDeltaLayer9 = std::abs(referenceCoordinates.YLayer9() - coordinates.YLayer9());
    if (yDeltaLayer9 < yDeltaLayer9Min)
      yDeltaLayer9Min = yDeltaLayer9;
  }

  TrackerMinDeltaXAtTRDCenterAllTracksToReference = xDeltaTRDCenterMin;
  TrackerMinDeltaYAtTRDCenterAllTracksToReference = yDeltaTRDCenterMin;
  TrackerMinDeltaXAtLayer4AllTracksToReference = xDeltaLayer4Min;
  TrackerMinDeltaYAtLayer4AllTracksToReference = yDeltaLayer4Min;
  TrackerMinDeltaXAtLayer9AllTracksToReference = xDeltaLayer9Min;
  TrackerMinDeltaYAtLayer9AllTracksToReference = yDeltaLayer9Min;

  TrackerSumOfUnusedRigidities = sumOfUnusedRigidities;
  TrackerNumberOfVertices = tracker.Vertices().size();

  if (particle && particle->HasTrackerTrack()) {
    const auto& ChargeYiJia = associatedTrackerTrack->ChargeYiJia();
    TrackerCharge = ChargeYiJia.Charge();
    TrackerChargeError = ChargeYiJia.Error();
    const Analysis::SplineTrack* spline = particle->GetSplineTrack();
   // TrackerCharge = associatedTrackerTrack->GetChargeAndError(3, 1 /* electron mass */).Charge();
    TrackerPattern = clampTo<Char_t>(particle->TrackerLayerPatternClassification());
    TrackerHitPatternY = static_cast<UShort_t>(associatedTrackerTrack->LayerYPatternBitset().to_ulong());
    TrackerHitPatternXY = static_cast<UShort_t>(associatedTrackerTrack->LayerXYPatternBitset().to_ulong());
    TrackerNumberOfLayersInnerWithYHit = clampTo<Short_t>(associatedTrackerTrack->NHyInner());
    CCBDT= particle->TrackerTrack()->CCBDT();

    for (unsigned int layer = 1; layer <= 9; ++layer)
      TrackerCharges().emplace_back(associatedTrackerTrack->GetLayerCharge(layer, -3, 1 /* electron mass */));

    assert(TrackerClusterDistances().empty());
    for (unsigned int trackerLayer = 0; trackerLayer < 9; ++trackerLayer)
      TrackerClusterDistances().push_back(associatedTrackerTrack->ClusterDistances()[trackerLayer]);

    for (unsigned int trackerLayer = 0; trackerLayer < 9; ++trackerLayer) {
      TrackerRigiditiesWithoutHitInLayer().push_back(-10000.0);
      TrackerClusterSignalRatios().push_back(-1.0);
    }

    for (const AC::TrackerReconstructedHit& reconstructedHit : associatedTrackerTrack->ReconstructedHits()) {
      TrackerRigiditiesWithoutHitInLayer().at(int(reconstructedHit.Layer()) - 1) = reconstructedHit.RigidityWithoutThisHit();
      TrackerClusterSignalRatios().at(int(reconstructedHit.Layer()) - 1) = reconstructedHit.SignalRatio();
    }

    // Choutko max-span
    int choutkoMaxSpanIndex = associatedTrackerTrack->GetFitFuzzy(AC::Choutko, AC::All, refitPattern, AC::DefaultMass);
    if (choutkoMaxSpanIndex >= 0) {
      auto& trackFit = associatedTrackerTrack->TrackFits().at(choutkoMaxSpanIndex);
      TrackerTrackChoutkoMaxSpanRigidity = trackFit.Rigidity();
      TrackerTrackChoutkoMaxSpanRigidityRelError = std::abs(trackFit.InverseRigidityError() * TrackerTrackChoutkoMaxSpanRigidity());
      TrackerTrackChoutkoMaxSpanChiSquareX = trackFit.ChiSquareNormalizedX();
      TrackerTrackChoutkoMaxSpanChiSquareY = trackFit.ChiSquareNormalizedY();
    }

        // GBL max-span
    int GBLMaxSpanIndex = associatedTrackerTrack->GetFitFuzzy(AC::GBL, AC::All, refitPatternV6, AC::DefaultMass);
    if (GBLMaxSpanIndex >= 0) {
      auto& trackFit = associatedTrackerTrack->TrackFits().at(GBLMaxSpanIndex);
      TrackerTrackGBLMaxSpanRigidity = trackFit.Rigidity();
      TrackerTrackGBLMaxSpanRigidityRelError = std::abs(trackFit.InverseRigidityError() * TrackerTrackGBLMaxSpanRigidity());
      TrackerTrackGBLMaxSpanChiSquareX = trackFit.ChiSquareNormalizedX();
      TrackerTrackGBLMaxSpanChiSquareY = trackFit.ChiSquareNormalizedY();
    }
        // Kalman max-span
    int KalmanMaxSpanIndex = associatedTrackerTrack->GetFitFuzzy(AC::Kalman, AC::All, refitPatternifneeded, AC::DefaultMass);
    if (KalmanMaxSpanIndex >= 0) {
      auto& trackFit = associatedTrackerTrack->TrackFits().at(KalmanMaxSpanIndex);
      TrackerTrackKalmanMaxSpanRigidity = trackFit.Rigidity();
      TrackerTrackKalmanMaxSpanRigidityRelError = std::abs(trackFit.InverseRigidityError() * TrackerTrackKalmanMaxSpanRigidity());
      TrackerTrackKalmanMaxSpanChiSquareX = trackFit.ChiSquareNormalizedX();
      TrackerTrackKalmanMaxSpanChiSquareY = trackFit.ChiSquareNormalizedY();

    }
// Sanity checks.
     // if (std::abs(particle->Rigidity() - TrackerTrackChoutkoMaxSpanRigidity()) > 1e-6)
      //  WARN_OUT << "Rigidity: " << particle->Rigidity() << " Refit: " <<  TrackerTrackChoutkoMaxSpanRigidity() << " difference!" << std::endl;

     // if (std::abs(particle->Chi2TrackerY() - TrackerTrackChoutkoMaxSpanChiSquareY()) > 1e-6)
       // WARN_OUT << "Chi2TrackerY: " << particle->Chi2TrackerY() << " Refit: " << TrackerTrackChoutkoMaxSpanChiSquareY() << " difference!" << std::endl;
    

    // Choutko max-span (electron mass)
    int choutkoMaxSpanElectronMassIndex = associatedTrackerTrack->GetFitFuzzy(AC::Choutko, AC::All, refitPattern, AC::ElectronMass);
    if (choutkoMaxSpanElectronMassIndex >= 0) {
      auto& trackFit = associatedTrackerTrack->TrackFits().at(choutkoMaxSpanElectronMassIndex);
      TrackerTrackChoutkoMaxSpanElecMassRigidity = trackFit.Rigidity();
      TrackerTrackChoutkoMaxSpanElecMassRigidityRelError = std::abs(trackFit.InverseRigidityError() * TrackerTrackChoutkoMaxSpanElecMassRigidity());
      TrackerTrackChoutkoMaxSpanElecMassChiSquareX = trackFit.ChiSquareNormalizedX();
      TrackerTrackChoutkoMaxSpanElecMassChiSquareY = trackFit.ChiSquareNormalizedY();
    }

    // Choutko inner-only
    int choutkoInnerOnlyIndex = associatedTrackerTrack->GetFitFuzzy(AC::Choutko, AC::Inner, refitPattern, AC::DefaultMass);
    if (choutkoInnerOnlyIndex >= 0) {
      auto& trackFit = associatedTrackerTrack->TrackFits().at(choutkoInnerOnlyIndex);
      TrackerTrackChoutkoInnerOnlyRigidity = trackFit.Rigidity();
      TrackerTrackChoutkoInnerOnlyRigidityRelError = std::abs(trackFit.InverseRigidityError() * TrackerTrackChoutkoInnerOnlyRigidity());
      TrackerTrackChoutkoInnerOnlyChiSquareY = trackFit.ChiSquareNormalizedY();
    }

    // Chikanian max-span
    int chikanianMaxSpanIndex = associatedTrackerTrack->GetFitFuzzy(AC::ChikanianF, AC::All, refitPattern, AC::DefaultMass);
    if (chikanianMaxSpanIndex >= 0) {
      auto& trackFit = associatedTrackerTrack->TrackFits().at(chikanianMaxSpanIndex);
      TrackerTrackChikanianMaxSpanRigidity = trackFit.Rigidity();
      TrackerTrackChikanianMaxSpanRigidityRelError = std::abs(trackFit.InverseRigidityError() * TrackerTrackChikanianMaxSpanRigidity());
      TrackerTrackChikanianMaxSpanChiSquareY = trackFit.ChiSquareNormalizedY();
    }

    // Choutko upper half
    int choutkoUpperHalfIndex = associatedTrackerTrack->GetFitFuzzy(AC::Choutko, AC::UpperHalf, refitPattern, AC::DefaultMass);
    if (choutkoUpperHalfIndex >= 0) {
      auto& trackFit = associatedTrackerTrack->TrackFits().at(choutkoUpperHalfIndex);
      TrackerTrackChoutkoUpperHalfRigidity = trackFit.Rigidity();
      TrackerTrackChoutkoUpperHalfRigidityRelError = std::abs(trackFit.InverseRigidityError() * TrackerTrackChoutkoUpperHalfRigidity());
      TrackerTrackChoutkoUpperHalfChiSquareY = trackFit.ChiSquareNormalizedY();
    }

    // Choutko lower half
    int choutkoLowerHalfIndex = associatedTrackerTrack->GetFitFuzzy(AC::Choutko, AC::LowerHalf, refitPattern, AC::DefaultMass);
    if (choutkoLowerHalfIndex >= 0) {
      auto& trackFit = associatedTrackerTrack->TrackFits().at(choutkoLowerHalfIndex);
      TrackerTrackChoutkoLowerHalfRigidity = trackFit.Rigidity();
      TrackerTrackChoutkoLowerHalfRigidityRelError = std::abs(trackFit.InverseRigidityError() * TrackerTrackChoutkoLowerHalfRigidity());
      TrackerTrackChoutkoLowerHalfChiSquareY = trackFit.ChiSquareNormalizedY();
    }

    if (spline) {
      if (particle->HasEcalShower()) {
        const Vector3& ecalCentreOfGravity = particle->EcalCentreOfGravity();
        const Vector3& pointAtEcalCentreOfGravity = spline->PositionAtZ(ecalCentreOfGravity.Z());
        TrackerTrackEcalCogDeltaX = ecalCentreOfGravity.X() - pointAtEcalCentreOfGravity.X();
        TrackerTrackEcalCogDeltaY = ecalCentreOfGravity.Y() - pointAtEcalCentreOfGravity.Y();
      }

      const Vector3& pointAtEcalTop = spline->PositionAtZ(AC::AMSGeometry::ZECALUpper);
      const Vector3& pointAtEcalBottom = spline->PositionAtZ(AC::AMSGeometry::ZECALLower);
      TrackerTrackAtEcalTopX = pointAtEcalTop.X();
      TrackerTrackAtEcalTopY = pointAtEcalTop.Y();
      TrackerTrackAtEcalBottomX = pointAtEcalBottom.X();
      TrackerTrackAtEcalBottomY = pointAtEcalBottom.Y();
    }
  }

  TrackerTrackIsNotInSolarArrayShadow = true;
  const AC::Event::ParticlesVector& particles = event.RawEvent()->Particles();
  for (unsigned int i = 0; i < particles.size(); ++i) {
    const AC::Particle& particle = particles[i];
    if (particle.IsInSolarArrayShadow()) {
      TrackerTrackIsNotInSolarArrayShadow = false;
      break;
    }
  }
 
   // new ECAL 3D BDT Estimator
  const AC::ECAL& ecal = event.RawEvent()->ECAL();
  EcalBDT3D = ecal.BDTEstimator3D();
  TotalEnergy3D = ecal.TotalEnergy3D(); 

  //Ecal new BDT estimator  **************************
   // event.RawEvent()->ECAL().BDTEstimator3D();
  //**********************************************

  // ECAL
  EcalNumberOfShowers = clampTo<UChar_t>(event.NumberOfEcalShower());
  //EcalBDT3D = ecal.BDTEstimator3D();
  //TotalEnergy3D = ecal.TotalEnergy3D();
  double ecalShowerEnergyDepositedMaximum = 0.0;
  double ecalShowerEnergyElectronMaximum = 0.0;
  double ecalShowerEnergyElectronNewMaximum = 0.0;
  double allEcalShowersEnergyDeposited = 0.0;

  const AC::ECAL::ShowersVector& showers = event.RawEvent()->ECAL().Showers();
  for (const auto& shower : showers) {
    Vector3 sTrackerTrackPoint;
    Vector3 sTrackerTrackDirection;

    if (const Analysis::SplineTrack* spline = particle->GetSplineTrack())
      spline->CalculateLocalPositionAndDirection(AC::AMSGeometry::ZECALUpper, sTrackerTrackPoint, sTrackerTrackDirection);
    else {
      sTrackerTrackPoint.SetXYZ(0, 0, 0);
      sTrackerTrackDirection.SetXYZ(0, 0, 0);
    }

    double depositedEnergy = shower.DepositedEnergyCorrectedForAnodeEfficiencyInMeV();
    if (depositedEnergy < 0) {
      depositedEnergy = 0.0;
      WARN_OUT_WITH_EVENT << "Deposited energy is negative. This should never happen. Aborting!" << std::endl;
    }

    allEcalShowersEnergyDeposited += depositedEnergy;

    if (depositedEnergy > ecalShowerEnergyDepositedMaximum)
      ecalShowerEnergyDepositedMaximum = depositedEnergy;

    double electronEnergy = shower.ReconstructedEnergyElectron();
    if (electronEnergy > ecalShowerEnergyElectronMaximum)
      ecalShowerEnergyElectronMaximum = electronEnergy;

    double electronEnergyNew = shower.ReconstructedEnergyElectron2017();
    if (std::abs(electronEnergyNew - shower.ReconstructedEnergyElectron2017()) > 1e-2)
      WARN_OUT_ON_MASTER_WITH_EVENT << "Energies don't match. Self computed ElectronEnergy2017=" << electronEnergyNew << " stored in ACQt file, ElectronEnergy2017=" << shower.ReconstructedEnergyElectron2017() << std::endl;

    if (electronEnergyNew > ecalShowerEnergyElectronNewMaximum)
      ecalShowerEnergyElectronNewMaximum = electronEnergyNew;

    if (&shower != particle->EcalShower())
      continue;

    EnergyFractionInLastTwoLayers = shower.RelativeRearLeak();
    EcalEnergyDepositedRaw = shower.DepositedEnergyInMeV();
    EcalEnergyDeposited = depositedEnergy;
    EcalEnergyElectron = electronEnergy;
    EcalEnergyElectronNew = electronEnergyNew;

    EcalChiSquareLongitudinal = shower.ChiSquareLongitudinalShowerFit();
    EcalShowerMaximum =shower.ShowerMaximumLongitudinalShowerFit(); 
  }

  EcalEnergyElectronMaximumShower = ecalShowerEnergyElectronMaximum;
  EcalEnergyElectronNewMaximumShower = ecalShowerEnergyElectronNewMaximum;
  EcalEnergyDepositedMaximumShower = ecalShowerEnergyDepositedMaximum;
  EcalEnergyDepositedAllShowers = allEcalShowersEnergyDeposited;

  if (particle && particle->HasEcalShower()) {
    const AC::ECALShower* shower = particle->EcalShower();
    assert(shower);
    EcalShowerStatus = clampTo<UInt_t>(shower->Status());
    EcalCentreOfGravityX = shower->X();
    EcalCentreOfGravityY = shower->Y();
    EcalCentreOfGravityZ = shower->Z();

    EcalBDT_v7 = particle->EcalEstimator(AC::ECALShower::BDTv7_EnergyElectron);
    EcalBDTSmoothed_v7 = particle->EcalEstimator(AC::ECALShower::BDTv7_EnergyElectron_Smoothed);

    EcalBDT_v7_EnergyD = particle->EcalEstimator(AC::ECALShower::BDTv7_EnergyD);
    EcalBDTSmoothed_v7_EnergyD = particle->EcalEstimator(AC::ECALShower::BDTv7_EnergyD_Smoothed);

    EcalChiSquareLateral = shower->ChiSquareEcalAxisLateralMethod();
    EcalChiSquareCellRatio = shower->ChiSquareEcalAxisCellRatioMethod();
  }

  // TRD
  TrdNumberOfHits = clampTo<UShort_t>(event.NumberOfTrdRawHits());

  int trdMaxSubLayersXZ = 0;
  for (unsigned int i = 0; i < event.TrdSegmentsXZ().size(); ++i)
    trdMaxSubLayersXZ = std::max(trdMaxSubLayersXZ, event.TrdSegmentsXZ()[i].NumberOfSublayersInSegment());
  TrdMaxSubLayersXZ = clampTo<UChar_t>(trdMaxSubLayersXZ);

  int trdMaxSubLayersYZ = 0;
  int trdFirstSubLayerYZ = 20;
  int trdLastSubLayerYZ = 10;
  for (unsigned int i = 0; i < event.TrdSegmentsYZ().size(); ++i) {
    trdMaxSubLayersYZ = std::max(trdMaxSubLayersYZ, event.TrdSegmentsYZ()[i].NumberOfSublayersInSegment());
    trdFirstSubLayerYZ = std::max(trdFirstSubLayerYZ, event.TrdSegmentsYZ()[i].FirstSublayerInSegment());
    trdLastSubLayerYZ = std::min(trdLastSubLayerYZ, event.TrdSegmentsYZ()[i].LastSublayerInSegment());
  }

  TrdMaxSubLayersYZ = clampTo<UChar_t>(trdMaxSubLayersYZ);
  TrdFirstSubLayerYZ = clampTo<Short_t>(trdFirstSubLayerYZ);
  TrdLastSubLayerYZ = clampTo<Short_t>(trdLastSubLayerYZ);

  // Trd-P: Parametrized PDFs from Aachen.
  if (particle) {
    bool pXeOk = false;
    double pXe = Utilities::SlowControlLookup::Self()->QueryXenonPressure(event.TimeStamp(), pXeOk);
    if (!pXeOk && event.IsISS()) {
      WARN_OUT << "Can not query xenon partial ressure for time stamp=" << event.TimeStamp().AsDouble() << std::endl;
    }

    auto& trdHitsFromTrackerTrack = particle->TrdHitsFromTrackerTrack();
    Analysis::TrdLikelihoodCalculation likelihoodCalculatorTracker(trdHitsFromTrackerTrack, pXe);
    TrdPActiveLayersTracker = clampTo<UShort_t>(likelihoodCalculatorTracker.NumberOfActiveLayers());

    auto& trdHitsFromTrdTrack = particle->TrdHitsFromTrdTrack();
    Analysis::TrdLikelihoodCalculation likelihoodCalculatorStandalone(trdHitsFromTrdTrack, pXe);
    TrdPActiveLayersStandalone = clampTo<UShort_t>(likelihoodCalculatorStandalone.NumberOfActiveLayers());

    auto& trdHitsFromTrdAndTrackerTracks = particle->TrdHitsFromTrdAndTrackerTracks();
    Analysis::TrdLikelihoodCalculation likelihoodCalculatorHybrid(trdHitsFromTrdAndTrackerTracks, pXe);
    TrdPActiveLayersHybrid = clampTo<UShort_t>(likelihoodCalculatorHybrid.NumberOfActiveLayers());

    double energy = EcalEnergyElectronNewMaximumShower();
    if (energy > 0) {
      if (particle->HasTrackerTrack()) {
        TrdPLikelihoodTrackerHitsElectronECAL = likelihoodCalculatorTracker.ComputeTrdLikelihood(ParticleId::Electron, energy);
        TrdPLikelihoodTrackerHitsProtonECAL = likelihoodCalculatorTracker.ComputeTrdLikelihood(ParticleId::Proton, energy * 2.0);
        TrdPLikelihoodTrackerHitsHeliumECAL = likelihoodCalculatorTracker.ComputeTrdLikelihood(ParticleId::Helium, energy * 2.0);
      }

      TrdPLikelihoodHybridHitsElectronECAL = likelihoodCalculatorHybrid.ComputeTrdLikelihood(ParticleId::Electron, energy);
      TrdPLikelihoodHybridHitsProtonECAL = likelihoodCalculatorHybrid.ComputeTrdLikelihood(ParticleId::Proton, energy * 2.0);
      TrdPLikelihoodHybridHitsHeliumECAL = likelihoodCalculatorHybrid.ComputeTrdLikelihood(ParticleId::Helium, energy * 2.0);

      TrdPLikelihoodStandaloneHitsElectronECAL = likelihoodCalculatorStandalone.ComputeTrdLikelihood(ParticleId::Electron, energy);
      TrdPLikelihoodStandaloneHitsProtonECAL = likelihoodCalculatorStandalone.ComputeTrdLikelihood(ParticleId::Proton, energy * 2.0);
      TrdPLikelihoodStandaloneHitsHeliumECAL = likelihoodCalculatorStandalone.ComputeTrdLikelihood(ParticleId::Helium, energy * 2.0);
    }

    if (particle->HasTrackerTrack()) {
      TrdPLikelihoodTrackerHitsElectron = likelihoodCalculatorTracker.ComputeTrdLikelihood(ParticleId::Electron, particle->Rigidity());
      TrdPLikelihoodTrackerHitsProton = likelihoodCalculatorTracker.ComputeTrdLikelihood(ParticleId::Proton, particle->Rigidity());
      TrdPLikelihoodTrackerHitsHelium = likelihoodCalculatorTracker.ComputeTrdLikelihood(ParticleId::Helium, particle->Rigidity());

      TrdPLikelihoodHybridHitsElectron = likelihoodCalculatorHybrid.ComputeTrdLikelihood(ParticleId::Electron, particle->Rigidity());
      TrdPLikelihoodHybridHitsProton = likelihoodCalculatorHybrid.ComputeTrdLikelihood(ParticleId::Proton, particle->Rigidity());
      TrdPLikelihoodHybridHitsHelium = likelihoodCalculatorHybrid.ComputeTrdLikelihood(ParticleId::Helium, particle->Rigidity());

      TrdPLikelihoodStandaloneHitsElectron = likelihoodCalculatorStandalone.ComputeTrdLikelihood(ParticleId::Electron, particle->Rigidity());
      TrdPLikelihoodStandaloneHitsProton = likelihoodCalculatorStandalone.ComputeTrdLikelihood(ParticleId::Proton, particle->Rigidity());
      TrdPLikelihoodStandaloneHitsHelium = likelihoodCalculatorStandalone.ComputeTrdLikelihood(ParticleId::Helium, particle->Rigidity());
    }
  }

  if (particle && particle->HasTrdTrack()) {
    const Analysis::TrdTrack* trdTrack = particle->GetTrdTrack();
    assert(trdTrack);
    
    TrdTrackNumberOfSubLayersXZ = clampTo<UChar_t>(trdTrack->SegmentXZ()->NumberOfSublayersInSegment());
    TrdTrackNumberOfSubLayersYZ = clampTo<UChar_t>(trdTrack->SegmentYZ()->NumberOfSublayersInSegment());
    TrdTrackFirstSubLayerYZ = clampTo<Short_t>(trdTrack->SegmentYZ()->FirstSublayerInSegment());
    TrdTrackLastSubLayerYZ = clampTo<Short_t>(trdTrack->SegmentYZ()->LastSublayerInSegment());

    if (particle->HasEcalShower()) {
      Analysis::TrdTrackEcalMatching& match = Analysis::TrdTrackEcalMatching::SharedInstance();
      double rigidity = TrackerTrackChoutkoMaxSpanRigidity();
      double chargeTimesMomentum = 0.0;
      if (rigidity <= 0)
        chargeTimesMomentum = -std::abs(EcalEnergyElectronNewMaximumShower());
      else
        chargeTimesMomentum = std::abs(EcalEnergyElectronNewMaximumShower());

      match.ApplyMatching(event, *particle->EcalShower(), *particle->GetTrdTrack(), chargeTimesMomentum);
      EcalShowerDirectionZ = match.ShowerDirection().Z();
      TrdTrackEcalCogDeltaX = match.DeltaX();
      TrdTrackEcalCogDeltaY = match.DeltaY();
      TrdTrackEcalCogAngleXZ = match.AngleXZ();
      TrdTrackEcalCogAngleYZ = match.AngleYZ();
    }
  }

  if (AC::CurrentACQtVersion() < AC::ACQtVersion(7, 2))
    return;

  // Electron CC MVA
  double energy = EcalEnergyElectronNewMaximumShower();
  fElectronChargeConfusionMva->ProcessEvent(event);
  if (fElectronChargeConfusionMva->IsApplicable(energy))
    ElectronCCMVABDT = fElectronChargeConfusionMva->EvaluateClassifier(fElectronChargeConfusionMva->CategoryForEvent(energy));
}

void LeptonAnalysisTree::UpdateInMemoryBranches() {

  McEventWeight = ComputeMcEventWeight();
  McEventWeightElectronWithCutOff = ComputeMcEventWeightElectronWithCutOff();
  McEventWeightPositronWithCutOff = ComputeMcEventWeightPositronWithCutOff();
  McEventWeightWithCutOff = ComputeMcEventWeightWithCutOff();
  EcalEnergyBest = EcalEnergyElectronNew();
  EcalEnergyBestMaximumShower = EcalEnergyElectronNewMaximumShower();
  TrackerRigidity = TrackerTrackChoutkoMaxSpanRigidity();
  TrackerChiSquareX = TrackerTrackChoutkoMaxSpanChiSquareX();
  TrackerChiSquareY = TrackerTrackChoutkoMaxSpanChiSquareY();
  TrackerRelativeSagittaError = TrackerTrackChoutkoMaxSpanRigidityRelError();
  TrackerPatternSortedByMDR = ComputeTrackerPatternSortedByMDR();
  TrdPLRElecProt = TrdLRElecProt_Rigidity_HybridHits_TrdP();
  TrdPLRElecProtECAL = TrdLRElecProt_Energy_HybridHits_TrdP();
  TrdPLRElecProtStandaloneECAL = TrdLRElecProt_Energy_TrdHits_TrdP();
  TrdPLRHeliElec = TrdLRHeliElec_Rigidity_HybridHits_TrdP();
  EcalBDTBest = EcalBDT_v7_EnergyD();
  EcalBDTBestSmoothed = EcalBDTSmoothed_v7_EnergyD();
  EnergyOverRigidity = TrackerRigidity() != 0.0f ? EcalEnergyDepositedMaximumShower() / TMath::Abs(TrackerRigidity()) : 0.0f;
  EcalShowerMaximumTransformed = ComputeEcalShowerMaximumTransformed();
  EcalChiSquareLateralNormalized = ComputeEcalChiSquareLateralNormalized();
  TrdEcalChi2 = ComputeTrdEcalChi2();
  TrdPActiveLayers = TrdPActiveLayersHybrid();
  TrackerTrackThetaAtEcalUpper = ComputeTrackerTrackThetaAtEcalUpper();

  if (fReEvaluateElectronCCMVABDT) {
    double energy = EcalEnergyElectronNewMaximumShower();
    fElectronChargeConfusionMva->ProcessEvent([this](Mva::MvaTreeInterface* tree) {
      Mva::ElectronChargeConfusionMvaTree* ccMvaTree = static_cast<Mva::ElectronChargeConfusionMvaTree*>(tree);
      ccMvaTree->TofUpperCharge = TofUpperCharge();
      ccMvaTree->TofLowerCharge = TofLowerCharge();

      ccMvaTree->EcalEnergyDeposited = EcalEnergyDepositedMaximumShower();
      ccMvaTree->TrackerPatternSortedByMDR = TrackerPatternSortedByMDR();
      ccMvaTree->TrackerNumberOfTracks = TrackerNumberOfTracks();
      ccMvaTree->TrackerNumberOfVertices = TrackerNumberOfVertices();
      ccMvaTree->TrackerSumOfUnusedRigidities = TrackerSumOfUnusedRigidities();
      ccMvaTree->TrackerMinDeltaXAtTRDCenterAllTracksToReference = TrackerMinDeltaXAtTRDCenterAllTracksToReference();
      ccMvaTree->TrackerMinDeltaYAtTRDCenterAllTracksToReference = TrackerMinDeltaYAtTRDCenterAllTracksToReference();
      ccMvaTree->TrackerMinDeltaXAtLayer4AllTracksToReference = TrackerMinDeltaXAtLayer4AllTracksToReference();
      ccMvaTree->TrackerMinDeltaYAtLayer4AllTracksToReference = TrackerMinDeltaYAtLayer4AllTracksToReference();
      ccMvaTree->TrackerMinDeltaXAtLayer9AllTracksToReference = TrackerMinDeltaXAtLayer9AllTracksToReference();
      ccMvaTree->TrackerMinDeltaYAtLayer9AllTracksToReference = TrackerMinDeltaYAtLayer9AllTracksToReference();
      ccMvaTree->TrackerHitPatternY = TrackerHitPatternY();
      ccMvaTree->TrackerChargesRaw = TrackerCharges();
      ccMvaTree->TrackerClusterDistances = TrackerClusterDistances();
      ccMvaTree->TrackerClusterSignalRatios = TrackerClusterSignalRatios();
      ccMvaTree->TrackerRigiditiesWithoutHitInLayer = TrackerRigiditiesWithoutHitInLayer();

      ccMvaTree->TrackerTrackChoutkoMaxSpanRigidity = TrackerTrackChoutkoMaxSpanRigidity();
      ccMvaTree->TrackerTrackChoutkoMaxSpanRigidityRelError = TrackerTrackChoutkoMaxSpanRigidityRelError();
      ccMvaTree->TrackerTrackChoutkoMaxSpanChiSquareX = TrackerTrackChoutkoMaxSpanChiSquareX();
      ccMvaTree->TrackerTrackChoutkoMaxSpanChiSquareY = TrackerTrackChoutkoMaxSpanChiSquareY();

      ccMvaTree->TrackerTrackChikanianMaxSpanRigidity = TrackerTrackChikanianMaxSpanRigidity();

      ccMvaTree->TrackerTrackChoutkoUpperHalfRigidity = TrackerTrackChoutkoUpperHalfRigidity();
      ccMvaTree->TrackerTrackChoutkoUpperHalfRigidityRelError = TrackerTrackChoutkoUpperHalfRigidityRelError();

      ccMvaTree->TrackerTrackChoutkoLowerHalfRigidity = TrackerTrackChoutkoLowerHalfRigidity();
      ccMvaTree->TrackerTrackChoutkoLowerHalfRigidityRelError = TrackerTrackChoutkoLowerHalfRigidityRelError();

      ccMvaTree->TrackerTrackChoutkoInnerOnlyRigidity = TrackerTrackChoutkoInnerOnlyRigidity();
      ccMvaTree->TrackerTrackChoutkoInnerOnlyRigidityRelError = TrackerTrackChoutkoInnerOnlyRigidityRelError();

      ccMvaTree->TrackerNumberOfLayersInnerWithYHit = TrackerNumberOfLayersInnerWithYHit();
    });

    if (fElectronChargeConfusionMva->IsApplicable(energy))
      ElectronCCMVABDT = fElectronChargeConfusionMva->EvaluateClassifier(fElectronChargeConfusionMva->CategoryForEvent(energy));
  }
}

Double_t LeptonAnalysisTree::ComputeMcEventWeight() const {

  if (!IsMC()) // Do nothing for ISS data.
    return 1.0;

  ParticleId::Species mcSpecies = ParticleId::ToSpecies(McParticleId());
  assert(mcSpecies == ParticleId::Electron || mcSpecies == ParticleId::Positron || mcSpecies == ParticleId::Proton);

  if (mcSpecies == ParticleId::Electron)
    return McEventWeightElectron();
  if (mcSpecies == ParticleId::Positron)
    return McEventWeightPositron();

  return 1.0; // No reweighting for the protons necessary
}

TF1* MeasuringTimeCorrection() {

  static TF1* sMeasuringTimeCorrection = nullptr;
  if (!sMeasuringTimeCorrection) {
    const Binning::Definition& binning = Binning::Predefined::AbsoluteEnergyBinning();
    sMeasuringTimeCorrection = new TF1("measuringTimeParameterization", MeasuringTimeParameterizationFunction, binning.Min(), binning.Max(), 10);
    sMeasuringTimeCorrection->SetNpx(1e5);

    // Dervied from PlotMeasuringTime.C.
    static const Double_t measuringTimeParameters[10] = { 9.99968e-01, 6.29753e+05, 2.81737e+06, -3.82643e+05, 2.33264e+04, -7.24399e+02, 1.10847e+01, -6.58435e-02, 3.40400e+01, 1.62743e+08 };
    sMeasuringTimeCorrection->SetParameters(measuringTimeParameters);
  }

  return sMeasuringTimeCorrection;
}

Double_t LeptonAnalysisTree::ComputeMcEventWeightElectronWithCutOff() const {

  if (!IsMC()) // Do nothing for ISS data.
    return 1.0;

  double energyScale = EcalEnergyElectronNewMaximumShower();
  if (energyScale <= 0.0)
    energyScale = McGeneratedMomentum();
  return McEventWeightElectron() * MeasuringTimeCorrection()->Eval(energyScale) / MeasuringTimeCorrection()->Eval(100.0) /* At 100 GeV we're sure at a constant level, normalize to that. */;
}

Double_t LeptonAnalysisTree::ComputeMcEventWeightPositronWithCutOff() const {

  if (!IsMC()) // Do nothing for ISS data.
    return 1.0;

  double energyScale = EcalEnergyElectronNewMaximumShower();
  if (energyScale <= 0.0)
    energyScale = McGeneratedMomentum();
  return McEventWeightPositron() * MeasuringTimeCorrection()->Eval(energyScale) / MeasuringTimeCorrection()->Eval(100.0) /* At 100 GeV we're sure at a constant level, normalize to that. */;
}

Double_t LeptonAnalysisTree::ComputeMcEventWeightWithCutOff() const {

  if (!IsMC()) // Do nothing for ISS data.
    return 1.0;

  ParticleId::Species mcSpecies = ParticleId::ToSpecies(McParticleId());
  assert(mcSpecies == ParticleId::Electron || mcSpecies == ParticleId::Positron || mcSpecies == ParticleId::Proton);

  if (mcSpecies == ParticleId::Electron)
    return ComputeMcEventWeightElectronWithCutOff();
  if (mcSpecies == ParticleId::Positron)
    return ComputeMcEventWeightPositronWithCutOff();

  return 1.0; // No reweighting for the protons necessary
}

Float_t LeptonAnalysisTree::ComputeEcalShowerMaximumTransformed() const {

  if (EcalShowerMaximum() < 0)
    return -9.0;

  // New transformation from H. Gast to make tmax (almost) independant of energy.
  return 1.0 / (0.78 + 0.21 / sqrt(EcalEnergyBestMaximumShower())) * (EcalShowerMaximum() - 0.282887 * log10(EcalEnergyBestMaximumShower()) * log10(EcalEnergyBestMaximumShower()) - 1.83598 * log10(EcalEnergyBestMaximumShower()) - 3.12711);
}

Float_t LeptonAnalysisTree::ComputeEcalChiSquareLateralNormalized(bool uncorrected) const {

  // In ACQt 7.0 the stored fNormalizedChiSquareEcalAxisLateralMethod is broken.
  // -> Ignore the stored value and recompute normalization offline
  double chi2 = EcalChiSquareLateral();
  if (chi2 == 0.0)
    return 0.0f;

  double energy = EcalEnergyBestMaximumShower();
  double x = energy;
  if (x <= 15.0)
    x = 15.01;
  else if (x >= 400.0)
    x = 399.99;
  x = log10(x);

  // parameters from gbatch as of 2015-09-04 (EcalPDF Version 2)
  double p0 =  0.281165;
  double p1 = -0.0493095;
  double p2 =  0.120408;
  double p3 = -0.0181409;
  double nchi2 = chi2 - (p0 + p1*x + p2*x*x + p3*x*x*x);
  return uncorrected ? nchi2 : EcalChiSquareLateralNormalizedCorrected(nchi2, energy, IsMC());
}

Float_t LeptonAnalysisTree::ComputeTrackerTrackThetaAtEcalUpper() const {

  double dx = TrackerTrackAtEcalTopX() - TrackerTrackAtEcalBottomX();
  double dy = TrackerTrackAtEcalTopY() - TrackerTrackAtEcalBottomY();
  double dz = AC::AMSGeometry::ZECALUpper - AC::AMSGeometry::ZECALLower;
  return std::atan2(std::sqrt(dx * dx + dy * dy), dz) * RadToDeg();
}

Float_t LeptonAnalysisTree::ComputeTrdEcalChi2() const {

  Analysis::TrdTrackEcalMatching& match = Analysis::TrdTrackEcalMatching::SharedInstance();
  float meanEnergy = EcalEnergyElectronNewMaximumShower();
  float trdTrackEcalCogAngleXZUncertainty = match.AngleXZEnergyDependantCut()->Eval(meanEnergy);
  float trdTrackEcalCogAngleYZUncertainty = match.AngleYZEnergyDependantCut()->Eval(meanEnergy);
  float trdTrackEcalCogDeltaXUncertainty = match.DeltaXEnergyDependantCut()->Eval(meanEnergy);
  float trdTrackEcalCogDeltaYUncertainty = match.DeltaYEnergyDependantCut()->Eval(meanEnergy);

  return 0.25 * (std::pow(TrdTrackEcalCogAngleXZ() / trdTrackEcalCogAngleXZUncertainty, 2) +
                 std::pow(TrdTrackEcalCogAngleYZ() / trdTrackEcalCogAngleYZUncertainty, 2) +
                 std::pow(TrdTrackEcalCogDeltaX() / trdTrackEcalCogDeltaXUncertainty, 2) +
                 std::pow(TrdTrackEcalCogDeltaY() / trdTrackEcalCogDeltaYUncertainty, 2));
}

bool LeptonAnalysisTree::IsMcPrimaryGeneratorInAcceptance() const {

  if (!IsMC())
    return false;

  // Load geometry file for the acceptance manager.
  static Acceptance::AcceptanceManager* sAcceptanceManager = nullptr;
  if (!sAcceptanceManager) {
    sAcceptanceManager = new Acceptance::AcceptanceManager;
    std::string geometryConfigFile = "${MY_ANALYSIS}/Configuration/LeptonAnalysisGeometry.cfg";
    Environment::ExpandEnvironmentVariables(geometryConfigFile);

    IO::FileManagerController::Self()->SetRunType(AC::MCRun);
    Detector::DetectorManager::Self()->SetUpdateGain(false);
    sAcceptanceManager->InitSetup(geometryConfigFile);
  }

  Analysis::StraightLineTrack straightLine;
  straightLine.SetParametersThetaPhi(McGeneratedPositionX(), McGeneratedPositionY(), McGeneratedPositionZ(), McGeneratedDirectionTheta(), McGeneratedDirectionPhi());
  return sAcceptanceManager->TrackPassesAllPlanes(straightLine);
}
