#include "PhotonTree.hh"

#define private public
#include "SplineTrack.hh"
#undef private

#include "AnalysisEvent.hh"
#include "TrdLikelihoodCalculation.hh"
#include "BadRunManager.hh"
#include "CoordinateSystems.hh"
#include "DeadStrawLookup.hh"
#include "DetectorManager.hh"
#include "MvaImplementation.hh"
#include "MvaInterface.hh"
#include "HeliumChargeConfusionMvaRS.hh"
#include "TrackerFootPositions.hh"
#include "RichRadiatorContour.hh"
#include "StraightLineTrack.hh"
#include "TofChargeCalculator.hh"
#include "TrackerLayerContour.hh"
#include "TrdContour.hh"
#include "TrdHitFactory.hh"
#include "TrackFactory.hh"
#include "third_party/TofGeometry.h"

#include "AMSGeometry.h"
#include "BinningDefinition.hh"
#include "Environment.hh"
#include "Event.h"
#include "MCProcess.h"
#include "RICHRing.h"

#include "third_party/FrameTrans.h"

#include <TGraph.h>
#include <TH1.h>
#include <TMath.h>
#include <TRandom3.h>

#include <cmath>

#ifdef HAVE_AMS_SUPPORT
#include "DisableWarnings.h"
#pragma GCC diagnostic ignored "-Wunused-function"
#include "root.h"
#include "EnableWarnings.h"
#endif

#define DEBUG_LEVEL 0
#define INFO_OUT_TAG "PhotonTree"
#include "debugging.hh"

static double issVelocityInGTOD(const Utilities::CoordinateSystems::ISSParameters& issParameters) {

  double x, y, z;
  double vx, vy, vz;
  Utilities::ThirdParty::FT_Angular2Cart(issParameters.distanceInKM, issParameters.latitudeInRadians, issParameters.longitudeInRadians, x, y, z);
  Utilities::ThirdParty::FT_Angular2Cart(1.0, issParameters.velocityLatitudeInRadians, issParameters.velocityLongitudeInRadians, vx, vy, vz);

  // One orbit every 92.6 minutes -> Speed = (Orbit Circumference / (92.6 * 60s))
  double vECI = issParameters.distanceInKM * 2 * M_PI / (92.6 * 60.0);

  double omega = -2 * M_PI / 86400.0;
  double p = 2 * (vx * omega * y - vy * omega * x);
  double q = (omega * y) * (omega * y) + (omega * x) * (omega * x) - vECI * vECI;

  double vGTOD = -p / 2. + std::sqrt(p * p / 4. - q);
  return vGTOD;
}

float PhotonTree::CalculatePrescalingWeight(const Analysis::Event& event) {

  float weight = 1.0f;
  int nEcalShowers = event.RawEvent()->ECAL().Showers3D().size();
  const auto& tracks = event.RawEvent()->Tracker().Tracks();
  int nTracks = tracks.size();
  if (fPrescaleTrdHits) {
    int nTrdHits = event.RawEvent()->TRD().RawHits().size();
    if (nTrdHits >= 16)
      weight *= 10.0f;
    if (nTrdHits >= 26)
      weight *= 10.0f;
  }
  if (fPrescaleSameSign) {
    if (nEcalShowers == 0) {
      if (nTracks >= 2) {
        bool samesign = true;
        float prevsign = 0;
        for (const auto& track : tracks) {
          const auto& fitid = track.GetFitDistinct(AC::GBL, AC::All, AC::RebuildFromTDV + AC::AlignV6, AC::ElectronMass);
          if (fitid >= 0) {
            const auto& fit = track.TrackFits()[fitid];
            float sign = fit.Rigidity() > 0 ? 1 : -1;
            if (prevsign == 0) {
              prevsign = sign;
            } else {
              if (sign != prevsign) {
                samesign = false; 
              }
            }
          }
        }
        if (samesign)
          weight *= 10;
      }
    }
  }
  if (fPrescaleSingleTrack) {
    if (nEcalShowers == 0 && nTracks == 1) {
      const auto& track = tracks[0];
      int fitid = track.GetFitDistinct(AC::GBL, AC::All, AC::RebuildFromTDV + AC::AlignV6, AC::ElectronMass);
      if (fitid < 0) {
        weight *= 10;
      } else {
        const auto& fit = track.TrackFits()[fitid];
        if (fit.Rigidity() < 20)
          weight *= 10;
      }
    }
  }
  if (fPrescaleHadronicShower) {
    if (nEcalShowers >= 1 && nTracks <= 1) {
      if (event.RawEvent()->ECAL().ReweightedLayerLikelihoodEstimator3D() >= 3.5) {
        weight *= 10;
      }
    }
  }
  return weight;
}


PhotonTree::PhotonTree()
  : IO::TreeInterface("PhotonTree", "Tree for ECAL and Conversion based Gamma ray analysis."),
    fWithAmsVariables(false),
    fWithTrackerHits(false),
    fPrescaleTrdHits(false),
    fPrescaleSameSign(false),
    fPrescaleSingleTrack(false),
    fPrescaleHadronicShower(false)
    {

  fTrackFactory = new Analysis::TrackFactory();
  fTrdHitFactory = new Analysis::TrdHitFactory();

  RegisterBranches();

  Analysis::SplineTrack::sWarnOutOfBounds = false;

}


void PhotonTree::Fill(const Analysis::Event& event) {

  Time = event.TimeStamp().GetSec();
  UTCTime = event.RawEvent()->EventHeader().UTCTimeStamp().AsDouble();
  RunNumber = event.Run();
  EventNumber = event.EventNumber();
  McWeight = event.McWeight();
  PrescalingWeight = CalculatePrescalingWeight(event);
  PhotonTreeVersion = 0;

  // ISS Parameters for coordinate transforms
  const Utilities::CoordinateSystems::ISSParameters issParameters(event.RawEvent()->EventHeader());

  double time = issParameters.utcTimeStamp;
  double posIssR = issParameters.distanceInKM;
  double posIssLong = issParameters.longitudeInRadians;
  double posIssLat = issParameters.latitudeInRadians;
  double posIssX, posIssY, posIssZ;
  Utilities::ThirdParty::FT_Angular2Cart(posIssR, posIssLat, posIssLong, posIssX, posIssY, posIssZ);
  double velIssR = issParameters.velocityInKmPerSecond > 0.0 ? issParameters.velocityInKmPerSecond : issVelocityInGTOD(issParameters);
  double velIssLong = issParameters.velocityLongitudeInRadians;
  double velIssLat = issParameters.velocityLatitudeInRadians;
  double velIssX, velIssY, velIssZ;
  Utilities::ThirdParty::FT_Angular2Cart(velIssR, velIssLat, velIssLong, velIssX, velIssY, velIssZ);
  Utilities::ThirdParty::FT_GTOD_to_TEME_Bugfix(posIssX, posIssY, posIssZ, velIssX, velIssY, velIssZ, time);

  double issYaw = issParameters.yawInRadians;
  double issPitch = issParameters.pitchInRadians;
  double issRoll = issParameters.rollInRadians;

  ISSParameters().push_back(posIssX);
  ISSParameters().push_back(posIssY);
  ISSParameters().push_back(posIssZ);
  ISSParameters().push_back(velIssX);
  ISSParameters().push_back(velIssY);
  ISSParameters().push_back(velIssZ);
  ISSParameters().push_back(issYaw);
  ISSParameters().push_back(issPitch);
  ISSParameters().push_back(issRoll);

  // Trigger

  const AC::Trigger& trigger = event.RawEvent()->Trigger();
  TriggerRateFT = trigger.TriggerRateFT();
  TriggerRateL1 = trigger.TriggerRateLV1();
  TriggerFlags = trigger.TriggerFlags();

  if (event.IsMC()) {
    McParticleID = event.McParticleId();
    McMomentum = event.McMomentum();
    const auto& primaryMcParticle = event.RawEvent()->MC().PrimaryParticle();
    const auto& eventGenerators = primaryMcParticle->EventGenerators();
    const auto& primaryGen = eventGenerators[0];
    const auto& finalGen = eventGenerators[eventGenerators.size() - 1];
    McPrimaryX = primaryGen.X0();
    McPrimaryY = primaryGen.Y0();
    McPrimaryZ = primaryGen.Z0();
    Vector3 mcPos(primaryGen.X0(), primaryGen.Y0(), primaryGen.Z0());
    Vector3 mcDir;
    mcDir.SetNormThetaPhi(1, primaryGen.Theta(), primaryGen.Phi());
    McPrimaryDirXYZ().push_back(mcDir.X());
    McPrimaryDirXYZ().push_back(mcDir.Y());
    McPrimaryDirXYZ().push_back(mcDir.Z());

    const auto mcSplineTrack = event.GetMcPrimarySplineTrack();
    mcSplineTrack->SetWarnOutOfBounds(false);
    float mcTrackMinYZ = mcSplineTrack->SplineZY().GetXmin();
    float mcTrackMinXZ = mcSplineTrack->SplineZX().GetXmin();
    float mcTrackMaxYZ = mcSplineTrack->SplineZY().GetXmax();
    float mcTrackMaxXZ = mcSplineTrack->SplineZX().GetXmax();
    McTrackZMin = std::min(mcTrackMinXZ, mcTrackMinYZ);
    McTrackZMax = std::max(mcTrackMaxXZ, mcTrackMaxYZ);
    for (int layer = 1; layer < 10; ++layer) {
      auto z = AC::AMSGeometry::ZTrackerLayer[layer - 1];
      const Vector3& trackAtLayer = mcSplineTrack->PositionAtZ(z);
      McTrackCoordX().push_back(trackAtLayer.X());
      McTrackCoordY().push_back(trackAtLayer.Y());
      McTrackCoordZ().push_back(z);
      Vector3 mcTrackDir = mcSplineTrack->DirectionAtZ(z);
      McTrackDirXYZ().push_back(mcTrackDir.X());
      McTrackDirXYZ().push_back(mcTrackDir.Y());
      McTrackDirXYZ().push_back(mcTrackDir.Z());
    }
    McPrimaryFinalMomentum = finalGen.Momentum();
    McPrimaryFinalX = finalGen.X0();
    McPrimaryFinalY = finalGen.Y0();
    McPrimaryFinalZ = finalGen.Z0();
    //Vector3 mcPos(finalGen.X0(), finalGen.Y0(), finalGen.Z0());
    Vector3 mcFinalDir;
    mcFinalDir.SetNormThetaPhi(1, finalGen.Theta(), finalGen.Phi());
    McPrimaryFinalDirXYZ().push_back(mcFinalDir.X());
    McPrimaryFinalDirXYZ().push_back(mcFinalDir.Y());
    McPrimaryFinalDirXYZ().push_back(mcFinalDir.Z());

    std::vector<uint16_t> mcSecondaryTrackIDs;
    std::map<uint16_t, float> mcSecondaryEnergyLoss;
    std::map<uint16_t, float> mcSecondaryEnergyLossL3;
    std::map<uint16_t, float> mcSecondaryEnergyLossL56;
    std::map<uint16_t, float> mcSecondaryEnergyLossL8;

    for (const auto& secondary : event.RawEvent()->MC().MCParticles()) {
      if (secondary.MotherParticle() > primaryMcParticle->TrackID())
        break;
      if (secondary.MotherParticle() == primaryMcParticle->TrackID()) {
        const auto& secondaryEvtGen = secondary.EventGenerators()[0];
        McSecondaryParticleIDs().push_back(secondary.ParticleID());
        McSecondaryCreatingProcesses().push_back(secondary.Process());
        McSecondaryMomenta().push_back(secondaryEvtGen.Momentum());
        McSecondaryPositions().push_back(secondaryEvtGen.X0());
        McSecondaryPositions().push_back(secondaryEvtGen.Y0());
        McSecondaryPositions().push_back(secondaryEvtGen.Z0());
        Vector3 evtGenDir;
        evtGenDir.SetNormThetaPhi(1, secondaryEvtGen.Theta(), secondaryEvtGen.Phi());
        McSecondaryDirectionsXYZ().push_back(evtGenDir.X());
        McSecondaryDirectionsXYZ().push_back(evtGenDir.Y());
        McSecondaryDirectionsXYZ().push_back(evtGenDir.Z());
        mcSecondaryTrackIDs.push_back(secondary.TrackID());
      }
    }
    for (uint16_t trackid : mcSecondaryTrackIDs) {
      mcSecondaryEnergyLoss[trackid] = 0.0f;
      mcSecondaryEnergyLossL3[trackid] = 0.0f;
      mcSecondaryEnergyLossL56[trackid] = 0.0f;
      mcSecondaryEnergyLossL8[trackid] = 0.0f;
    }
    for (const auto& mcParticle : event.RawEvent()->MC().MCParticles()) {
      if (mcParticle.Process() == AC::Bremsstrahlung) {
        if (mcSecondaryEnergyLoss.find(mcParticle.MotherParticle()) != mcSecondaryEnergyLoss.end()) {
          auto& evtGen = mcParticle.EventGenerators()[0];
          float energyLoss = evtGen.Momentum();
          mcSecondaryEnergyLoss[mcParticle.MotherParticle()] += energyLoss;
          if (evtGen.Z0() >= AC::AMSGeometry::ZTrackerLayer3)
            mcSecondaryEnergyLossL3[mcParticle.MotherParticle()] += energyLoss;
          if (evtGen.Z0() >= AC::AMSGeometry::ZTrackerLayer56)
            mcSecondaryEnergyLossL56[mcParticle.MotherParticle()] += energyLoss;
          if (evtGen.Z0() >= AC::AMSGeometry::ZTrackerLayer8)
            mcSecondaryEnergyLossL8[mcParticle.MotherParticle()] += energyLoss;
        }
      }
    }
    for (uint16_t secondaryTrackId : mcSecondaryTrackIDs) {
      McSecondaryBremsstrahlungEnergyLossTotal().push_back(mcSecondaryEnergyLoss[secondaryTrackId]);
      McSecondaryBremsstrahlungEnergyLossBeforeL3().push_back(mcSecondaryEnergyLossL3[secondaryTrackId]);
      McSecondaryBremsstrahlungEnergyLossBeforeL56().push_back(mcSecondaryEnergyLossL56[secondaryTrackId]);
      McSecondaryBremsstrahlungEnergyLossBeforeL8().push_back(mcSecondaryEnergyLossL8[secondaryTrackId]);
    }
  }

  // ACC

  AccNClusters = event.NumberOfAccHits();
  AccNClustersTrigger = trigger.ACCFlagsBitset().count();
  for (const auto& cluster : event.RawEvent()->ACC().Clusters()) {
    AccClustersZ().push_back(cluster.Z());
    AccClustersPhi().push_back(cluster.PhiInDegrees());
    AccClustersEnergy().push_back(cluster.Energy());
    AccClustersNumberOfPairs().push_back(cluster.NumberOfPairs());
  }

  // Coordinates

  Longitude = event.LongitudeDegree();
  Latitude = event.LatitudeDegree();

  // ToF
  const auto& tof = event.RawEvent()->TOF();
  for (unsigned int layer = 0; layer < 4; ++layer) {
    TofClustersInLayer().push_back(0);
  }
  for (unsigned int clusterIndex = 0; clusterIndex < tof.Clusters().size(); ++clusterIndex) {
    const auto& cluster = tof.Clusters()[clusterIndex];
    TofClustersInLayer()[cluster.Layer()] += 1;
    TofClusterCoordinates().push_back(cluster.X());
    TofClusterCoordinates().push_back(cluster.Y());
    TofClusterCoordinates().push_back(cluster.Z());
    TofClusterEnergies().push_back(cluster.EnergyInMeV());
    TofClusterCharges().push_back(cluster.Charge());
    TofClusterLayers().push_back(cluster.Layer());
  }

  // Tracker
  const auto& tracker = event.RawEvent()->Tracker();
  TrkNTracks = tracker.Tracks().size();
  std::vector<const AC::TrackerTrackCoordinates*> trackCoordinates;

  for (unsigned int trackIndex = 0; trackIndex < tracker.Tracks().size(); ++trackIndex) {
    const auto& track = tracker.Tracks()[trackIndex];
    const auto& trackCoords = track.AllTrackFitCoordinates();
    TrkTrackHits().push_back(track.NHxyInner());
    TrkTrackLayerPattern().push_back(static_cast<short>(track.LayerYPatternBitset().to_ulong()));
    TrkTrackCharges().push_back(track.ChargeYiJia().Charge());
    TrkTrackChargeErrors().push_back(track.ChargeYiJia().Error());

    std::vector<float> layerCharges;
    std::vector<float> layerCoordinatesX;
    std::vector<float> layerCoordinatesY;
    std::vector<float> layerCoordinatesZ;
    for (unsigned int layer = 0; layer < 9; ++layer) {
      layerCharges.push_back(0.0f);
      layerCoordinatesX.push_back(-100.0f);
      layerCoordinatesY.push_back(-100.0f);
      layerCoordinatesZ.push_back(-200.0f);
    }
    for (const auto& hit : track.ReconstructedHits()) {
      layerCharges[hit.Layer() - 1] = hit.ChargeYiJiaXY();
      layerCoordinatesX[hit.Layer() - 1] = hit.X();
      layerCoordinatesX[hit.Layer() - 1] = hit.Y();
      layerCoordinatesX[hit.Layer() - 1] = hit.Z();
    }
    for (unsigned int layer = 0; layer < 9; ++layer) {
      TrkTrackLayerCharges().push_back(layerCharges[layer]); 
      TrkTrackHitCoordinates().push_back(layerCoordinatesX[layer]);
      TrkTrackHitCoordinates().push_back(layerCoordinatesY[layer]);
      TrkTrackHitCoordinates().push_back(layerCoordinatesZ[layer]);
    }


    int fitid = -1;
    int fitidAlt = -1;
    if (AC::CurrentACQtVersion() < AC::ACQtVersion(8, 0)) {
      fitid = track.GetFitDistinct(AC::Choutko, AC::Inner, AC::RebuildFromTDV + AC::PGMA, AC::AdaptiveMass);
    } else {
      fitid = track.GetFitDistinct(AC::GBL, AC::Inner, AC::RebuildFromTDV + AC::AlignV6, AC::AdaptiveMass);
      fitidAlt = track.GetFitDistinct(AC::GBL, AC::All, AC::RebuildFromTDV + AC::AlignV6, AC::AdaptiveMass);
      if (fitid < 0)
        fitid = fitidAlt;
    }
    int fitidElectron = -1;
    if (AC::CurrentACQtVersion() < AC::ACQtVersion(8, 0)) {
      fitidElectron = track.GetFitDistinct(AC::Choutko, AC::Inner, AC::RebuildFromTDV + AC::PGMA, AC::ElectronMass);
    } else {
      fitidElectron = track.GetFitDistinct(AC::GBL, AC::Inner, AC::RebuildFromTDV + AC::AlignV6, AC::ElectronMass);
      if (fitidElectron < 0)
        fitidElectron = track.GetFitDistinct(AC::GBL, AC::All, AC::RebuildFromTDV + AC::AlignV6, AC::ElectronMass);
    }

    const AC::TrackerTrackCoordinates* coords = nullptr;
    if (fitid >= 0) {
      const auto& fit = track.TrackFits().at(fitid);
      TrkTrackRigidities().push_back(fit.Rigidity());
      TrkTrackInverseRigidityErrors().push_back(fit.InverseRigidityError());
      TrkTrackRigidityChiSquaresY().push_back(fit.ChiSquareNormalizedY());
      TrkTrackRigidityChiSquaresX().push_back(fit.ChiSquareNormalizedX());
      TrkTrackFitResidualsL1Y().push_back(fit.YResidualJLayer1());
    } else {
      TrkTrackRigidities().push_back(0.0f);
      TrkTrackInverseRigidityErrors().push_back(0.0f);
      TrkTrackRigidityChiSquaresY().push_back(0.0f);
      TrkTrackRigidityChiSquaresX().push_back(0.0f);
      TrkTrackFitResidualsL1Y().push_back(0.0f);
    }
    for (unsigned int coordsIndex = 0; coordsIndex < trackCoords.size(); ++coordsIndex) {
      if (trackCoords.at(coordsIndex).TrackFitIndex() == fitid || trackCoords.at(coordsIndex).TrackFitIndex() == fitidAlt) {
        coords = &trackCoords.at(coordsIndex);
        break;
      }
    }
    if (fitidElectron >= 0) {
      const auto& fit = track.TrackFits().at(fitidElectron);
      TrkTrackElectronRigidities().push_back(fit.Rigidity());
      TrkTrackInverseElectronRigidityErrors().push_back(fit.InverseRigidityError());
      TrkTrackElectronRigidityChiSquaresY().push_back(fit.ChiSquareNormalizedY());
      TrkTrackElectronRigidityChiSquaresX().push_back(fit.ChiSquareNormalizedX());
      TrkTrackElectronFitResidualsL1Y().push_back(fit.YResidualJLayer1());
    } else {
      TrkTrackElectronRigidities().push_back(0.0f);
      TrkTrackInverseElectronRigidityErrors().push_back(0.0f);
      TrkTrackElectronRigidityChiSquaresY().push_back(0.0f);
      TrkTrackElectronRigidityChiSquaresX().push_back(0.0f);
      TrkTrackElectronFitResidualsL1Y().push_back(0.0f);
    }
    trackCoordinates.push_back(coords);
    if (coords != nullptr) {
      Analysis::SplineTrack spline;
      fTrackFactory->CreateSplineTrackFrom(*coords, spline);
      const Vector3 posUTof = spline.PositionAtZ(AC::AMSGeometry::ZTOFUpper);
      const Vector3 dirUTof = spline.DirectionAtZ(AC::AMSGeometry::ZTOFUpper);
      const Vector3 posLTof = spline.PositionAtZ(AC::AMSGeometry::ZTOFLower);
      const Vector3 dirLTof = spline.DirectionAtZ(AC::AMSGeometry::ZTOFLower);
      const Vector3 posLTRD = spline.PositionAtZ(AC::AMSGeometry::ZTRDLower);
      const Vector3 dirLTRD = spline.DirectionAtZ(AC::AMSGeometry::ZTRDLower);
      const Vector3 posEcal = spline.PositionAtZ(AC::AMSGeometry::ZECALUpper);
      const Vector3 dirEcal = spline.DirectionAtZ(AC::AMSGeometry::ZECALUpper);
      TrkTrackCoordinatesAtUpperTof().push_back(posUTof.X());
      TrkTrackCoordinatesAtUpperTof().push_back(posUTof.Y());
      TrkTrackCoordinatesAtUpperTof().push_back(posUTof.Z());
      TrkTrackCoordinatesAtLowerTof().push_back(posLTof.X());
      TrkTrackCoordinatesAtLowerTof().push_back(posLTof.Y());
      TrkTrackCoordinatesAtLowerTof().push_back(posLTof.Z());
      TrkTrackCoordinatesAtLowerTrd().push_back(posLTRD.X());
      TrkTrackCoordinatesAtLowerTrd().push_back(posLTRD.Y());
      TrkTrackCoordinatesAtLowerTrd().push_back(posLTRD.Z());
      TrkTrackCoordinatesAtECAL().push_back(posEcal.X());
      TrkTrackCoordinatesAtECAL().push_back(posEcal.Y());
      TrkTrackCoordinatesAtECAL().push_back(posEcal.Z());
      TrkTrackDirectionsAtUpperTofXYZ().push_back(dirUTof.X());
      TrkTrackDirectionsAtUpperTofXYZ().push_back(dirUTof.Y());
      TrkTrackDirectionsAtUpperTofXYZ().push_back(dirUTof.Z());
      TrkTrackDirectionsAtLowerTofXYZ().push_back(dirLTof.X());
      TrkTrackDirectionsAtLowerTofXYZ().push_back(dirLTof.Y());
      TrkTrackDirectionsAtLowerTofXYZ().push_back(dirLTof.Z());
      TrkTrackDirectionsAtLowerTrdXYZ().push_back(dirLTRD.X());
      TrkTrackDirectionsAtLowerTrdXYZ().push_back(dirLTRD.Y());
      TrkTrackDirectionsAtLowerTrdXYZ().push_back(dirLTRD.Z());
      TrkTrackDirectionsAtECALXYZ().push_back(dirEcal.X());
      TrkTrackDirectionsAtECALXYZ().push_back(dirEcal.Y());
      TrkTrackDirectionsAtECALXYZ().push_back(dirEcal.Z());
      for (unsigned int layer = 1; layer <= 9; ++layer) {
        float z = AC::AMSGeometry::ZTrackerLayer[layer - 1];
        const Vector3 pos = spline.PositionAtZ(z);
        const Vector3 dir = spline.DirectionAtZ(z);
        TrkTrackFitCoordinates().push_back(pos.X());
        TrkTrackFitCoordinates().push_back(pos.Y());
        TrkTrackFitCoordinates().push_back(pos.Z());
        TrkTrackFitDirectionsXYZ().push_back(dir.X());
        TrkTrackFitDirectionsXYZ().push_back(dir.Y());
        TrkTrackFitDirectionsXYZ().push_back(dir.Z());
      }
    } else {
      TrkTrackCoordinatesAtUpperTof().push_back(0.0f);
      TrkTrackCoordinatesAtUpperTof().push_back(0.0f);
      TrkTrackCoordinatesAtUpperTof().push_back(0.0f);
      TrkTrackCoordinatesAtLowerTof().push_back(0.0f);
      TrkTrackCoordinatesAtLowerTof().push_back(0.0f);
      TrkTrackCoordinatesAtLowerTof().push_back(0.0f);
      TrkTrackCoordinatesAtLowerTrd().push_back(0.0f);
      TrkTrackCoordinatesAtLowerTrd().push_back(0.0f);
      TrkTrackCoordinatesAtLowerTrd().push_back(0.0f);
      TrkTrackCoordinatesAtECAL().push_back(0.0f);
      TrkTrackCoordinatesAtECAL().push_back(0.0f);
      TrkTrackCoordinatesAtECAL().push_back(0.0f);
      TrkTrackDirectionsAtUpperTofXYZ().push_back(0.0f);
      TrkTrackDirectionsAtUpperTofXYZ().push_back(0.0f);
      TrkTrackDirectionsAtUpperTofXYZ().push_back(0.0f);
      TrkTrackDirectionsAtLowerTofXYZ().push_back(0.0f);
      TrkTrackDirectionsAtLowerTofXYZ().push_back(0.0f);
      TrkTrackDirectionsAtLowerTofXYZ().push_back(0.0f);
      TrkTrackDirectionsAtLowerTrdXYZ().push_back(0.0f);
      TrkTrackDirectionsAtLowerTrdXYZ().push_back(0.0f);
      TrkTrackDirectionsAtLowerTrdXYZ().push_back(0.0f);
      TrkTrackDirectionsAtECALXYZ().push_back(0.0f);
      TrkTrackDirectionsAtECALXYZ().push_back(0.0f);
      TrkTrackDirectionsAtECALXYZ().push_back(0.0f);
      for (unsigned int layer = 1; layer <= 9; ++layer) {
        for (unsigned int entry = 0; entry < 3; ++entry) {
          TrkTrackFitCoordinates().push_back(0.0f);
          TrkTrackFitDirectionsXYZ().push_back(0.0f);
        }
      }
    }
    float betaValue = 0.0f;
    std::vector<short> clusterIndices;
    for (unsigned short index = 0; index < 4; ++index) {
        clusterIndices.push_back(-index - 1);
    }
    for (const auto& beta : event.RawEvent()->TOF().Betas()) {
      if (static_cast<unsigned int>(beta.TrackerTrackIndex()) == trackIndex) {
        betaValue = beta.Beta();
        for (const auto& tofClusterIndex : beta.TOFClusterIndex()) {
          if (tofClusterIndex >= 0) {
            const auto& tofCluster = event.RawEvent()->TOF().Clusters()[tofClusterIndex];
            clusterIndices[tofCluster.Layer()] = tofClusterIndex;
          }
        }
      }
    }
    TrkTrackAssociatedTofBetas().push_back(betaValue);
    for (unsigned int index = 0; index < 4; ++index) {
      TrkTrackAssociatedTofClusterIndices().push_back(clusterIndices[index]);
    }
  }

  // Tracker Distances
  const float stepSize = 1.0f;
  const float fineStepSize = 0.005f;
  for (unsigned int firstIndex = 0; firstIndex < trackCoordinates.size(); ++firstIndex) {
    for (unsigned int secondIndex = firstIndex + 1; secondIndex < trackCoordinates.size(); ++secondIndex) {
      if (trackCoordinates[firstIndex] == nullptr || trackCoordinates[secondIndex] == nullptr) {
        continue;
      }
      Analysis::SplineTrack firstSpline;
      Analysis::SplineTrack secondSpline;
      fTrackFactory->CreateSplineTrackFrom(*trackCoordinates[firstIndex], firstSpline);
      fTrackFactory->CreateSplineTrackFrom(*trackCoordinates[secondIndex], secondSpline);
      float minDistance = 1000.0f;
      float minDistanceZ = 200.0f;
      float minDistanceAngle = 0.0f;
      Vector3 minDistancePos;
      Vector3 minDistanceDir;
      // coarse
      for (float z = 200.0f; z >= -200.0f; z -= stepSize) {
        const Vector3 firstPos = firstSpline.PositionAtZ(z);
        const Vector3 secondPos = secondSpline.PositionAtZ(z);
        const float distance = (firstPos - secondPos).Norm();
        if (distance < minDistance) {
          minDistance = distance;
          minDistancePos = (firstPos + secondPos) * 0.5f;
          minDistanceZ = z;
          const Vector3 firstDir = firstSpline.DirectionAtZ(z);
          const Vector3 secondDir = secondSpline.DirectionAtZ(z);
          float firstRig = std::abs(TrkTrackElectronRigidities()[firstIndex]);
          float secondRig = std::abs(TrkTrackElectronRigidities()[secondIndex]);
          minDistanceDir = (firstDir * firstRig + secondDir * secondRig) * (1 / (firstRig + secondRig));
          minDistanceAngle = firstDir.Angle(secondDir);
        }
      }
      // fine 
      for (float z = minDistanceZ - stepSize; z <= minDistanceZ + stepSize; z += fineStepSize) {
        const Vector3 firstPos = firstSpline.PositionAtZ(z);
        const Vector3 secondPos = secondSpline.PositionAtZ(z);
        const float distance = (firstPos - secondPos).Norm();
        if (distance < minDistance) {
          minDistance = distance;
          minDistancePos = (firstPos + secondPos) * 0.5f;
          const Vector3 firstDir = firstSpline.DirectionAtZ(z);
          const Vector3 secondDir = secondSpline.DirectionAtZ(z);
          float firstRig = std::abs(TrkTrackElectronRigidities()[firstIndex]);
          float secondRig = std::abs(TrkTrackElectronRigidities()[secondIndex]);
          minDistanceDir = (firstDir * firstRig + secondDir * secondRig) * (1 / (firstRig + secondRig));
          minDistanceAngle = firstDir.Angle(secondDir);
        }       
      }
      TrkTrackPairTrackIndices().push_back(firstIndex);
      TrkTrackPairTrackIndices().push_back(secondIndex);
      TrkTrackPairMinDistances().push_back(minDistance);
      TrkTrackPairMinDistanceCoordinates().push_back(minDistancePos.X());
      TrkTrackPairMinDistanceCoordinates().push_back(minDistancePos.Y());
      TrkTrackPairMinDistanceCoordinates().push_back(minDistancePos.Z());
      TrkTrackPairMinDistanceDirectionsXYZ().push_back(minDistanceDir.X());
      TrkTrackPairMinDistanceDirectionsXYZ().push_back(minDistanceDir.Y());
      TrkTrackPairMinDistanceDirectionsXYZ().push_back(minDistanceDir.Z());
      TrkTrackPairMinDistanceAngles().push_back(minDistanceAngle);
    }
  }

  // Track Vertices
  for (const auto& vertex : tracker.Vertices()) {
    TrkVertexCoordinates().push_back(vertex.X());
    TrkVertexCoordinates().push_back(vertex.Y());
    TrkVertexCoordinates().push_back(vertex.Z());
    TrkVertexCoordinates().push_back(vertex.Theta());
    TrkVertexCoordinates().push_back(vertex.Phi());
    TrkVertexMomentum().push_back(vertex.Momentum());
    TrkVertexTrackIndices().push_back(vertex.FirstTrackIndex());
    TrkVertexTrackIndices().push_back(vertex.SecondTrackIndex());
  }

  // Non-Track Layer Charges
  for (unsigned int layer = 0; layer < 9; ++layer) {
    TrkMaxLayerCharges().push_back(0.0f);
  }
  for (auto& track : event.RawEvent()->Tracker().Tracks()) {
    for (auto& hit : track.ReconstructedHits()) {
      float charge = hit.QY();
      if (hit.QX() > charge)
        charge = hit.QX();
      if (charge > TrkMaxLayerCharges()[hit.Layer() - 1])
        TrkMaxLayerCharges()[hit.Layer() - 1] = charge;
    }
  }
  for (auto& hit : event.RawEvent()->Tracker().UnassociatedReconstructedHits()) {
    float charge = hit.QY();
    if (charge > TrkMaxLayerCharges()[hit.Layer() - 1])
      TrkMaxLayerCharges()[hit.Layer() - 1] = charge;
  }

  if (fWithTrackerHits) {
    std::unordered_map<short, short> clusterIndexMap;
    short clusterIndexX = 0;
    short clusterIndexY = 0;
    short mapIndex = 0;
    for (auto& cluster : event.RawEvent()->Tracker().Clusters()) {
      if (cluster.Orientation() == AC::MeasurementMode::XZMeasurement) {
        TrkClusterLayersX().push_back(cluster.Layer());
        TrkClusterLengthsX().push_back(cluster.ClusterLength());
        TrkClusterOffsetsX().push_back(cluster.FirstStripGlobalIndex());
        for (const auto& amplitude : cluster.Amplitudes()) {
          TrkClusterAmplitudesX().push_back(amplitude.Amplitude());
        }
        clusterIndexMap[mapIndex++] = clusterIndexX++;
      } else {
        TrkClusterLayersY().push_back(cluster.Layer());
        TrkClusterLengthsY().push_back(cluster.ClusterLength());
        TrkClusterOffsetsY().push_back(cluster.FirstStripGlobalIndex());
        TrkClusterCoordinatesY().push_back(cluster.Coordinate());
        for (const auto& amplitude : cluster.Amplitudes()) {
          TrkClusterAmplitudesY().push_back(amplitude.Amplitude());
        }
        clusterIndexMap[mapIndex++] = clusterIndexY++;
      }
    }

    short trackIndex = 0;
    for (auto& track : event.RawEvent()->Tracker().Tracks()) {
      for (auto& hit : track.ReconstructedHits()) {
        TrkHitLayers().push_back(hit.Layer());
        TrkHitCharges().push_back(hit.QX());
        TrkHitCharges().push_back(hit.QY());
        TrkHitCoords().push_back(hit.X());
        TrkHitCoords().push_back(hit.Y());
        TrkHitCoords().push_back(hit.Z());
        TrkHitClusterIndicesX().push_back(hit.ClusterIndexX() > -1 ? clusterIndexMap[hit.ClusterIndexX()] : -1);
        TrkHitClusterIndicesY().push_back(hit.ClusterIndexY() > -1 ? clusterIndexMap[hit.ClusterIndexY()] : -1);
        TrkHitTrackIndices().push_back(trackIndex);
      }
      trackIndex++;
    }
    for (auto& hit : event.RawEvent()->Tracker().UnassociatedReconstructedHits()) {
      TrkHitLayers().push_back(hit.Layer());
      TrkHitCharges().push_back(hit.QX());
      TrkHitCharges().push_back(hit.QY());
      TrkHitCoords().push_back(hit.X());
      TrkHitCoords().push_back(hit.Y());
      TrkHitCoords().push_back(hit.Z());
      TrkHitClusterIndicesX().push_back(hit.ClusterIndexX() > -1 ? clusterIndexMap[hit.ClusterIndexX()] : -1);
      TrkHitClusterIndicesY().push_back(hit.ClusterIndexY() > -1 ? clusterIndexMap[hit.ClusterIndexY()] : -1);
      TrkHitTrackIndices().push_back(-1);
    }


  }

  // TRD
  const auto& trd = event.RawEvent()->TRD();
  std::vector<short> trdHitsInLayer;
  for (unsigned short layer = 0; layer < 20; ++layer) {
    trdHitsInLayer.push_back(0);
  }
  for (const auto& rawHit : trd.RawHits()) {
    trdHitsInLayer[rawHit.Layer()] += 1;
  }
  std::vector<Analysis::TrdHit> trdHits;
  NTrdHits = trd.RawHits().size();
  if (trd.RawHits().size() <= 50) {
    fTrdHitFactory->ProduceBasicTrdHitsFrom(*event.RawEvent(), Analysis::GainCorrection3d, trdHits);
    for (const auto& hit : trdHits) {
      if (Utilities::DeadStrawLookup::Self()->IsStrawDeadAnywhereAlongStraw(hit.GlobalStrawNumber(), event.TimeStamp(), Utilities::DeadStrawType::Noisy)) {
        continue;
      }
      TrdHitCoordinates().push_back(hit.Position3D().X());
      TrdHitCoordinates().push_back(hit.Position3D().Y());
      TrdHitCoordinates().push_back(hit.Position3D().Z());
      TrdHitEnergies().push_back(hit.GetAmplitude());
      TrdHitLayers().push_back(hit.Layer());
    }
  }
  short nActiveLayers = 0;
  for (const auto nHits : trdHitsInLayer) {
    TrdHitsInLayer().push_back(nHits);
    if (nHits > 0)
      nActiveLayers += 1;
  }
  TrdNActiveLayers = nActiveLayers;

  // TRD Segments
  for (const auto& segment : event.TrdSegmentsXZ()) {
    TrdSegmentsXZNHits().push_back(segment.RawHits().size());
    uint32_t sublayerPattern = 0;
    uint32_t mask = 1;
    for (int sublayer = 8; sublayer < 32; ++sublayer) {
      if (segment.HasRawHitOnSublayer(sublayer))
        sublayerPattern += mask;
      mask = mask << 1;
    }
    TrdSegmentsXZSublayerPattern().push_back(sublayerPattern);
    TrdSegmentsXZFitParameters().push_back(segment.SlopeFromFit());
    TrdSegmentsXZFitParameters().push_back(segment.ErrorSlopeFromFit());
    TrdSegmentsXZFitParameters().push_back(segment.OffsetFromFit());
    TrdSegmentsXZFitParameters().push_back(segment.ErrorOffsetFromFit());
    TrdSegmentsXZFitParameters().push_back(segment.OffsetReferenceZ());
    TrdSegmentsXZFitParameters().push_back(segment.ChiSquareFromFit());
  }
  for (const auto& segment : event.TrdSegmentsYZ()) {
    TrdSegmentsYZNHits().push_back(segment.RawHits().size());
    uint32_t sublayerPattern = 0;
    uint32_t mask = 1;
    for (int sublayer = 0; sublayer < 40; ++sublayer) {
      if (segment.HasRawHitOnSublayer(sublayer))
        sublayerPattern += mask;
      mask = mask << 1;
      if (sublayer == 7)
        sublayer += 24;
    }
    TrdSegmentsYZSublayerPattern().push_back(sublayerPattern);
    TrdSegmentsYZFitParameters().push_back(segment.SlopeFromFit());
    TrdSegmentsYZFitParameters().push_back(segment.ErrorSlopeFromFit());
    TrdSegmentsYZFitParameters().push_back(segment.OffsetFromFit());
    TrdSegmentsYZFitParameters().push_back(segment.ErrorOffsetFromFit());
    TrdSegmentsYZFitParameters().push_back(segment.OffsetReferenceZ());
    TrdSegmentsYZFitParameters().push_back(segment.ChiSquareFromFit());
  }

  // ECAL
  NEcalShowers = event.RawEvent()->ECAL().Showers3D().size();
  EcalEnergy = event.RawEvent()->ECAL().TotalEnergy3D();
  EcalBdt = event.RawEvent()->ECAL().BDTEstimator3D();
  EcalIntegralLikelihood = event.RawEvent()->ECAL().IntegralLikelihoodEstimator3D();
  EcalReweightedLikelihood = event.RawEvent()->ECAL().ReweightedLayerLikelihoodEstimator3D();

  for (const auto& shower : event.RawEvent()->ECAL().Showers3D()) {
    EcalShowerEnergies().push_back(shower.Energy());
    EcalShowerPositions().push_back(shower.PositionX());
    EcalShowerPositions().push_back(shower.PositionY());
    EcalShowerPositions().push_back(shower.PositionZ());
    EcalShowerDirectionsXYZ().push_back(shower.DirectionX());
    EcalShowerDirectionsXYZ().push_back(shower.DirectionY());
    EcalShowerDirectionsXYZ().push_back(shower.DirectionZ());
  }

  NEcal2DShowers = event.RawEvent()->ECAL().Showers().size();
  for (const auto& shower : event.RawEvent()->ECAL().Showers()) {
    Ecal2DShowerEnergies().push_back(shower.ReconstructedEnergyPhoton());
    Ecal2DShowerDepositedEnergies().push_back(shower.DepositedEnergyInMeV() / 1000.0f);
    Ecal2DShowerBdts().push_back(shower.Estimator(AC::ECALShower::BDTv7_EnergyElectron_Smoothed));
    Ecal2DShowerPositions().push_back(shower.X());
    Ecal2DShowerPositions().push_back(shower.Y());
    Ecal2DShowerPositions().push_back(shower.Z());
    Ecal2DShowerEntryPositions().push_back(shower.EntryX());
    Ecal2DShowerEntryPositions().push_back(shower.EntryY());
    Ecal2DShowerEntryPositions().push_back(shower.EntryZ());
    Vector3 cogDir;
    cogDir.SetNormThetaPhi(1, shower.ThetaSimpleCoGMethod(), shower.PhiSimpleCoGMethod());
    Ecal2DShowerDirectionsCoGXYZ().push_back(cogDir.X());
    Ecal2DShowerDirectionsCoGXYZ().push_back(cogDir.Y());
    Ecal2DShowerDirectionsCoGXYZ().push_back(cogDir.Z());
    Vector3 crDir;
    crDir.SetNormThetaPhi(1, shower.ThetaCellRatioMethod(), shower.PhiCellRatioMethod());
    Ecal2DShowerDirectionsCRXYZ().push_back(crDir.X());
    Ecal2DShowerDirectionsCRXYZ().push_back(crDir.Y());
    Ecal2DShowerDirectionsCRXYZ().push_back(crDir.Z());
    Vector3 emDir;
    emDir.SetNormThetaPhi(1, shower.ThetaEMDirMethod(), shower.PhiEMDirMethod());
    Ecal2DShowerDirectionsEMXYZ().push_back(emDir.X());
    Ecal2DShowerDirectionsEMXYZ().push_back(emDir.Y());
    Ecal2DShowerDirectionsEMXYZ().push_back(emDir.Z());

    Ecal2DShowerLongitudinalChiSquares().push_back(shower.ChiSquareProfile());
    Ecal2DShowerLateralChiSquares().push_back(shower.ChiSquareLateral());
    Ecal2DShowerRatio1cm().push_back(shower.EnergyAt1CMRatio());
    Ecal2DShowerRatio3cm().push_back(shower.EnergyAt3CMRatio());

    Ecal2DShowerStatus().push_back(shower.Status());
  }
}
