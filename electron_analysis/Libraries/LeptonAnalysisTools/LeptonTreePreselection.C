#include "LeptonTreePreselection.hh"

#include "AMSGeometry.h"
#include "ConfigHandler.hh"
#include "CutFactory.hh"
#include "Event.h"
#include "EventFactory.hh"
#include "Environment.hh"
#include "FileManager.hh"
#include "MeasuringTimeParameterization.hh"
#include "ObjectManager.hh"
#include "SelectionParser.hh"
#include "Selector.hh"
#include "EfficiencyHistograms.hh"
#include "EfficiencyVsTimeHistograms.hh"
#include "ValueHistograms.hh"
#include "AnalysisSettings.hh"
#include "RTIReader.hh"
#include "RTIRecord.hh"

#include <cmath>

#define INFO_OUT_TAG "LeptonTreePreseletion"
#include "debugging.hh"

static const char* gPreselectionStatisticsDirectory = "Preselection/Statistics";

LeptonTreePreselection::LeptonTreePreselection(Utilities::ConfigHandler& config, Utilities::ObjectManager& objectManager, IO::FileManager* fileManager) 
  : fFileManager(fileManager) {

  std::string cutConfigfile = "${MY_ANALYSIS}/Configuration/LeptonTreeWriterCuts.cfg";
  Environment::ExpandEnvironmentVariables(cutConfigfile);
  config.Read(cutConfigfile);

  Cuts::SelectionParser selectionParser(config);
  fBadRunsSelector                = dynamic_cast<Cuts::Selector*>(objectManager.Add(selectionParser.GetSelector("BadRuns"),                      gPreselectionStatisticsDirectory));
  fRtiSelector                    = dynamic_cast<Cuts::Selector*>(objectManager.Add(selectionParser.GetSelector("RTI"),                          gPreselectionStatisticsDirectory));
  fTrdCalibrationSelector         = dynamic_cast<Cuts::Selector*>(objectManager.Add(selectionParser.GetSelector("TrdCalibration"),               gPreselectionStatisticsDirectory));
  fMcPreselectionSelector         = dynamic_cast<Cuts::Selector*>(objectManager.Add(selectionParser.GetSelector("McPreselection"),               gPreselectionStatisticsDirectory));
}

void LeptonTreePreselection::PerformTrdTracking(Analysis::EventFactory& eventFactory, Analysis::Event& event) {

  assert(!event.TrdTrackingDone());
  if (event.NumberOfEcalShower() > 0)
    eventFactory.SetTrdTrackMatchingMethod(Analysis::FromEcalShower);
  else if (event.NumberOfTrackerTracks() > 0)
    eventFactory.SetTrdTrackMatchingMethod(Analysis::FromSplineTrack);
  else
    eventFactory.SetTrdTrackMatchingMethod(Analysis::TrdStandalone);
  eventFactory.PerformTrdTracking(event);
  eventFactory.PerformTrdVertexFinding(event);
}

bool LeptonTreePreselection::ShouldStoreISSEventWithoutEcalShower(const Analysis::Particle* particle) {

  // Prerequisite: no ECAL shower is present
  // Idea: Keep all electron candidates for efficiency studies (to be able to probe if electrons have an ECAL shower as function of rigidity).

  // Case a) if we have no tracker track, discard the event.
  const AC::TrackerTrack* associatedTrackerTrack = particle->TrackerTrack();
  if (!associatedTrackerTrack)
    return false;

  // Case b) if we have no tracker track fit, discard the event.
  const int refitPattern = AC::PGMA + AC::RebuildFromTDV;
  int choutkoMaxSpanIndex = associatedTrackerTrack->GetFitFuzzy(AC::Choutko, AC::All, refitPattern, AC::DefaultMass);
  if (choutkoMaxSpanIndex < 0)
    return false;

  // Case c) if we have no rigidity measurement or it's a proton candidate, discard the event.
  auto& trackFit = associatedTrackerTrack->TrackFits().at(choutkoMaxSpanIndex);
  if (trackFit.Rigidity() >= 0)
    return false;

  // Case d) if we are a negative rigidity event, but have no spline track, discard the event - nothing we can do about it.
  const Analysis::SplineTrack* splineTrack = particle->GetSplineTrack();
  if (!splineTrack)
    return false;

  // Case e) if we are a negative rigidity event, whose tracker track extrapolation doesn't hit the ECAL - discard it.
  const Vector3& ecalEntrance = splineTrack->PositionAtZ(AC::AMSGeometry::ZECALUpper);
  const Vector3& ecalExit     = splineTrack->PositionAtZ(AC::AMSGeometry::ZECALLower);
  if (std::abs(ecalEntrance.X()) > 34.0 || std::abs(ecalEntrance.Y()) > 34.0
   || std::abs(ecalExit.X())     > 34.0 || std::abs(ecalExit.Y())     > 34.0)
    return false;

  // Case f) if we are a negative rigidity event, with inner tracker Z < 0.01 || Z > 2.5 - discard it.
  if (particle->TrackerCharge() < 0.01 || particle->TrackerCharge() > 2.5)
    return false;

  // Case g) if we are a negative rigidity event, with very small/large track fit chi squares - discard it.
  if (particle->Chi2TrackerX() < 0.001 || particle->Chi2TrackerX() > 50.0
   || particle->Chi2TrackerY() < 0.001 || particle->Chi2TrackerY() > 50.0)
    return false;

  return true;
}

void LeptonTreePreselection::ApplyEventPreProcessing(Analysis::Event&) {

  // Potential place to use AMS-ROOT support.
}

bool LeptonTreePreselection::ProcessEvent(Analysis::EventFactory& eventFactory, Analysis::Event& event) {

  ApplyEventPreProcessing(event);

  bool passesBadRuns = true;
  bool passesRTI = true;
  bool passesTrdCalibration = true;
  if (event.IsISS() || event.IsBeamTest())
    passesBadRuns = fBadRunsSelector->Passes(event);

  if (event.IsISS()) {
    passesRTI = fRtiSelector->Passes(event);
    passesTrdCalibration = fTrdCalibrationSelector->Passes(event);
  }

  if (!passesBadRuns || !passesRTI || !passesTrdCalibration)
    return false;

  assert(!event.Particles().size());
  eventFactory.CreateParticles(event);
  eventFactory.FillParticles(event, Analysis::CreateSplineTrack);

  // Ignore all events that do not pass the MC preselection. This is only safe as this is the FIRST selector on MC.
  if (!fMcPreselectionSelector->Passes(event))
    return false;

  double ecalEnergyElectronMaximumShower = 0.0;
  double ecalEnergyElectronNewMaximumShower = 0.0;

  const AC::ECAL::ShowersVector& showers = event.RawEvent()->ECAL().Showers();
  for (const auto& shower : showers) {
    double energyElectron = shower.ReconstructedEnergyElectron();
    if (energyElectron > ecalEnergyElectronMaximumShower)
      ecalEnergyElectronMaximumShower = energyElectron;

    double energyElectronNew = shower.ReconstructedEnergyElectron2017();
    if (energyElectronNew > ecalEnergyElectronNewMaximumShower)
      ecalEnergyElectronNewMaximumShower = energyElectronNew;
  }

  const Analysis::Particle* particle = event.PrimaryParticle();

  // Check if the ECAL shower with the highest energy is over the minimum energy.
  const Binning::Definition& binning = Binning::Predefined::AbsoluteEnergyBinning();
  if (ecalEnergyElectronMaximumShower >= binning.Min() || ecalEnergyElectronNewMaximumShower >= binning.Min()) {
    if (!particle) {
      WARN_OUT_WITH_EVENT << "No particle available, even though we had an ECAL shower. This is a major error! Dumping event:" << std::endl;
      event.RawEvent()->Dump();
      return false;
    }

    // No further requirements for MC/BT data: write all events in the tree.
    if (event.IsMC() || event.IsBeamTest()) {
      // Only ever perform TRD tracking, if needed.
      PerformTrdTracking(eventFactory, event);
      eventFactory.FillParticles(event, Analysis::CreateTrdTrack);
      return true;
    }

    // For ISS data require the ECAL shower with the highest energy to be greater than the geomagnetic cutoff.
    bool overCutOff = true;
    if (event.IsISS()) {
      double cutOff25PN = std::min(std::abs(event.GeomagneticMaxCutOff(RTI::CutOffMode::CutOff25PN)), std::abs(event.IGRFMaxCutOff(RTI::CutOffMode::CutOff25PN)));
      double cutOff30PN = std::min(std::abs(event.GeomagneticMaxCutOff(RTI::CutOffMode::CutOff30PN)), std::abs(event.IGRFMaxCutOff(RTI::CutOffMode::CutOff30PN)));
      double cutOff35PN = std::min(std::abs(event.GeomagneticMaxCutOff(RTI::CutOffMode::CutOff35PN)), std::abs(event.IGRFMaxCutOff(RTI::CutOffMode::CutOff35PN)));
      double cutOff40PN = std::min(std::abs(event.GeomagneticMaxCutOff(RTI::CutOffMode::CutOff40PN)), std::abs(event.IGRFMaxCutOff(RTI::CutOffMode::CutOff40PN)));
      double cutOff = std::min(cutOff25PN, std::min(cutOff30PN, std::min(cutOff35PN, cutOff40PN)));

      if (ecalEnergyElectronMaximumShower < binning.Max() || ecalEnergyElectronNewMaximumShower < binning.Max()) {
        bool overCutOff1 = false;
        if (ecalEnergyElectronMaximumShower >= binning.Min())
          overCutOff1 = binning.FindBinLowEdge(ecalEnergyElectronMaximumShower) > cutOff;

        bool overCutOff2 = false;
        if (ecalEnergyElectronNewMaximumShower >= binning.Min())
          overCutOff2 = binning.FindBinLowEdge(ecalEnergyElectronNewMaximumShower) > cutOff;

        overCutOff = overCutOff1 || overCutOff2;
      }
    }

    if (!overCutOff)
      return false;

    PerformTrdTracking(eventFactory, event);
    eventFactory.FillParticles(event, Analysis::CreateTrdTrack);
    return true;
  }

  // Skip events without a particle on BT/ISS data.
  if (!particle && !event.IsMC())
    return false;

  if (event.IsISS()) {
    if (ShouldStoreISSEventWithoutEcalShower(particle)) {
      PerformTrdTracking(eventFactory, event);
      eventFactory.FillParticles(event, Analysis::CreateTrdTrack);
      return true;
    }

    // Ignore the event on ISS data, if it came here (no ECAL shower available).
    return false;
  }

  // Write all MC/BT events into the tree, even if there is no ECAL shower.
  PerformTrdTracking(eventFactory, event);
  eventFactory.FillParticles(event, Analysis::CreateTrdTrack);
  return true;
}

void LeptonTreePreselection::PrintSummary() const {

  std::cout << std::endl;
  fBadRunsSelector->PrintSummary();
  fRtiSelector->PrintSummary();
  fTrdCalibrationSelector->PrintSummary();
  fMcPreselectionSelector->PrintSummary();
  std::cout << std::endl;
}

