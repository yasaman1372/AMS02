#include "PhotonTree.hh"

// ACsoft includes
#include "AnalysisEvent.hh"
#include "ConfigHandler.hh"
#include "EventFactory.hh"
#include "FileManager.hh"
#include "Selector.hh"
#include "SelectionParser.hh"
#include "TreeWriter.hh"
#include "Environment.hh"
#include "McSpectrumScaler.hh"
#include "PowerSpectrum.hh"

#include "CutFactory.hh"
#include "PredefinedBinnings.hh"
#include "TwoSidedCut.hh"

#include "Event.h"


// ROOT includes
#include <TStopwatch.h>
#include <TH1.h>
#include <TMath.h>
#include <TRandom3.h>

#ifdef HAVE_AMS_SUPPORT
#include "DisableWarnings.h"
#include "RichCharge.h"
#include "Tofrec02_ihep.h"
#include "root.h"
#include "tkdcards.h"
#include "EnableWarnings.h"
#endif

#define DEBUG_LEVEL 0
#define INFO_OUT_TAG "am_photon_writer"
#include "debugging.hh"

static std::string timeToString(double time) {

  static char buffer[1024];
  int hh = floor((int)time / 60 / 60);
  int mm = floor((int)time / 60 % 60);
  int ss = floor((int)time % 60);
  sprintf(buffer, "%02d:%02d:%02d", hh, mm, ss);
  return std::string(buffer);
}

int main(int argc, char** argv) {

  // Command line option handling.
  Utilities::ConfigHandler& config = Utilities::ConfigHandler::GetGlobalInstance();
  config.ReadCommandLine(argc, argv);

  config.SetProgramHelpText("am_photon_writer", "Write tree for anti matter search.");

  config.AddHelpExample("Loop over given filelist.", "--inputlist list.txt");

  std::string inputList;
  config.GetValue("OPTIONS", "inputlist", inputList, "List of ACQt input files (full path).");

  std::string resultDirectory;
  config.GetValue("OPTIONS", "resultdir", resultDirectory, "General directory where result files should be stored. Current directory is used if option is not specified.");

  std::string suffix;
  config.GetValue("OPTIONS", "suffix", suffix, "A string identifier to be used in parallel computing, to uniquely identify result files.");

  int maxEntries = -1;
  config.GetValue("OPTIONS", "max-entries", maxEntries, "Maximum number of events to write.");

  bool skipSelection = config.GetFlag("OPTIONS", "skip-selection", "Do not remove events not passing the selection.");

  bool withAmsVariables = config.GetFlag("OPTIONS", "with-ams-variables", "Load and calculate AMS (gbatch) variables.");
  bool withTrackerHits = config.GetFlag("OPTIONS", "with-tracker-hits", "Store individual tracker hits in tree.");
  bool prescaleTrdHits = config.GetFlag("OPTIONS", "prescale-trd-hits", "Only store a part of events with many TRD hits.");
  bool prescaleSameSign = config.GetFlag("OPTIONS", "prescale-same-sign", "Only store a part of events with no ECAL shower and same sign tracks.");
  bool prescaleSingleTrack = config.GetFlag("OPTIONS", "prescale-single-track", "Only store a part of events with a single low rigidity (R<10) track.");
  bool prescaleHadronicShower = config.GetFlag("OPTIONS", "prescale-hadronic-shower", "Only store a part of events with a hadronic ECAL showewith a hadronic ECAL showerr.");

  std::string preselectionNameConfig = "TwoTracks,EcalShower";
  config.GetValue("OPTIONS", "preselection", preselectionNameConfig, "Preselection(s) to apply (any has to be passed), separated by commas.");
  std::vector<std::string> preselectionNames = split(preselectionNameConfig, ",");

  REGISTER_CUT(McPrimaryDoesPairProduction,
               "MC Primary particle undergoes pair production",
               [] (const Analysis::Event& event, double& value) -> bool {
                  const auto& mc = event.RawEvent()->MC();
                  const int primaryTrackId = mc.PrimaryParticle()->TrackID();
                  bool hasElectron = false;
                  bool hasPositron = false;
                  for (const auto& particle : mc.MCParticles()) {
                    if (particle.MotherParticle() > primaryTrackId)
                      break;
                    if (particle.MotherParticle() == primaryTrackId && particle.Process() == AC::MCProcess::PairProduction) {
                      if (particle.ParticleID() == 2)
                        hasPositron = true;
                      else if (particle.ParticleID() == 3)
                        hasElectron = true;
                    }
                    if (hasPositron && hasElectron)
                      break;
                  }
                  value = hasPositron && hasElectron;
                  return value;
               },
               Binning::Predefined::BooleanBinning)

  REGISTER_CUT(McPrimaryDoesHadronicPi0,
               "MC Primary particle undergoes hadronic interaction creating a Pi0",
               [] (const Analysis::Event& event, double& value) -> bool {
                  const auto& mc = event.RawEvent()->MC();
                  const int primaryTrackId = mc.PrimaryParticle()->TrackID();
                  bool hasPi0 = false;
                  for (const auto& particle : mc.MCParticles()) {
                    if (particle.MotherParticle() > primaryTrackId)
                      break;
                    if (particle.MotherParticle() == primaryTrackId && particle.Process() == AC::MCProcess::HadronicInteraction) {
                      if (particle.ParticleID() == 7)
                        hasPi0 = true;
                    }
                    if (hasPi0)
                      break;
                  }
                  value = hasPi0;
                  return value;
               },
               Binning::Predefined::BooleanBinning)

  //REGISTER_CUT(EcalNumberOfShowersKXR,
  //           "Number of ECAL KXR Showers",
  //           [](const Analysis::Event& ev, double& valueForHistograms, double min, double max) -> bool {
  //             const AC::ECAL& ecal = ev.RawEvent()->ECAL();
  //             valueForHistograms = ecal.Showers3D().size();
  //             return Cuts::TwoSidedCut::IsInRange(valueForHistograms, min, max);
  //           },
  //           Binning::Functions::MakeUpToNBinning(20))


  // Load & parse cut configuration file.
  std::string cutConfigfile = "${PHOTONFRAMEWORK}/Configuration/photon_cuts.cfg";
  Environment::ExpandEnvironmentVariables(cutConfigfile);
  config.Read(cutConfigfile);
  Cuts::SelectionParser selectionParser(config);

  // Setup file manager to process ACQt data.
  IO::FileManager fileManager(&config);

  Analysis::EventFactory* eventFactory = Analysis::EventFactory::Create(&config);
  Analysis::Event event;

  // 'AuxiliaryObjectManager' holds all auxiliary histograms / selectors created while processing the ACQt files.
  // NOTE: You should NOT write a TTree together with other histograms etc. in the ROOT file. You most likely
  // want to merge your histograms / selectors from batch jobs, but not the trees. That's why it's a good idea
  // in general to split up in two files: one for holding the tree, one for the rest.
  Utilities::ObjectManager auxiliaryObjectManager("AuxiliaryObjectManager", &config, resultDirectory, suffix);
  auxiliaryObjectManager.SetPrefix("PhotonFramework_Auxiliary");

  // McSpectrumScaler for MC event weights
  Utilities::McSpectrumScaler scaler(&config, resultDirectory, suffix);
  scaler.SetDefaultTargetSpectra();
  eventFactory->RegisterMcSpectrumScaler(&scaler);

  if (!config.PerformChecksAfterOptionParsing())
    return EXIT_FAIL_CONFIG;

  if (!fileManager.ReadFileList(inputList))
    return EXIT_FAIL_FILEMANAGER;

  // Construct tree manager which will manage the output file to hold the resulting tree.
  std::vector<PhotonTree*> trees;
  std::vector<IO::TreeWriter*> treeWriters;
  for (unsigned int index = 0; index < std::pow(2, preselectionNames.size()); ++index) {
    if (index == 0) {
      // no selection passed, don't need to write
      // Create one anyway as a default for prescale weight calculation
      PhotonTree* tree = new PhotonTree;
      trees.push_back(tree);
      tree->fWithAmsVariables = withAmsVariables;
      tree->fWithTrackerHits = withTrackerHits;
      tree->fPrescaleTrdHits = prescaleTrdHits;
      tree->fPrescaleSameSign = prescaleSameSign;
      tree->fPrescaleSingleTrack = prescaleSingleTrack;
      tree->fPrescaleHadronicShower = prescaleHadronicShower;
      treeWriters.push_back(nullptr);
    } else {
      PhotonTree* tree = new PhotonTree;
      trees.push_back(tree);
      tree->fWithAmsVariables = withAmsVariables;
      tree->fWithTrackerHits = withTrackerHits;
      tree->fPrescaleTrdHits = prescaleTrdHits;
      tree->fPrescaleSameSign = prescaleSameSign;
      tree->fPrescaleSingleTrack = prescaleSingleTrack;
      tree->fPrescaleHadronicShower = prescaleHadronicShower;
      IO::TreeWriter* treeWriter = new IO::TreeWriter(tree, IO::TreeOptions::WriteInMemoryBranches);
      treeWriters.push_back(treeWriter);
      std::ostringstream treeName;
      treeName << "PhotonTree";
      for (unsigned int preselectionIndex = 0; preselectionIndex < preselectionNames.size(); ++preselectionIndex) {
        bool preselectionPassed = (index & (1 << preselectionIndex)) > 0;
        if (preselectionPassed) {
          treeName << preselectionNames[preselectionIndex];
        }
      }
      std::string treeFileName = Utilities::ObjectManager::MakeStandardRootFileName(resultDirectory, treeName.str(), suffix);
      treeWriter->Initialize(treeFileName);
    }
  }

  REGISTER_CUT(ThinningCut,
           "Thinning cut for prescaling",
           [&trees] (const Analysis::Event& event, double&) -> bool {
             float weight = 1.0f / trees[0]->CalculatePrescalingWeight(event);
             TRandom3 random;
             random.SetSeed(event.RawEvent()->EventHeader().Random());
             return random.Rndm() < weight;
           },
           Binning::Predefined::BooleanBinning)


  // Load cut selector(s).
  Cuts::Selector* RTISelector = auxiliaryObjectManager.Add(selectionParser.GetSelector("RTI"));
  Cuts::Selector* MCSelector = auxiliaryObjectManager.Add(selectionParser.GetSelector("MC"));
  Cuts::Selector* BadRunsSelector = auxiliaryObjectManager.Add(selectionParser.GetSelector("BadRuns"));
  Cuts::Selector* BasicQualitySelector = auxiliaryObjectManager.Add(selectionParser.GetSelector("BasicQuality"));
  Cuts::Selector* TriggerSelector = auxiliaryObjectManager.Add(selectionParser.GetSelector("Trigger"));
  Cuts::Selector* ThinningSelector = new Cuts::Selector("Thinning", "Thinning");
  std::vector<Cuts::Selector*> PreselectionSelectors;
  for (const auto& name : preselectionNames) {
    PreselectionSelectors.push_back(auxiliaryObjectManager.Add(selectionParser.GetSelector(name)));
  }

  ThinningSelector->RegisterCut(Cuts::CreateCut("ThinningCut"));
  auxiliaryObjectManager.Add(ThinningSelector);

  auto axisFunction = [](const Analysis::Event& event, double& axisValue) { axisValue = event.McMomentum() / 2.0; };
  //auto binningFunction = []()->Binning::Definition { return MakeHeAnalysisSmartBinning(); };
  auto binningFunction = []()->Binning::Definition { return Binning::Predefined::RigidityBinning(); };
  std::string xTitle("R_{MC} / GV");
  MCSelector->SetupCommonXAxisInformation(axisFunction, xTitle, binningFunction);
  BasicQualitySelector->SetupCommonXAxisInformation(axisFunction, xTitle, binningFunction);
  TriggerSelector->SetupCommonXAxisInformation(axisFunction, xTitle, binningFunction);
  ThinningSelector->SetupCommonXAxisInformation(axisFunction, xTitle, binningFunction);
  for (auto& PreselectionSelector : PreselectionSelectors) {
    PreselectionSelector->SetupCommonXAxisInformation(axisFunction, xTitle, binningFunction);
  }

  // Begin event loop.
  TStopwatch stopwatch;
  stopwatch.Start();
  INFO_OUT_ON_MASTER << "Looping over " << fileManager.GetEntries() << " events..." << std::endl;

  int entries = 0;

  while (fileManager.GetNextEvent()) {
    fileManager.DumpEventLoopProgress();

    eventFactory->SetupEmptyEvent(event);
    eventFactory->CreateParticles(event);

    if (!RTISelector->Passes(event) && !skipSelection) continue;
    if (!MCSelector->Passes(event) && !skipSelection) continue;
    if (!BadRunsSelector->Passes(event) && !skipSelection) continue;
    if (!BasicQualitySelector->Passes(event) && !skipSelection) continue;
    if (!TriggerSelector->Passes(event) && !skipSelection) continue;
    unsigned int passesSelectionFlags = 0;
    for (unsigned int preselectionIndex = 0; preselectionIndex < PreselectionSelectors.size(); ++preselectionIndex) {
      auto PreselectionSelector = PreselectionSelectors[preselectionIndex];
      if (PreselectionSelector->Passes(event)) {
        passesSelectionFlags += (1 << preselectionIndex);
      }
    }
    if (passesSelectionFlags == 0 && !skipSelection) continue;
    if (!ThinningSelector->Passes(event) && !skipSelection) continue;

    eventFactory->FillParticles(event, Analysis::CreateSplineTrack);

    // Create TRD segments
    if (event.RawEvent()->TRD().RawHits().size() <= 50) {
      eventFactory->PerformTrdTracking(event);
      eventFactory->PerformTrdSegmentFits(event);
    }


#ifdef HAVE_AMS_SUPPORT
  if (withAmsVariables) {
    RichPMTCalib::loadPmtCorrections = false;
    RichRingR::loadPmtCorrections = false;
    fileManager.AssociatedAMSEvent();
  }
#endif

    treeWriters[passesSelectionFlags]->Fill(event);

    if (++entries >= maxEntries && maxEntries > 0)
      break;
  }

  stopwatch.Stop();

  // Print statistics.
  INFO_OUT << "Timing information for event loop: "
           << "CPUTime " << timeToString(stopwatch.CpuTime()) << " <> "
           << "RealTime " << timeToString(stopwatch.RealTime()) << " <> "
           << "Fraction " << Form("%.1f", 100. * stopwatch.CpuTime() / stopwatch.RealTime()) << "%" << std::endl;

  std::cout << std::endl;
  for (const Cuts::Selector* sel : selectionParser.GetListOfSelectors())
    sel->PrintSummary();

  // Finish writing tree file.
  for (const auto& treeWriter : treeWriters) {
    if (treeWriter)
      treeWriter->Finish();
  }

  // Write auxiliary output file.
  auxiliaryObjectManager.WriteToFile();

  return EXIT_SUCCESS;
}
