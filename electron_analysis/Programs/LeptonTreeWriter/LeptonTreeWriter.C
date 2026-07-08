#include "AcceptanceManager.hh"
#include "AnalysisEvent.hh"
#include "AnalysisSettings.hh"
#include "ConfigHandler.hh"
#include "Cut.hh"
#include "CustomSpectrum.hh"
#include "ElectronAndPositronFluxModel.hh"
#include "EffectiveMeasuringTime.hh"
#include "EventFactory.hh"
#include "Environment.hh"
#include "FileManager.hh"
#include "LeptonAnalysisTree.hh"
#include "LeptonTreePreselection.hh"
#include "McSpectrumScaler.hh"
#include "ObjectManager.hh"
#include "ParticleFactoryForEcalBasedMatching.hh"
#include "TreeWriter.hh"

#ifdef ENABLE_MPI
#include "MPIEnvironment.hh"
#endif

#include <TApplication.h>
#include <TCanvas.h>
#include <TF1.h>
#include <TH1.h>
#include <TROOT.h>
#include <TCanvas.h>

#define INFO_OUT_TAG "LeptonTreeWriter"
#include "debugging.hh"

int main(int argc, char** argv) {

  gROOT->SetBatch(1);

  // Workaround to avoid ROOT option parsing
  static int gNull = 0;
  TApplication* fApp = new TApplication("Application", &gNull, (char**)0);
  AnalysisSettings::Initialize();

  Utilities::ConfigHandler& config = Utilities::ConfigHandler::GetGlobalInstance();
  config.ReadCommandLine(argc, argv);
  config.SetProgramHelpText("LeptonTreeWriter", "LeptonTreeWriter");
  config.AddHelpExample("Loop over given filelist.", "--inputlist list.txt");

  std::string inputlist = "";
  config.GetValue("OPTIONS", "inputlist", inputlist, "List of ACQT input files (full directory).");

  std::string resultdir = "";
  config.GetValue("OPTIONS", "resultdir", resultdir, "General directory where result files should be stored. Current directory is used if option is not specified.");

  std::string suffix = "";
  config.GetValue("OPTIONS", "suffix", suffix, "A string identifier to be used in parallel computing, to uniquely identify result files.");

  unsigned long long maxEventsToProcess = std::numeric_limits<unsigned long long>::max();
  config.GetValue("OPTIONS", "maxEvents", maxEventsToProcess, "Maximum number of events to process - only use WITHOUT MPI for debugging.");

  int amsRun = -1;
  config.GetValue("OPTIONS", "amsrun", amsRun, "Select specific AMS run.");

  int amsEvent = -1;
  config.GetValue("OPTIONS", "amsevent", amsEvent, "Select specific AMS event.");

  // Used by ISS recipe.
  int toUnixTime = -1;
  int fromUnixTime = -1;

  bool inputlistContainsList = config.GetFlag("OPTIONS", "inputlist-contains-list", "Whether the input list references another list of files or not.");
  if (inputlistContainsList) {
    std::ifstream list(inputlist.c_str());
    if (!list.good())
      FATAL_OUT << "ERROR opening file \"" << inputlist << "\"!" << std::endl;

    std::vector<std::string> fileNames;

    std::string line;
    while (list.good()) {
      std::getline(list, line);
      if (list.eof())
        break;
      fileNames.push_back(line);
    }

    assert(fileNames.size() == 1);
    inputlist = fileNames.front();

    std::vector<std::string> tokens;
    StringTokenize(inputlist, '_', tokens);
    assert(tokens.size() > 2);

    std::string fromUnixTimeString = tokens[tokens.size() - 2];
    std::string toUnixTimeString = tokens[tokens.size() - 1];
    StringReplace(toUnixTimeString, ".list", "");

    toUnixTime = StringToNumber<int>(toUnixTimeString);
    fromUnixTime = StringToNumber<int>(fromUnixTimeString);
    INFO_OUT_ON_MASTER << "Processing ISS data, from unix time=" << fromUnixTime << " toUnixTime=" << toUnixTime << std::endl;
  }

  IO::FileManager fileManager(&config);
  Analysis::EventFactory* eventFactory = Analysis::EventFactory::Create(&config, new Analysis::ParticleFactoryForEcalBasedMatching);

  // 'AuxiliaryObjectManager' holds all auxiliary histograms / selectors created while processing the ACQt files.
  Utilities::ObjectManager auxiliaryObjectManager("AuxiliaryObjectManager", &config, resultdir, suffix);
  Utilities::McSpectrumScaler scaler(&config, resultdir, suffix);

  if (!config.PerformChecksAfterOptionParsing())
    return EXIT_FAIL_CONFIG;

  // If a file list is given and no input file, immediately read the list to extract time binning information.
  assert(!auxiliaryObjectManager.HasFile());
  if (!fileManager.ReadFileList(inputlist)) {
#ifdef ENABLE_MPI
    IO::MPIEnvironment::Create();
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    INFO_OUT_ON_MASTER << "No files to process, still creating the LeptonAnalysis_* ROOT files for consistency." << std::endl;

    std::string leptonTreeFileName = Utilities::ObjectManager::MakeStandardRootFileName(resultdir, "LeptonAnalysis_Tree_ISS", suffix);
    auxiliaryObjectManager.SetPrefix("LeptonAnalysis_Auxiliary_ISS");
    IO::TreeWriter leptonTreeWriter(new LeptonAnalysisTree);
    leptonTreeWriter.Initialize(leptonTreeFileName);

    leptonTreeWriter.Finish();
    auxiliaryObjectManager.WriteToFile();

#ifdef ENABLE_MPI
    IO::MPIEnvironment::Destroy();
#endif
    return EXIT_SUCCESS;
  }

  // Load geometry file for the acceptance manager.
  Acceptance::AcceptanceManager acceptanceManager;
  std::string geometryConfigFile = "${MY_ANALYSIS}/Configuration/LeptonAnalysisGeometry.cfg";
  Environment::ExpandEnvironmentVariables(geometryConfigFile);
  eventFactory->RegisterAcceptanceManager(&acceptanceManager);

  std::string leptonTreeFileName;
  bool isMC = (fileManager.GetRunType() == AC::MCRun);
  bool isBT = (fileManager.GetRunType() == AC::BTRun);
  if (isMC) {
    eventFactory->RegisterMcSpectrumScaler(&scaler);
    auto fluxes = PredefinedElectronPositronModel();
    scaler.SetTargetSpectrum(ParticleId::Electron, Utilities::CustomSpectrum {fluxes.first,  Utilities::KinematicVariable::Energy, ParticleId::Electron});
    scaler.SetTargetSpectrum(ParticleId::Positron, Utilities::CustomSpectrum {fluxes.second, Utilities::KinematicVariable::Energy, ParticleId::Positron});

    leptonTreeFileName = Utilities::ObjectManager::MakeStandardRootFileName(resultdir, "LeptonAnalysis_Tree_MC", suffix);
    auxiliaryObjectManager.SetPrefix("LeptonAnalysis_Auxiliary_MC");
  } else if (isBT) {
    leptonTreeFileName = Utilities::ObjectManager::MakeStandardRootFileName(resultdir, "LeptonAnalysis_Tree_BT", suffix);
    auxiliaryObjectManager.SetPrefix("LeptonAnalysis_Auxiliary_BT");
  } else {
    assert(fileManager.GetRunType() == AC::ISSRun);
    leptonTreeFileName = Utilities::ObjectManager::MakeStandardRootFileName(resultdir, "LeptonAnalysis_Tree_ISS", suffix);
    auxiliaryObjectManager.SetPrefix("LeptonAnalysis_Auxiliary_ISS");
  }

  IO::TreeWriter leptonTreeWriter(new LeptonAnalysisTree);
  leptonTreeWriter.Initialize(leptonTreeFileName);

  LeptonTreePreselection leptonTreePreselection(config, auxiliaryObjectManager, &fileManager);

  INFO_OUT_ON_MASTER << "Looping over " << fileManager.GetEntries() << " events..." << std::endl;
  Analysis::Event event;
  bool firstEvent = true;

  unsigned long long eventCounter = 0;
  while (fileManager.GetNextEvent()) {
    ++eventCounter;
    if (eventCounter > maxEventsToProcess)
      break;

    // Initialize the AcceptanceManager, after the DetectorManager received the run type (needed for MPI).
    if (firstEvent) {
      acceptanceManager.InitSetup(geometryConfigFile);
      firstEvent = false;
    }

    fileManager.DumpEventLoopProgress(20000);
    eventFactory->SetupEmptyEvent(event);

    if (fromUnixTime != -1 && event.EventTime() < fromUnixTime)
      continue;

    if (toUnixTime != -1 && event.EventTime() >= toUnixTime)
      continue;

    if (amsRun != -1 && event.Run() != amsRun)
      continue;

    if (amsEvent != -1 && event.EventNumber() != amsEvent)
      continue;

    if (leptonTreePreselection.ProcessEvent(*eventFactory, event))
      leptonTreeWriter.Fill(event);
  }

  leptonTreePreselection.PrintSummary();

  leptonTreeWriter.Finish();
  auxiliaryObjectManager.WriteToFile();

  (void) fApp;
  return EXIT_SUCCESS;
}

