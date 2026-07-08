#include "IssTree.hh"

// ACsoft includes
#include "AnalysisEvent.hh"
#include "ConfigHandler.hh"
#include "EventFactory.hh"
#include "FileManager.hh"
#include "McSpectrumScaler.hh"
#include "ObjectManager.hh"
#include "Selector.hh"
#include "SelectionParser.hh"
#include "TreeWriter.hh"
#include "Environment.hh"

#include "CutFactory.hh"
#include "PredefinedBinnings.hh"

#include "Event.h"


// ROOT includes
#include <TStopwatch.h>
#include <TH1.h>
#include <TMath.h>
#include <TRandom3.h>

#define DEBUG_LEVEL 0
#define INFO_OUT_TAG "am_iss_writer"
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

  config.SetProgramHelpText("am_iss_writer", "Write tree for ISS positions.");

  config.AddHelpExample("Loop over given filelist.", "--inputlist list.txt");

  std::string inputList;
  config.GetValue("OPTIONS", "inputlist", inputList, "List of ACQt input files (full path).");

  std::string resultDirectory;
  config.GetValue("OPTIONS", "resultdir", resultDirectory, "General directory where result files should be stored. Current directory is used if option is not specified.");

  std::string suffix;
  config.GetValue("OPTIONS", "suffix", suffix, "A string identifier to be used in parallel computing, to uniquely identify result files.");

  int maxEntries = -1;
  config.GetValue("OPTIONS", "max-entries", maxEntries, "Maximum number of events to write.");

  REGISTER_CUT(McPrimaryDoesPairProduction,
               "MC Primary particle undergoes pair production (dummy cut)",
               [] (const Analysis::Event& event, double& value) -> bool {
                 (void) event;
                 value = false;
                 // Dummy implementation, to be able to read the cut config file.
                 return value;
               },
               Binning::Predefined::BooleanBinning)



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
  IssTree* tree = new IssTree;
  IO::TreeWriter treeWriter(tree, IO::TreeOptions::WriteInMemoryBranches);
  std::string treeFileName = Utilities::ObjectManager::MakeStandardRootFileName(resultDirectory, "IssTree", suffix);
  treeWriter.Initialize(treeFileName);

  // Load cut selector(s).
  Cuts::Selector* RTISelector = auxiliaryObjectManager.Add(selectionParser.GetSelector("RTI"));
  Cuts::Selector* MCSelector = auxiliaryObjectManager.Add(selectionParser.GetSelector("MC"));
  Cuts::Selector* BadRunsSelector = auxiliaryObjectManager.Add(selectionParser.GetSelector("BadRuns"));
  Cuts::Selector* BasicQualitySelector = auxiliaryObjectManager.Add(selectionParser.GetSelector("BasicQuality"));
  Cuts::Selector* TriggerSelector = auxiliaryObjectManager.Add(selectionParser.GetSelector("Trigger"));

  auto axisFunction = [](const Analysis::Event& event, double& axisValue) { axisValue = event.McMomentum() / 2.0; };
  //auto binningFunction = []()->Binning::Definition { return MakeHeAnalysisSmartBinning(); };
  auto binningFunction = []()->Binning::Definition { return Binning::Predefined::RigidityBinning(); };
  std::string xTitle("R_{MC} / GV");
  MCSelector->SetupCommonXAxisInformation(axisFunction, xTitle, binningFunction);
  BasicQualitySelector->SetupCommonXAxisInformation(axisFunction, xTitle, binningFunction);
  TriggerSelector->SetupCommonXAxisInformation(axisFunction, xTitle, binningFunction);

  // Begin event loop.
  TStopwatch stopwatch;
  stopwatch.Start();
  INFO_OUT_ON_MASTER << "Looping over " << fileManager.GetEntries() << " events..." << std::endl;

  int entries = 0;
  unsigned long lastTime = 0;

  while (fileManager.GetNextEvent()) {
    fileManager.DumpEventLoopProgress();

    eventFactory->SetupEmptyEvent(event);
    eventFactory->CreateParticles(event);

    if (!RTISelector->Passes(event)) continue;
    if (!MCSelector->Passes(event)) continue;
    if (!BadRunsSelector->Passes(event)) continue;
    if (!BasicQualitySelector->Passes(event)) continue;
    if (!TriggerSelector->Passes(event)) continue;

    unsigned long time = event.TimeStamp().GetSec();
    if (time == lastTime)
      continue;
    lastTime = time;

    treeWriter.Fill(event);

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
  treeWriter.Finish();

  // Write auxiliary output file.
  auxiliaryObjectManager.WriteToFile();

  return EXIT_SUCCESS;
}
