#include "ExampleAnalysisTree.hh"

// ACsoft includes
#include "Event.h"
#include "CutFactory.hh"
#include "AnalysisEvent.hh"
#include "ConfigHandler.hh"
#include "EventFactory.hh"
#include "FileManager.hh"
#include "Selector.hh"
#include "SelectionParser.hh"
#include "TreeWriter.hh"
#include "Environment.hh"
#include "ObjectManager.hh"
#include "McSpectrumScaler.hh"
#include "PredefinedBinnings.hh"
#define INFO_OUT_TAG "ExampleAnalysisTreeWriter"
#include "debugging.hh"
int main(int argc, char** argv) {

  // Command line option handling.
  Utilities::ConfigHandler& config = Utilities::ConfigHandler::GetGlobalInstance();
  config.ReadCommandLine(argc, argv);

  config.SetProgramHelpText("ExampleAnalysisTreeWriter",
                            "Illustrates the usage of IO::TreeWriter to write ROOT trees from ACQt files.");

  config.AddHelpExample("Loop over given filelist.", "--inputlist list.txt");

  std::string inputList;
  config.GetValue("OPTIONS", "inputlist", inputList,
                  "List of ACQt input files (full path).");

  std::string resultDirectory;
  config.GetValue("OPTIONS", "resultdir", resultDirectory,
                  "General directory where result files should be stored. Current directory is used if option is not specified.");

  std::string suffix;
  config.GetValue("OPTIONS", "suffix", suffix,
                 "A string identifier to be used in parallel computing, to uniquely identify result files.");

  REGISTER_CUT(HasEcalShower3D,

   "Has Ecal Shower 3D",

   [](const Analysis::Event& ev, double& valueForHistograms) -> bool {

   const auto& ecal = ev.RawEvent()->ECAL();

   valueForHistograms = ecal.Showers3D().size();

     return valueForHistograms > 0;

   },
  Binning::Functions::MakeUpToNBinning(5));

  // Load & parse cut configuration file.
  std::string cutConfigfile = "${MY_ANALYSIS}/Configuration/ExampleAnalysisTreeWriter.cfg";
  Environment::ExpandEnvironmentVariables(cutConfigfile);
  config.Read(cutConfigfile);
  Cuts::SelectionParser selectionParser(config);

  // Setup file manager to process ACQt data.
  IO::FileManager fileManager(&config);

  Analysis::EventFactory* eventFactory = Analysis::EventFactory::Create(&config);
  Analysis::Event event;

  // McSpectrumScaler for MC event weights
  Utilities::McSpectrumScaler scaler(&config, resultDirectory, suffix);
  scaler.SetDefaultTargetSpectra();
  eventFactory->RegisterMcSpectrumScaler(&scaler);

  // 'AuxiliaryObjectManager' holds all auxiliary histograms / selectors created while processing the ACQt files.
  // NOTE: You should NOT write a TTree together with other histograms etc. in the ROOT file. You most likely
  // want to merge your histograms / selectors from batch jobs, but not the trees. That's why it's a good idea
  // in general to split up in two files: one for holding the tree, one for the rest.
  Utilities::ObjectManager auxiliaryObjectManager("AuxiliaryObjectManager", &config, resultDirectory, suffix);
  auxiliaryObjectManager.SetPrefix("ExampleAnalysis_Auxiliary");

  if (!config.PerformChecksAfterOptionParsing())
    return EXIT_FAIL_CONFIG;

  if (!fileManager.ReadFileList(inputList))
    return EXIT_FAIL_FILEMANAGER;

  // Construct tree manager which will manage the output file to hold the resulting tree.
  IO::TreeWriter treeWriter(new ExampleAnalysisTree, IO::TreeOptions::DontWriteInMemoryBranches);

  std::string treeFileName = Utilities::ObjectManager::MakeStandardRootFileName(resultDirectory, "ExampleAnalysis_Tree", suffix);
  treeWriter.Initialize(treeFileName);

  // Load cut selector(s).
  Cuts::Selector* preselectionSelector = auxiliaryObjectManager.Add(selectionParser.GetSelector("Preselection"));
  Cuts::Selector* rtiSelector = auxiliaryObjectManager.Add(selectionParser.GetSelector("RTI"));
  Cuts::Selector* mcSelector = auxiliaryObjectManager.Add(selectionParser.GetSelector("MC"));
  Cuts::Selector* badrunsSelector = auxiliaryObjectManager.Add(selectionParser.GetSelector("BadRuns"));
  Cuts::Selector* basicqualitySelector = auxiliaryObjectManager.Add(selectionParser.GetSelector("BasicQuality"));
  Cuts::Selector* triggerSelector = auxiliaryObjectManager.Add(selectionParser.GetSelector("Trigger"));

  auto axisFunction = [](const Analysis::Event& event, double& axisValue)
  { axisValue = event.McMomentum(); };
  auto binningFunction = []()->Binning::Definition { return
  Binning::Predefined::RigidityBinning(); };
  std::string xTitle("R_{MC} / GV");
  preselectionSelector->SetupCommonXAxisInformation(axisFunction, xTitle,
  binningFunction);
  rtiSelector->SetupCommonXAxisInformation(axisFunction, xTitle,
  binningFunction);
  mcSelector->SetupCommonXAxisInformation(axisFunction, xTitle,
  binningFunction);
  badrunsSelector->SetupCommonXAxisInformation(axisFunction, xTitle,
  binningFunction);
  basicqualitySelector->SetupCommonXAxisInformation(axisFunction, xTitle,
  binningFunction);
  triggerSelector->SetupCommonXAxisInformation(axisFunction, xTitle,
  binningFunction);


  // Begin event loop.
  INFO_OUT_ON_MASTER << "Looping over " << fileManager.GetEntries() << " events..." << std::endl;

//  static int sProductionSteps = Analysis::CreateSplineTrack |
//  Analysis::CreateTrdTrack;

  while (fileManager.GetNextEvent()) {
    fileManager.DumpEventLoopProgress(20000);

    eventFactory->SetupEmptyEvent(event);  
    eventFactory->CreateParticles(event);
        
    if (!rtiSelector->Passes(event))
      continue;
    if (!badrunsSelector->Passes(event))
      continue;
     
    eventFactory->FillParticles(event, Analysis::CreateSplineTrack); 
   
    if (!preselectionSelector->Passes(event))
      continue;
    if (!mcSelector->Passes(event))
      continue;  
 
    eventFactory->PerformTrdTracking(event);
    eventFactory->PerformTrdVertexFinding(event);
    eventFactory->FillParticles(event, Analysis::CreateTrdTrack);

    if (!basicqualitySelector->Passes(event))
      continue;
    if (!triggerSelector->Passes(event))
      continue;


    treeWriter.Fill(event);
  }

  // Print Preselection statistics
  preselectionSelector->PrintSummary();
  rtiSelector->PrintSummary();
  mcSelector->PrintSummary();
  badrunsSelector->PrintSummary();
  basicqualitySelector->PrintSummary();
  triggerSelector->PrintSummary();
  // Finish writing tree file.
  treeWriter.Finish();

  // Write auxiliary output file.
  auxiliaryObjectManager.WriteToFile();

  return EXIT_SUCCESS;
}
