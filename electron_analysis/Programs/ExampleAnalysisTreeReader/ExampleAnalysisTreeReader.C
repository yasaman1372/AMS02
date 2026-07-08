#include "ExampleAnalysisTreeReader.hh"

#include "ExampleAnalysisTree.hh"

// ACsoft includes
#include "BinningFunctions.hh"
#include "BinningTools.hh"
#include "ConfigHandler.hh"
#include "Environment.hh"
#include "ObjectManager.hh"
#include "PredefinedBinnings.hh"
#include "Selector.hh"
#include "SelectionParser.hh"
#include "TreeWriter.hh"

// ROOT includes
#include <TFile.h>
#include <TH1.h>

#define INFO_OUT_TAG "ExampleAnalysisTreeReader"
#include "debugging.hh"

ExampleAnalysisTreeReader::ExampleAnalysisTreeReader()
  : IO::TreeReader()
  , fTree(nullptr)
  , fEcalEnergyElectronHistogram(nullptr)
  , fTrackerChargeLayer9Histogram(nullptr)
  , fEoverAbsRHistogram(nullptr)
  , fTreeAfterSelectionWriter(nullptr)
  , fSelector(nullptr) {

}

ExampleAnalysisTreeReader::~ExampleAnalysisTreeReader() {

  delete fTree;
  delete fEcalEnergyElectronHistogram;
  delete fTrackerChargeLayer9Histogram;
  delete fEoverAbsRHistogram;
  delete fTreeAfterSelectionWriter;
  delete fSelector;
  // Don't delete your cuts, they're owned by the selector.
}

IO::TreeInterface* ExampleAnalysisTreeReader::Initialize(Utilities::ObjectManager& objectManager) {

  assert(!fTree);
  fTree = new ExampleAnalysisTree;

  // Load config file and parse it.
  std::string configFile = "${MY_ANALYSIS}/Configuration/ExampleAnalysisTreeReader.cfg";
  Environment::ExpandEnvironmentVariables(configFile);

  Utilities::ConfigHandler config;
  config.Read(configFile);

  // Book histogram(s).
  const Binning::Definition& binning = Binning::Predefined::AbsoluteEnergyBinning();
  fEcalEnergyElectronHistogram = objectManager.Add(Make<TH1D>("fEcalEnergyElectronHistogram", "ECAL energy (electrons)", binning));
  fTrackerChargeLayer9Histogram = objectManager.Add(Make<TH1D>("fTrackerChargeLayer9Histogram", "Tracker charge layer 9", Binning::Equidistant(100, 0, 10)));
  fEoverAbsRHistogram = objectManager.Add(Make<TH1D>("fEoverAbsRHistogram", "ECAL Energy_{electrons} / |Rigidity|", Binning::Equidistant(200, 0, 20)));

  // Book selector(s).
  Cuts::SelectionParser parser(config, this);
  fSelector = objectManager.Add(parser.GetSelector("SimpleSelector"));

  // In order to fill any cut value histograms, the x axis value must be known.
  // We define this in a generic way using a lamdba function and pass it on
  // to Selector::SetupCommonXAxisInformation() for each selector.
  auto xAxisValue = [this] (const Analysis::Event&, double& lastBaseValue) {
    if (fTree->Rigidity() != 0)
      lastBaseValue = std::abs(fTree->Rigidity());
  };

  fSelector->SetupCommonXAxisInformation(xAxisValue, "|Rigidity| / GV", Binning::Predefined::AbsoluteRigidityBinning);

  // Return the tree interface pointer to the TreeReader, so it knows which kind of tree we're analzing.
  return fTree;
}

void ExampleAnalysisTreeReader::TreeChanged(TTree* tree) {

  IO::TreeReader::TreeChanged(tree);

  // Book tree(s).
  // The idea is to clone the very same tree that we're analyzing, but only write out events that fulfil our selection criteria.
  if (!fTreeAfterSelectionWriter) {
    assert(fOutputFile);
    fTreeAfterSelectionWriter = new IO::TreeWriter(fTree, IO::TreeOptions::WriteInMemoryBranches);
    fTreeAfterSelectionWriter->Initialize(fOutputFile);
  }
}

void ExampleAnalysisTreeReader::ParseOptions(Utilities::ConfigHandler& config) {

  // You could register custom command-line options here.
  IO::TreeReader::ParseOptions(config);
}

bool ExampleAnalysisTreeReader::ProcessEvent() {

  fEcalEnergyElectronHistogram->Fill(fTree->EcalEnergyElectron());

  assert(fTree->TrackerCharges().size() == 9); // We store exactly 9 values for each tracker layer in this branch.
  fTrackerChargeLayer9Histogram->Fill(fTree->TrackerCharges().at(8));

  // Example call for the selector based on config file information.
  if (fSelector->Passes(DummyEvent())) {
    fEoverAbsRHistogram->Fill(fTree->EoverAbsR());
    fTreeAfterSelectionWriter->Fill();
  }

  return true;
}

void ExampleAnalysisTreeReader::Finish() {

  fOutputFile->cd();

  assert(fSelector);
  fSelector->PrintSummary();

  if (fTreeAfterSelectionWriter)
    fTreeAfterSelectionWriter->Finish();
}
