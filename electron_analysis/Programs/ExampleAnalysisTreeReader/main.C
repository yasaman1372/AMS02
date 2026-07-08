#include "ExampleAnalysisTreeReader.hh"

// ACsoft includes
#include "ConfigHandler.hh"
#include "ObjectManager.hh"

// ROOT includes
#include <TApplication.h>
#include <TProof.h>
#include <TROOT.h>

#define INFO_OUT_TAG "ExampleAnalysisTreeReader"
#include "debugging.hh"

int main(int argc, char** argv) {

  // Workaround to avoid ROOT option parsing
  static int sNull = 0;
  TApplication* fApp = new TApplication("Application", &sNull, (char**)0);

  Utilities::ConfigHandler& config = Utilities::ConfigHandler::GetGlobalInstance();
  config.ReadCommandLine(argc, argv);

  config.SetProgramHelpText("ExampleAnalysisTreeReader",
                            "Illustrates the usage of IO::TreeManager to read ROOT trees.");

  config.AddHelpExample("Loop over given ROOT tree.", "--input ExampleAnalysis_Tree.root");

  ExampleAnalysisTreeReader eventLoop;
  eventLoop.ParseOptions(config);

  if (!config.PerformChecksAfterOptionParsing())
    return EXIT_FAIL_CONFIG;

  eventLoop.StartEventLoop("ExampleAnalysisTree", "libExampleAnalysisLibrary");

  const bool keepGUI = eventLoop.UseProofMode() && !gROOT->IsBatch();
  if (keepGUI)
    fApp->Run();
  else {
    if (eventLoop.UseProofMode()) {
      gProof->Close();
      gSystem->Exit(EXIT_SUCCESS);
    }
  }

  return EXIT_SUCCESS;
}
