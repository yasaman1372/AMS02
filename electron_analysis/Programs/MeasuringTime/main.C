#include "AnalysisEvent.hh"
#include "BinningDefinition.hh"
#include "ConfigHandler.hh"
#include "CutOffMode.hh"
#include "CutFactory.hh"
#include "Environment.hh"
#include "Event.h"
#include "FileManager.hh"
#include "FileManagerController.hh"
#include "MeasuringTime.hh"
#include "MPIEnvironment.hh"
#include "ObjectManager.hh"
#include "StringTools.hh"
#include "Selector.hh"
#include <TTimeStamp.h>

#define INFO_OUT_TAG "ExampleAnalysisHeMeasuringTime"
#include "debugging.hh"


int main(int argc, char** argv) {

  Utilities::ConfigHandler config = Utilities::ConfigHandler::GetGlobalInstance();
  config.ReadCommandLine(argc, argv);

  config.SetProgramHelpText("am_measuring_time", "Create measuring time vs. rigidity histogram.");
  config.AddHelpExample("Pass ISS filelist as filelist.", "--inputlist list.txt");

  std::string resultdir = "results";
  config.GetValue("OPTIONS", "resultdir", resultdir, "Directory to store results in.");

  float cutoffFactor = 1.1f;
  config.GetValue("OPTIONS", "cutoff-factor", cutoffFactor, "Geomagnetic cutoff safety factor.");

  std::string cutoffType = "IGRF";
  config.GetValue("OPTIONS", "cutoff-type", cutoffType, "Geomagnetic cutoff type (IGRF or Stoermer)");

  std::string cutoffModeStr = "40PN";
  config.GetValue("OPTIONS", "cutoff-mode", cutoffModeStr, "Geomagnetic cutoff opening cone (e.g. 40PN)");

  std::string suffix = "";
  config.GetValue("OPTIONS", "suffix", suffix, "Parallel computing identifier.");

  std::string startTimeStr = "1305846311";
  config.GetValue("OPTIONS", "start-time", startTimeStr, "Start of meausuring time (timestamp)");

  std::string endTimeStr = "1591441051";
  config.GetValue("OPTIONS", "end-time", endTimeStr, "Start of meausuring time (YYYY-MM-DD HH:MM:SS)");

  std::string binningFilename = "";
  config.GetValue("OPTIONS", "binning", binningFilename, "Path to file containing rigidity bin edges");

  if (binningFilename.size() == 0)
    return EXIT_FAIL_CONFIG;


  REGISTER_CUT(HasEcalShower3D,

   "Has Ecal Shower 3D",

   [](const Analysis::Event& ev, double& valueForHistograms) -> bool {

   const auto& ecal = ev.RawEvent()->ECAL();

   valueForHistograms = ecal.Showers3D().size();

     return valueForHistograms > 0;

   },
  Binning::Functions::MakeUpToNBinning(5));

  std::string cutConfigFile = "${MY_ANALYSIS}/Configuration/LeptonTreeWriterCuts.cfg";
  Environment::ExpandEnvironmentVariables(cutConfigFile);
  config.Read(cutConfigFile);

  TTimeStamp startTime(stoi(startTimeStr), 0);
  TTimeStamp endTime(stoi(endTimeStr), 0);

  INFO_OUT_ON_MASTER << "Computing measuring time from " << startTime.GetSec() << " to " << endTime.GetSec() << std::endl;

  bool isIGRF = cutoffType == "IGRF";
  RTI::CutOffMode cutoffMode = RTI::CutOffMode::CutOff40PN;
  if (cutoffModeStr == "25P") {
    cutoffMode = RTI::CutOffMode::CutOff25P;
  } else if (cutoffModeStr == "25N") {
    cutoffMode = RTI::CutOffMode::CutOff25N;
  } else if (cutoffModeStr == "25PN") {
    cutoffMode = RTI::CutOffMode::CutOff25PN;
  } else if (cutoffModeStr == "30P") {
    cutoffMode = RTI::CutOffMode::CutOff30P;
  } else if (cutoffModeStr == "30N") {
    cutoffMode = RTI::CutOffMode::CutOff30N;
  } else if (cutoffModeStr == "30PN") {
    cutoffMode = RTI::CutOffMode::CutOff30PN;
  } else if (cutoffModeStr == "35P") {
    cutoffMode = RTI::CutOffMode::CutOff35P;
  } else if (cutoffModeStr == "35N") {
    cutoffMode = RTI::CutOffMode::CutOff35N;
  } else if (cutoffModeStr == "35PN") {
    cutoffMode = RTI::CutOffMode::CutOff35PN;
  } else if (cutoffModeStr == "40P") {
    cutoffMode = RTI::CutOffMode::CutOff40P;
  } else if (cutoffModeStr == "40N") {
    cutoffMode = RTI::CutOffMode::CutOff40N;
  } else if (cutoffModeStr == "40PN") {
    cutoffMode = RTI::CutOffMode::CutOff40PN;
  } else {
    INFO_OUT_ON_MASTER << "Unknown cutoff mode " << cutoffModeStr << std::endl;
    return EXIT_FAIL_CONFIG;
  }

  std::vector<double> edges;
  std::ifstream binningFile(binningFilename);
  if (binningFile.fail()) {
    FATAL_OUT << "Could not open binning file " << binningFilename << std::endl;
  }
  while (!binningFile.eof()) {
    double edge = -10000.0f;
    binningFile >> edge;
    if (edge > -9999.0f)
      edges.push_back(edge);
  }

  auto binning = Binning::Tools::FromVector(edges);
  IO::FileManagerController::Self()->SetRunType(AC::ISSRun);
  
  if (IO::MPIEnvironment::IsMPIEnabled())
    IO::MPIEnvironment::Create();

  Utilities::ObjectManager objectManager(&config, resultdir, suffix);
  objectManager.SetMergeMPI(true);
  objectManager.SetPrefix("MeasuringTime");

  IO::FileManager fm(&config);

  IO::FileManagerController::Self()->SetFirstAndLastEventTimes(startTime, endTime);

  RTI::MeasuringTime measuringTime(config, objectManager, cutoffFactor, cutoffMode, isIGRF, binning);

  if (!config.PerformChecksAfterOptionParsing())
    return EXIT_FAIL_CONFIG;

  measuringTime.ComputeMeasuringTime();

  INFO_OUT << "Total measuring time: " << measuringTime.IntegratedMeasuringTime() << "s (of " << (endTime.GetTime() - startTime.GetTime()) << "s)" << std::endl;

  objectManager.WriteToFile();

  if (IO::MPIEnvironment::IsMPIEnabled())
    IO::MPIEnvironment::Destroy();

}

