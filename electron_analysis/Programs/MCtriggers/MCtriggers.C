#include "ConfigHandler.hh"
#include "FileManager.hh"
#include "McSpectrumScaler.hh"
#include "McGeneratedEventsCollection.hh"
#include "ObjectManager.hh"
#include "ParticleId.hh"
#include "BinningTools.hh"
#include <fstream>
#include "BinningDefinition.hh"
#include "TH1F.h"

#include "TFile.h"
#define INFO_OUT_TAG "MCtriggers"
#include "debugging.hh"
namespace Utilities {
void WriteSpectrumScalerWeightFile(McSpectrumScaler& scaler, const IO::FileManager& fm) {
    scaler.Fill(fm);
}
}

int main(int argc, char** argv) {

  Utilities::ConfigHandler config = Utilities::ConfigHandler::GetGlobalInstance();
  config.ReadCommandLine(argc, argv);

  config.SetProgramHelpText("MCtriggers", "Write number of triggers of a Mc dataset for AntiMatterSearch.");
  config.AddHelpExample("Pass MC filelist as filelist.", "--inputlist list.txt");

  std::string inputlist = "";
  config.GetValue("OPTIONS", "inputlist", inputlist, "List of ACQt files.");

  std::string resultdir = ".";
  config.GetValue("OPTIONS", "resultdir", resultdir, "Directory to store results in.");

  std::string speciesName = "";
  config.GetValue("OPTIONS", "species", speciesName, "MC species of the dataset.");
  
  std::string binningfileName = "";
  config.GetValue("OPTIONS", "binningfileName", binningfileName, "MC species of the dataset.");
  
  std::string suffix = "";
  config.GetValue("OPTIONS", "suffix", suffix, "Parallel computing identifier.");

  IO::FileManager fm(&config);
  Utilities::McSpectrumScaler scaler(&config, resultdir, suffix);

  if (!config.PerformChecksAfterOptionParsing())
    return EXIT_FAIL_CONFIG;

  const auto species = Utilities::ParticleId::FromString(speciesName);
  if (species == ParticleId::Species::NoSpecies) {
    FATAL_OUT << "No species passed." << std::endl;
  }

  std::vector<double> edges;
  std::ifstream binningFile(binningfileName);
  if (binningFile.fail()) {
    FATAL_OUT << "Could not open binning file " << binningfileName << std::endl;
  }
  while (!binningFile.eof()) {
    double edge = -10000.0f;
    binningFile >> edge;
    if (edge > -9999.0f)
      edges.push_back(edge);
  }

  auto binning = Binning::Tools::FromVector(edges);


  if (!scaler.HasInputFile()) {
    if (!fm.ReadFileList(inputlist))
        return EXIT_FAIL_FILEMANAGER;

    WriteSpectrumScalerWeightFile(scaler, fm);
    scaler.Dump();

    const auto collection = scaler.CollectionForSpecies(species);
    if (!collection) {
      FATAL_OUT << "No collection found for species " << ParticleId::Name(species) << std::endl;
    }

    double pmin = collection->MinimumMomentum();
    double pmax = collection->MaximumMomentum();
    double triggers = collection->Integrate(pmin, pmax);

    INFO_OUT << ParticleId::Name(species) << ": " << pmin << " - " << pmax << " GeV: " << triggers << " triggers." << std::endl;

    std::ofstream resultfile(resultdir + "/triggers.txt");
    resultfile << pmin << " " << pmax << " " << (unsigned long)triggers << std::endl;
    resultfile.close();
    
    TH1* TriggerCounts = Make<TH1F>("TriggerCounts", "Number of MC triggers", binning);

    for (unsigned int index = 1; index <= binning.NumberOfBins();++index) {
    TriggerCounts->SetBinContent(index, collection->Integrate(binning.LowEdge(index), binning.UpEdge(index)));
}
   
   TFile* ofile = new TFile("Triggercount.root", "RECREATE");
   ofile->cd();
   TriggerCounts->Write();
   ofile->Close();   

  }

}

