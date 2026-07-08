#include "ConfigHandler.hh"
#include "FileManager.hh"
#include "McSpectrumScaler.hh"
#include "McGeneratedEventsCollection.hh"
#include "ObjectManager.hh"
#include "ParticleId.hh"

#include <fstream>

#define INFO_OUT_TAG "am_mc_triggers"
#include "debugging.hh"

namespace Utilities {
void WriteSpectrumScalerWeightFile(McSpectrumScaler& scaler, const IO::FileManager& fm) {
    scaler.Fill(fm);
}
}

int main(int argc, char** argv) {

  Utilities::ConfigHandler config = Utilities::ConfigHandler::GetGlobalInstance();
  config.ReadCommandLine(argc, argv);

  config.SetProgramHelpText("am_mc_triggers", "Write number of triggers of a Mc dataset for PhotonFramework.");
  config.AddHelpExample("Pass MC filelist as filelist.", "--inputlist list.txt");

  std::string inputlist = "";
  config.GetValue("OPTIONS", "inputlist", inputlist, "List of ACQt files.");

  std::string resultdir = ".";
  config.GetValue("OPTIONS", "resultdir", resultdir, "Directory to store results in.");

  std::string speciesName = "";
  config.GetValue("OPTIONS", "species", speciesName, "MC species of the dataset.");

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
  }

}
