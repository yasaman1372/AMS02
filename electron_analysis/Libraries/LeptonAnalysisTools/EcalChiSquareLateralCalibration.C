#include "EcalChiSquareLateralCalibration.hh"
#include "Environment.hh"

#include <cassert>
#include <TFile.h>
#include <TGraph.h>

float EcalChiSquareLateralNormalizedCorrected(float nchi2, float ecalEnergyElectron, bool isMC) {

  static TFile* sCalibrationFile = nullptr;
  if (!sCalibrationFile) {
    std::string ecalChiSquareLateralCalibrationFile = "${MY_ANALYSIS}/LookupFiles/LeptonAnalysis_EcalChiSquareLateralNormalizedCalibration.root";
    Environment::ExpandEnvironmentVariables(ecalChiSquareLateralCalibrationFile);
    sCalibrationFile = TFile::Open(ecalChiSquareLateralCalibrationFile.c_str(), "OPEN");
    assert(sCalibrationFile);
  }

  static TGraph* sMeanGraphISS = nullptr;
  static TGraph* sSigmaGraphISS = nullptr;
  if (!sMeanGraphISS) {
    sMeanGraphISS = dynamic_cast<TGraph*>(sCalibrationFile->Get("meanISSGraphSmoothed"));
    assert(sMeanGraphISS);

    sSigmaGraphISS = dynamic_cast<TGraph*>(sCalibrationFile->Get("sigmaISSGraphSmoothed"));
    assert(sSigmaGraphISS);
  }

  static TGraph* sMeanGraphMC = nullptr;
  static TGraph* sSigmaGraphMC = nullptr;
  if (!sMeanGraphMC) {
    sMeanGraphMC = dynamic_cast<TGraph*>(sCalibrationFile->Get("meanMCGraphSmoothed"));
    assert(sMeanGraphMC);

    sSigmaGraphMC = dynamic_cast<TGraph*>(sCalibrationFile->Get("sigmaMCGraphSmoothed"));
    assert(sSigmaGraphMC);
  }

  if (isMC)
    return (nchi2 - sMeanGraphMC->Eval(ecalEnergyElectron)) / sSigmaGraphMC->Eval(ecalEnergyElectron);
  return (nchi2 - sMeanGraphISS->Eval(ecalEnergyElectron)) / sSigmaGraphISS->Eval(ecalEnergyElectron);
}

