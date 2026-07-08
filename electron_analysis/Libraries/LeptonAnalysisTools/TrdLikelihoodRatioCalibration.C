#include "TrdLikelihoodRatioCalibration.hh"
#include "Environment.hh"

#include <cassert>
#include <TFile.h>
#include <TGraph.h>

float TrdLRElecProt_Energy_TrdHits_TrdP_CalibrationFactor(float ecalEnergyElectron) {

  static TFile* sCalibrationFile = nullptr;
  if (!sCalibrationFile) {
    std::string trdLikelihoodRatioCalibrationFile = "${MY_ANALYSIS}/LookupFiles/LeptonAnalysis_TrdLikelihoodRatioCalibration_TrdLRElecProt_Energy_TrdHits_TrdP.root";
    Environment::ExpandEnvironmentVariables(trdLikelihoodRatioCalibrationFile);
    sCalibrationFile = TFile::Open(trdLikelihoodRatioCalibrationFile.c_str(), "OPEN");
    assert(sCalibrationFile);
  }

  static TGraph* sGraph = nullptr;
  if (!sGraph) {
    sGraph = dynamic_cast<TGraph*>(sCalibrationFile->Get("TrdLRElecProt_Energy_TrdHits_TrdP"));
    assert(sGraph);
  }

  return sGraph->Eval(ecalEnergyElectron);
}

float TrdLRElecProt_Energy_HybridHits_TrdP_CalibrationFactor(float ecalEnergyElectron) {

  static TFile* sCalibrationFile = nullptr;
  if (!sCalibrationFile) {
    std::string trdLikelihoodRatioCalibrationFile = "${MY_ANALYSIS}/LookupFiles/LeptonAnalysis_TrdLikelihoodRatioCalibration_TrdLRElecProt_Energy_HybridHits_TrdP.root";
    Environment::ExpandEnvironmentVariables(trdLikelihoodRatioCalibrationFile);
    sCalibrationFile = TFile::Open(trdLikelihoodRatioCalibrationFile.c_str(), "OPEN");
    assert(sCalibrationFile);
  }

  static TGraph* sGraph = nullptr;
  if (!sGraph) {
    sGraph = dynamic_cast<TGraph*>(sCalibrationFile->Get("TrdLRElecProt_Energy_HybridHits_TrdP"));
    assert(sGraph);
  }

  return sGraph->Eval(ecalEnergyElectron);
}

float TrdLRElecProt_Rigidity_HybridHits_TrdP_CalibrationFactor(float ecalEnergyElectron) {

  static TFile* sCalibrationFile = nullptr;
  if (!sCalibrationFile) {
    std::string trdLikelihoodRatioCalibrationFile = "${MY_ANALYSIS}/LookupFiles/LeptonAnalysis_TrdLikelihoodRatioCalibration_TrdLRElecProt_Rigidity_HybridHits_TrdP.root";
    Environment::ExpandEnvironmentVariables(trdLikelihoodRatioCalibrationFile);
    sCalibrationFile = TFile::Open(trdLikelihoodRatioCalibrationFile.c_str(), "OPEN");
    assert(sCalibrationFile);
  }

  static TGraph* sGraph = nullptr;
  if (!sGraph) {
    sGraph = dynamic_cast<TGraph*>(sCalibrationFile->Get("TrdLRElecProt_Rigidity_HybridHits_TrdP"));
    assert(sGraph);
  }

  return sGraph->Eval(ecalEnergyElectron);
}

float TrdLRHeliElec_Rigidity_HybridHits_TrdP_CalibrationFactor(float ecalEnergyElectron) {

  static TFile* sCalibrationFile = nullptr;
  if (!sCalibrationFile) {
    std::string trdLikelihoodRatioCalibrationFile = "${MY_ANALYSIS}/LookupFiles/LeptonAnalysis_TrdLikelihoodRatioCalibration_TrdLRHeliElec_Rigidity_HybridHits_TrdP.root";
    Environment::ExpandEnvironmentVariables(trdLikelihoodRatioCalibrationFile);
    sCalibrationFile = TFile::Open(trdLikelihoodRatioCalibrationFile.c_str(), "OPEN");
    assert(sCalibrationFile);
  }

  static TGraph* sGraph = nullptr;
  if (!sGraph) {
    sGraph = dynamic_cast<TGraph*>(sCalibrationFile->Get("TrdLRHeliElec_Rigidity_HybridHits_TrdP"));
    assert(sGraph);
  }

  return sGraph->Eval(ecalEnergyElectron);
}

