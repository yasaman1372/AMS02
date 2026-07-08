#include "ElectronAndPositronFluxModel.hh"

#include <TCanvas.h>
#include <TF1.h>
#include <TH1D.h>

#include "FitFunction.hh"
#include "HistogramDataset.hh"
#include "ModelAnalysis.hh"
#include "ModelFunctions.hh"
#include "ModellingData.hh"
#include "Utilities.hh"

using Modelling::PowerLawSolarMod;
using Modelling::ExpCutoffPowerLawSolarMod;
using Modelling::SmoothlyBrokenPowerLawSolarMod;

CombinedElectronPositronModel::CombinedElectronPositronModel(double _E0, double _E1)
  : Modelling::Model()
  , E0(_E0)
  , E1(_E1)
  , M(0.511e-3) {
  // start values
  double Nposi = 0.216 * std::pow(E0 / 5.0, -3.75);
  double Nelec = 6.67 * std::pow(E0 / 5.0, -3.85);
  double normS = 3.28e-2 * std::pow(10.0, -2.4) * std::pow(E1 / 50.0, -2.4);

  DefineParameter(Cminus = Modelling::ModelParameter("C_{-}", Nelec, 0.1 * Nelec));
  DefineParameter(gammaMinus = Modelling::ModelParameter("#gamma_{-}", 3.85, 0.05));
  DefineParameter(deltaGamma = Modelling::ModelParameter("#Delta#gamma_{-}", 0.565, 0.02));
  DefineParameter(b = Modelling::ModelParameter("b", 0.417, 0.02));
  DefineParameter(invBreakEnergy = Modelling::ModelParameter("1/E_{b,-}", 0.03, 0.002));

  DefineParameter(Cplus = Modelling::ModelParameter("C_{+}", Nposi, 0.1 * Nposi, 0.0, 120.0));
  DefineParameter(gammaPlus = Modelling::ModelParameter("#gamma_{+}", 3.83, 0.05));

  DefineParameter(Csource = Modelling::ModelParameter("C_{S}", normS, 0.1 * normS, 0.0, 100.0));
  DefineParameter(gammaSource = Modelling::ModelParameter("#gamma_{S}", 2.53, 0.05));
  DefineParameter(lambdaSource = Modelling::ModelParameter("#lambda_{S}", 1.02e-3, 1.e-4));

  DefineParameter(phiMinus = Modelling::ModelParameter("#phi_{-}", 1.41, 0.05));
  DefineParameter(phiPlus = Modelling::ModelParameter("#phi_{+}", 1.02, 0.05));

  double Emin = 0.1;
  double Emax = 6000.0;

  PosiFlux = StyleTF1(new TF1("PosiFlux", this, &CombinedElectronPositronModel::PosiFluxF, Emin, Emax, 0, "CombinedElectronPositronModel", "PosiFluxF"), kCyan);
  ElecFlux = StyleTF1(new TF1("ElecFlux", this, &CombinedElectronPositronModel::ElecFluxF, Emin, Emax, 0, "CombinedElectronPositronModel", "ElecFluxF"), kMagenta);

  PosiPowerLawComponent = StyleTF1(new TF1("PosiPL", this, &CombinedElectronPositronModel::PosiPowerLawComponentF, Emin, Emax, 0, "CombinedElectronPositronModel", "PosiPowerLawComponentF"), kCyan, 7);
  ElecPowerLawComponent = StyleTF1(new TF1("ElecPL", this, &CombinedElectronPositronModel::ElecPowerLawComponentF, Emin, Emax, 0, "CombinedElectronPositronModel", "ElecPowerLawComponentF"), kMagenta, 7);
  SourceTermPosi = StyleTF1(new TF1("SourceTermPosi", this, &CombinedElectronPositronModel::SourceTermPosiF, Emin, Emax, 0, "CombinedElectronPositronModel", "SourceTermPosiF"), kCyan, 7);
  SourceTermElec = StyleTF1(new TF1("SourceTermElec", this, &CombinedElectronPositronModel::SourceTermElecF, Emin, Emax, 0, "CombinedElectronPositronModel", "SourceTermElecF"), kMagenta, 7);

  PosiFluxE3 = StyleTF1(new TF1("PosiFluxE3", this, &CombinedElectronPositronModel::PosiFluxE3F, Emin, Emax, 0, "CombinedElectronPositronModel", "PosiFluxE3F"), kRed);
  ElecFluxE3 = StyleTF1(new TF1("ElecFluxE3", this, &CombinedElectronPositronModel::ElecFluxE3F, Emin, Emax, 0, "CombinedElectronPositronModel", "ElecFluxE3F"), kRed);
  PosiPowerLawComponentE3 = StyleTF1(new TF1("PosiPLE3", this, &CombinedElectronPositronModel::PosiPowerLawComponentE3F, Emin, Emax, 0, "CombinedElectronPositronModel", "PosiPowerLawComponentE3F"), kGreen, 7);
  ElecPowerLawComponentE3 = StyleTF1(new TF1("ElecPLE3", this, &CombinedElectronPositronModel::ElecPowerLawComponentE3F, Emin, Emax, 0, "CombinedElectronPositronModel", "ElecPowerLawComponentE3F"), kGreen, 7);
  SourceTermPosiE3 = StyleTF1(new TF1("SourceTermPosiE3", this, &CombinedElectronPositronModel::SourceTermPosiE3F, Emin, Emax, 0, "CombinedElectronPositronModel", "SourceTermPosiE3F"), kMagenta, 7);
  SourceTermElecE3 = StyleTF1(new TF1("SourceTermElecE3", this, &CombinedElectronPositronModel::SourceTermElecE3F, Emin, Emax, 0, "CombinedElectronPositronModel", "SourceTermElecE3F"), kMagenta, 7);

  PosiFlux->SetNpx(1e6);
  ElecFlux->SetNpx(1e6);

  PosiPowerLawComponent->SetNpx(1e6);
  ElecPowerLawComponent->SetNpx(1e6);
  SourceTermPosi->SetNpx(1e6);
  SourceTermElec->SetNpx(1e6);

  PosiFluxE3->SetNpx(1e6);
  ElecFluxE3->SetNpx(1e6);
  PosiPowerLawComponentE3->SetNpx(1e6);
  ElecPowerLawComponentE3->SetNpx(1e6);
  SourceTermPosiE3->SetNpx(1e6);
  SourceTermElecE3->SetNpx(1e6);
}

double CombinedElectronPositronModel::PosiFluxF(double* x, double* par) {

  return PosiPowerLawComponentF(x, par) + SourceTermPosiF(x, par);
}

double CombinedElectronPositronModel::ElecFluxF(double* x, double* par) {

  return ElecPowerLawComponentF(x, par) + SourceTermElecF(x, par);
}

double CombinedElectronPositronModel::PosiPowerLawComponentF(double* x, double*) {

  return PowerLawSolarMod(x[0], Cplus, gammaPlus, E0, phiPlus, M);
}

double CombinedElectronPositronModel::ElecPowerLawComponentF(double* x, double*) {

  return SmoothlyBrokenPowerLawSolarMod(x[0], Cminus, gammaMinus, deltaGamma, b, invBreakEnergy, E0, phiMinus, M);
}

double CombinedElectronPositronModel::SourceTermPosiF(double* x, double*) {

  return ExpCutoffPowerLawSolarMod(x[0], Csource, gammaSource, lambdaSource, E1, phiPlus, M);
}

double CombinedElectronPositronModel::SourceTermElecF(double* x, double*) {

  return ExpCutoffPowerLawSolarMod(x[0], Csource, gammaSource, lambdaSource, E1, phiMinus, M);
}

double CombinedElectronPositronModel::PosiFluxE3F(double* x, double* par) {

  return std::pow(x[0], 3) * PosiFluxF(x, par);
}

double CombinedElectronPositronModel::ElecFluxE3F(double* x, double* par) {

  return std::pow(x[0], 3) * ElecFluxF(x, par);
}

double CombinedElectronPositronModel::PosiPowerLawComponentE3F(double* x, double* par) {

  return std::pow(x[0], 3) * PosiPowerLawComponentF(x, par);
}

double CombinedElectronPositronModel::ElecPowerLawComponentE3F(double* x, double* par) {

  return std::pow(x[0], 3) * ElecPowerLawComponentF(x, par);
}

double CombinedElectronPositronModel::SourceTermPosiE3F(double* x, double* par) {

  return std::pow(x[0], 3) * SourceTermPosiF(x, par);
}

double CombinedElectronPositronModel::SourceTermElecE3F(double* x, double* par) {

  return std::pow(x[0], 3) * SourceTermElecF(x, par);
}

void FitFluxes(TH1D* electronFlux, TH1D* positronFlux, TF1*& electronModel, TF1*& positronModel) {

  double Estart = 1.0;
  double E0 = 5.0;
  double E1 = 60.0;
  bool useMinosErrors = false;

  // Add 1 percent syst. error
  electronFlux = static_cast<TH1D*>(electronFlux->Clone());
  for (int bin = 1; bin <= electronFlux->GetNbinsX(); ++bin)
    electronFlux->SetBinError(bin, std::sqrt(std::pow(electronFlux->GetBinError(bin) / electronFlux->GetBinContent(bin), 2) + std::pow(0.01, 2)) * electronFlux->GetBinContent(bin));

  positronFlux = static_cast<TH1D*>(positronFlux->Clone());
  for (int bin = 1; bin <= positronFlux->GetNbinsX(); ++bin)
    positronFlux->SetBinError(bin, std::sqrt(std::pow(positronFlux->GetBinError(bin) / positronFlux->GetBinContent(bin), 2) + std::pow(0.01, 2)) * positronFlux->GetBinContent(bin));

  CombinedElectronPositronModel* fullModel = new CombinedElectronPositronModel(E0, E1);

  Modelling::HistogramDataset* elecFlux = new Modelling::HistogramDataset(electronFlux);
  Modelling::HistogramDataset* posiFlux = new Modelling::HistogramDataset(positronFlux);

  Modelling::Data data;
  elecFlux->SetXRange(Estart, 800.0);
  posiFlux->SetXRange(Estart, 800.0);
  data.AddDataset(0, elecFlux);
  data.AddDataset(1, posiFlux);

  Modelling::FitFunction* fitfunction = new Modelling::FitFunction(&data, fullModel);
  Modelling::ModelAnalysis analysis(fitfunction, 2, useMinosErrors);
  analysis.RunAnalysis();
  fullModel->PrintParameters();

  electronModel = static_cast<TF1*>(fullModel->GetPredictionForIdentifier(0)->Clone("electronModel"));
  positronModel = static_cast<TF1*>(fullModel->GetPredictionForIdentifier(1)->Clone("positronModel"));

  TCanvas* modelFitCanvas = Utilities::MakeCanvasWithXYAxes("modelFitCanvas", "E / GeV", "flux / (GeV^{-1} m^{-2} sr^{-1} s^{-1})", 0.5, 1000.0, 1.e-9, 1.e+3);
  electronFlux->DrawCopy("SAME");
  positronFlux->DrawCopy("SAME");

  electronModel->Draw("L SAME");
  positronModel->Draw("L SAME");

  analysis.DrawStatsBox();

  gPad->SetGrid();
  gPad->SetLogx();
  gPad->SetLogy();
  modelFitCanvas->SaveAs(Form("%s.png", modelFitCanvas->GetName()));
}

std::pair<TF1*, TF1*> PredefinedElectronPositronModel() {

  double E0 = 5.0;
  double E1 = 60.0;
  CombinedElectronPositronModel* fullModel = new CombinedElectronPositronModel(E0, E1);

  // Fit parameters from PlotFluxUncertainties.C, averaged over 87 BRs
  fullModel->Cminus.SetValue(6.97091e+00);
  fullModel->gammaMinus.SetValue(3.84573e+00);
  fullModel->deltaGamma.SetValue(4.50068e-01);
  fullModel->b.SetValue(1.90361e-01);
  fullModel->invBreakEnergy.SetValue(4.01666e-02);;
  fullModel->Cplus.SetValue(1.93710e-01);
  fullModel->gammaPlus.SetValue(3.52822e+00);
  fullModel->Csource.SetValue(5.07890e-05);
  fullModel->gammaSource.SetValue(1.99286e+00);
  fullModel->lambdaSource.SetValue(3.76207e-03);
  fullModel->phiMinus.SetValue(1.50453e+00);
  fullModel->phiPlus.SetValue(8.08159e-01);

  auto* electronFlux = static_cast<TF1*>(fullModel->GetPredictionForIdentifier(0)->Clone("electronModel"));
  auto* positronFlux = static_cast<TF1*>(fullModel->GetPredictionForIdentifier(1)->Clone("positronModel"));
  delete fullModel;

  return { electronFlux, positronFlux };
}

