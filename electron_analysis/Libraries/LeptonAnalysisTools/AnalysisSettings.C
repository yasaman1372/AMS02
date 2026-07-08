#include "AnalysisSettings.hh"

#include "BinningTools.hh"
#include "Cut.hh"
#include "EfficiencyHistograms.hh"
#include "EfficiencyVsTimeHistograms.hh"
#include "FileManagerController.hh"
#include "Palettes.hh"
#include "PredefinedBinnings.hh"

#include <cmath>
#include <cassert>

#include <TCanvas.h>
#include <TColor.h>
#include <TGaxis.h>
#include <TH1.h>
#include <TPaveText.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TSystem.h>

const long gAMSFirstEvent = 1305853510; // Time of first event in first run 1305853512 - 1s.
const long gAMSLastEvent = 1510523907; // Time of the last event in the last run 1510523906 + 1s.
const long gAMSLastEventPass4 = 1385484729; // Time of the last event in the last run 1385484728 + 1s.
const char* gAMSDataPeriod = "May 20^{th}, 2011 - November 12^{th}, 2017";

const double gAMSMinEnergy = 0.25;
const double gAMSMaxEnergy = 1500.0;

const double gAMSStartShowEnergy = 0.5;
const double gAMSStopShowEnergy = 1100.0;

const double gAMSTimeDependentStartShowEnergy = 1.01;
const double gAMSTimeDependentStopShowEnergy = 49.33;

const double gAMSTimeDependentFluxesMaxEnergy = 52.33 - 0.01; // before up edge of EnergyBinning[  54] =       [49.33, 52.33]: xCenter=   50.83 xTrue= 50.8005
const double gAMSTimeDependentFluxesMinEnergy =  0.65 + 0.01; // after low edge of EnergyBinning[   3] =         [0.65, 0.82]: xCenter=   0.735 xTrue=0.728432
const double gAMSLaffertyWyattSpectralIndex = 3.0;

const int gTrackerPatternColors[gTrackerPattern] = { kBlue, kRed, kGreen + 2, kMagenta, kOrange + 2, kBlack };

const int gElectronColor = kBlue + 2;
const int gCCElectronColor = kBlue - 7;

const int gPositronColor = kCyan + 2;
const int gCCPositronColor = kCyan - 7;

const int gProtonColor = kRed + 2;
const int gCCProtonColor = kRed - 7;

const int gCCColor = kMagenta;
const int gFitResultColor = kOrange - 3;

int TransparentColor(int color, float opacity) {

  return TColor::GetColorTransparent(color, opacity);
}

void SaveCanvas(TCanvas* canvas, std::string fileName) {

  assert(canvas);
  canvas->Update();
  fileName = "Plots/" + fileName;
  canvas->SaveAs(fileName.c_str());
}

void MoveTitleBoxToCenter() {

  gPad->Update();

  TPaveText* titleBox = dynamic_cast<TPaveText*>(gPad->FindObject("title"));
  if (!titleBox)
    return;

  assert(titleBox);
  titleBox->SetX1NDC(0.1546047);
  titleBox->SetY1NDC(0.9314814);
  titleBox->SetX2NDC(0.8554359);
  titleBox->SetY2NDC(0.9937892);
}

// Lepton Flux Binning as agreed on in May 2014
static const Binning::Definition& LeptonFluxBinning_May2014() {

  static const Binning::Definition sBinning(Binning::FromVector({
      0.25, 0.50, 0.65, 0.82, 1.01, 1.22,
      1.46, 1.72, 2.00, 2.31, 2.65,
      3.00, 3.36, 3.73, 4.12, 4.54,
      5.00, 5.49, 6.00, 6.54, 7.10,
      7.69, 8.30, 8.95, 9.62, 10.32,
      11.04, 11.80, 12.59, 13.41, 14.25,
      15.14, 16.05, 17.00, 17.98, 18.99,
      20.04, 21.13, 22.25, 23.42, 24.62,
      25.90, 27.25, 28.68, 30.21, 31.82,
      33.53, 35.36, 37.31, 39.39, 41.61,
      44.00, 46.57, 49.33, 52.33, 55.58,
      59.13, 63.02, 67.30, 72.05, 77.37,
      83.36, 90.19, 98.08, 107.34, 118.42,
      132.11, 148.81, 169.86, 197.69, 237.16,
      290.0, 370.0, 500.0, 700.0, 1000.0,
      1500.0
  }));

  return sBinning;
}

void AnalysisSettings::Initialize() {

  Cuts::EfficiencyHistograms::sEnableTagAndProbeControlHistograms = true;
  Cuts::EfficiencyVsTimeHistograms::sActivationState = Cuts::CutAttachment::ActivationMode::Selective;

  IO::FileManagerController::Self()->SetFirstAndLastEventTimes(gAMSFirstEvent, gAMSLastEvent);
  gSystem->Setenv("LC_ALL", "en_US");

  // Central place to define binning for the whole lepton analysis.
  Binning::Predefined::SetDefaultAbsoluteEnergyBinning(LeptonFluxBinning_May2014());
  TH1::AddDirectory(kFALSE);

  gROOT->SetStyle("amspub");
  gStyle->SetOptFit(0);
  gStyle->SetLineWidth(1);

  // for nice pdfs
  gStyle->SetCanvasDefW(gStyle->GetCanvasDefW() / 2);
  gStyle->SetCanvasDefH(gStyle->GetCanvasDefH() / 2);
  gStyle->SetMarkerSize(1.0);
  gROOT->ForceStyle();

  static const Int_t NRGBs = 5;
  static const Int_t NCont = 255;
  static Double_t stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
  static Double_t red[NRGBs]   = { 0.00, 0.00, 0.87, 1.00, 0.51 };
  static Double_t green[NRGBs] = { 0.00, 0.81, 1.00, 0.20, 0.00 };
  static Double_t blue[NRGBs]  = { 0.51, 1.00, 0.12, 0.00, 0.00 };
  TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
  gStyle->SetNumberContours(NCont);
}

void AnalysisSettings::SwitchToQuadraticCanvas() {

  gStyle->SetCanvasDefW(1000);
  gStyle->SetCanvasDefH(1000);

  gStyle->SetPadLeftMargin(0.09);
  gStyle->SetPadRightMargin(0.037);
  gStyle->SetPadTopMargin(0.04);
  gStyle->SetPadBottomMargin(0.1);
}

void AnalysisSettings::SwitchToRectangularCanvas() {

  gStyle->SetCanvasDefW(1000);
  gStyle->SetCanvasDefH(750);

  gStyle->SetPadLeftMargin(0.075);
  gStyle->SetPadRightMargin(0.02);
  gStyle->SetPadTopMargin(0.05);
  gStyle->SetPadBottomMargin(0.1);
}

bool IsBlacklistedBartelsRotation(unsigned int bartelsRotation) {

  return bartelsRotation == 46 || bartelsRotation == 47;
}

bool IsBlacklistedBartelsRotation(const std::string& inputFileSuffix) {

  return inputFileSuffix == "Bartels_46" || inputFileSuffix == "Bartels_47";
}

int ClassifyEnergyIntoBin(double energy, const Binning::Definition& binning) {

  unsigned int energyBin = binning.FindBin(energy);
  if (energyBin == 0 || energyBin == binning.NumberOfBins() + 1)
    return -1;

  assert(energyBin >= 1);
  assert(energyBin <= binning.NumberOfBins());
  return energyBin - 1;
}

int ClassifyEnergyIntoBin(double energy) {

  return ClassifyEnergyIntoBin(energy, Binning::Predefined::AbsoluteEnergyBinning());
}

TH1* DrawEnergyFrame(double xMin, double xMax, double yMin, double yMax, const std::string& xAxisTitle, const std::string& yAxisTitle) {

  std::vector<double> originalBinEdges = Binning::SubRange(Binning::Predefined::AbsoluteEnergyBinning(), xMin, xMax, false).Bins();
  if (xMax == gAMSStopShowEnergy)
    originalBinEdges.emplace_back(gAMSStopShowEnergy);
  auto alignedSubBinning  = Binning::FromVector(originalBinEdges);

  auto* hEnergyFrame = Make<TH1D>("hEnergyFrame", "", alignedSubBinning);
  hEnergyFrame->GetXaxis()->SetTitle(xAxisTitle.c_str());
  hEnergyFrame->GetXaxis()->SetNoExponent(kTRUE);
  hEnergyFrame->GetXaxis()->SetMoreLogLabels(kTRUE);

  hEnergyFrame->GetYaxis()->SetTitle(yAxisTitle.c_str());
  hEnergyFrame->GetYaxis()->SetRangeUser(yMin, yMax);
  hEnergyFrame->SetStats(0);
  hEnergyFrame->Draw();
  return hEnergyFrame;
}

std::string MigrationMatrixElectronAllTracksName(bool useMITEnergyScaleCorrection) {

  if (useMITEnergyScaleCorrection)
    return "fMigrationMatrixMITCorrectedWeightedByTrueElectronFlux_Electrons";
  return "fMigrationMatrixWeightedByTrueElectronFlux_Electrons";
}

std::string MigrationMatrixElectronSingleTrackName(bool useMITEnergyScaleCorrection) {

  if (useMITEnergyScaleCorrection)
    return "fMigrationMatrixMITCorrectedWeightedByTrueElectronFlux_SingleTrack_Electrons";
  return "fMigrationMatrixWeightedByTrueElectronFlux_Electrons";
}

std::string MigrationMatrixPositronAllTracksName(bool useMITEnergyScaleCorrection) {

  if (useMITEnergyScaleCorrection)
    return "fMigrationMatrixMITCorrectedWeightedByTruePositronFlux_Electrons";
  return "fMigrationMatrixWeightedByTruePositronFlux_Electrons";
}

std::string MigrationMatrixPositronSingleTrackName(bool useMITEnergyScaleCorrection) {

  if (useMITEnergyScaleCorrection)
    return "fMigrationMatrixMITCorrectedWeightedByTruePositronFlux_SingleTrack_Electrons";
  return "fMigrationMatrixWeightedByTruePositronFlux_SingleTrack_Electrons";
}

