#include "EffectiveMeasuringTime.hh"

#include "AnalysisSettings.hh"
#include "Environment.hh"
#include "MeasuringTime.hh"
#include "ObjectManager.hh"

#include <TCanvas.h>
#include <TF1.h>
#include <TH2.h>
#include <TH3.h>
#include <TImage.h>
#include <TLegend.h>
#include <TLine.h>
#include <TPaveText.h>
#include <TProfile2D.h>
#include <TStyle.h>

#include <cassert>
#include <sstream>
#include <fstream>
#include <iomanip>

static const double sSecondsToDays = 60.0 * 60.0 * 24.0;
static const char* sMeasuringTimeDirectory = "MeasuringTime";

EffectiveMeasuringTime::EffectiveMeasuringTime(Utilities::ConfigHandler& config, Utilities::ObjectManager& objectManager,
                                               double cutOffSafetyFactor, RTI::CutOffMode cutOffMode, bool useIGRGCutOff)
  : fTimeTool(new RTI::MeasuringTime(config, objectManager, cutOffSafetyFactor, cutOffMode, useIGRGCutOff, Binning::Predefined::AbsoluteEnergyBinning(), sMeasuringTimeDirectory)) {

}

EffectiveMeasuringTime::~EffectiveMeasuringTime() {

  delete fTimeTool;
}

void EffectiveMeasuringTime::ComputeMeasuringTime() {

  fTimeTool->ComputeMeasuringTime();
}

TH1D* EffectiveMeasuringTime::IntegratedMeasuringTimeOverCutOff() const {

  return fTimeTool->IntegratedMeasuringTimeOverCutOff();
}

void EffectiveMeasuringTime::AnalyzeResults() {

  assert(fTimeTool->MeasuringTimeVsLiveTime());
  double totalMeasuringTime = fTimeTool->MeasuringTimeVsLiveTime()->GetEntries();
  double weightedMeasuringTime = fTimeTool->IntegratedMeasuringTime();
  TH1D* fTimeStatistics = fTimeTool->Statistics();

  double totalSecondsSurvived = fTimeStatistics->GetBinContent(1);
  double secondsRemovedByBadRuns = fTimeStatistics->GetBinContent(2);
  double secondsRemovedByRTI = fTimeStatistics->GetBinContent(3);
  double secondsRemovedByTrdCali = fTimeStatistics->GetBinContent(4);
  double secondsRemovedSum = secondsRemovedByBadRuns + secondsRemovedByRTI + secondsRemovedByTrdCali;
  double totalExposureTimeRuns = gAMSLastEvent - gAMSFirstEvent;
  double totalExposureTime = totalSecondsSurvived + secondsRemovedSum;

  std::stringstream totalExposureTimeRunsInDays;   totalExposureTimeRunsInDays   << std::setw(6) << std::fixed << std::setprecision(2) << totalExposureTimeRuns / sSecondsToDays   << "d";
  std::stringstream totalExposureTimeInDays;       totalExposureTimeInDays       << std::setw(6) << std::fixed << std::setprecision(2) << totalExposureTime / sSecondsToDays       << "d";
  std::stringstream secondsRemovedByBadRunsInDays; secondsRemovedByBadRunsInDays << std::setw(6) << std::fixed << std::setprecision(2) << secondsRemovedByBadRuns / sSecondsToDays << "d";
  std::stringstream secondsRemovedByRTIInDays;     secondsRemovedByRTIInDays     << std::setw(6) << std::fixed << std::setprecision(2) << secondsRemovedByRTI / sSecondsToDays     << "d";
  std::stringstream secondsRemovedByTrdCaliInDays; secondsRemovedByTrdCaliInDays << std::setw(6) << std::fixed << std::setprecision(2) << secondsRemovedByTrdCali / sSecondsToDays << "d";
  std::stringstream secondsRemovedSumInDays;       secondsRemovedSumInDays       << std::setw(6) << std::fixed << std::setprecision(2) << secondsRemovedSum / sSecondsToDays       << "d";
  std::stringstream totalMeasuringTimeInDays;      totalMeasuringTimeInDays      << std::setw(6) << std::fixed << std::setprecision(2) << totalMeasuringTime / sSecondsToDays      << "d";
  std::stringstream weightedMeasuringTimeInDays;   weightedMeasuringTimeInDays   << std::setw(6) << std::fixed << std::setprecision(2) << weightedMeasuringTime / sSecondsToDays   << "d";

  std::stringstream rtiEfficiency;
  rtiEfficiency << std::setw(6) << std::fixed << std::setprecision(2) << (totalMeasuringTime / totalExposureTime * 100.0) << " %";

  std::stringstream effectiveLiveness;
  effectiveLiveness << std::setw(6) << std::fixed << std::setprecision(2) << (weightedMeasuringTime / totalMeasuringTime * 100.0) << " %";

  std::cout << "####### MEASURING TIME SUMMARY #######" << std::endl;
  std::cout << "Total exposure time from run numbers: " << std::setw(12) << totalExposureTimeRuns   << "s  <->  " << totalExposureTimeRunsInDays.str()   << std::endl;
  std::cout << "Total exposure time                 : " << std::setw(12) << totalExposureTime       << "s  <->  " << totalExposureTimeInDays.str()       << std::endl;
  std::cout << std::endl;

  std::cout << "Total bad times removed by bad runs : " << std::setw(12) << secondsRemovedByBadRuns << "s  <->  " << secondsRemovedByBadRunsInDays.str() << std::endl;
  std::cout << "Total bad times removed by RTI      : " << std::setw(12) << secondsRemovedByRTI     << "s  <->  " << secondsRemovedByRTIInDays.str()     << std::endl;
  std::cout << "Total bad times removed by TRD cali.: " << std::setw(12) << secondsRemovedByTrdCali << "s  <->  " << secondsRemovedByTrdCaliInDays.str() << std::endl;
  std::cout << " -> Excluding a total of            : " << std::setw(12) << secondsRemovedSum       << "s  <->  " << secondsRemovedSumInDays.str()       << std::endl;
  std::cout << std::endl;

  std::cout << "Total measuring time                : " << std::setw(12) << totalMeasuringTime      << "s  <->  " << totalMeasuringTimeInDays.str()      << std::endl;
  std::cout << "  -> Weighted for live-time         : " << std::setw(12) << weightedMeasuringTime   << "s  <->  " << weightedMeasuringTimeInDays.str()   << std::endl;
  std::cout << std::endl;

  std::cout << "RTI efficiency                      : " << std::setw(12) << rtiEfficiency.str()     << std::endl;
  std::cout << "Effective liveness                  : " << std::setw(12) << effectiveLiveness.str() << std::endl;

  assert(totalSecondsSurvived == totalMeasuringTime); // This is guaranteed to by in-sync by TimeTools. Just here for sanity.
}

TCanvas* EffectiveMeasuringTime::CreateLiveTimeDistribution(Utilities::ObjectManager& objectManager) const {

  TCanvas* canvasLiveTimeDistribution = dynamic_cast<TCanvas*>(objectManager.Add(new TCanvas("canvasLiveTimeDistribution", "Measuring time: Live-time distribution"), sMeasuringTimeDirectory));
  canvasLiveTimeDistribution->cd();
  gPad->SetLogy();
  gPad->SetLeftMargin(0.13);
  gPad->SetTopMargin(0.03);
  gPad->SetBottomMargin(0.14);

  TH1D* fMeasuringTimeVsLiveTime = fTimeTool->MeasuringTimeVsLiveTime();
  fMeasuringTimeVsLiveTime->UseCurrentStyle();
  fMeasuringTimeVsLiveTime->GetXaxis()->SetRangeUser(0.5, 1);
  fMeasuringTimeVsLiveTime->SetLineColor(kRed);
  fMeasuringTimeVsLiveTime->SetFillColor(kGray + 2);
  fMeasuringTimeVsLiveTime->SetFillStyle(3004);
  fMeasuringTimeVsLiveTime->SetTitle("AMS-02 detector live-time");
  fMeasuringTimeVsLiveTime->GetXaxis()->SetTitle("Live-time fraction");
  fMeasuringTimeVsLiveTime->GetYaxis()->SetTitle("Seconds");
  fMeasuringTimeVsLiveTime->Draw("hist");

  std::stringstream liveTimeLabelText;
  double weightedMeasuringTime = fTimeTool->IntegratedMeasuringTime();
  std::stringstream weightedMeasuringTimeInDays;
  weightedMeasuringTimeInDays << std::setw(6) << std::fixed << std::setprecision(2) << weightedMeasuringTime / sSecondsToDays << "d";
  liveTimeLabelText << std::setw(12) << weightedMeasuringTime << "s  #Leftrightarrow  " << weightedMeasuringTimeInDays.str() << std::endl;

  TPaveText* periodLabel = new TPaveText(0.153494,0.789887,0.5470683,0.9466667,"brNDC");
  periodLabel->SetFillColor(kWhite);
  periodLabel->SetFillStyle(4050);
  periodLabel->SetBorderSize(0);
  periodLabel->SetTextAlign(11);
  periodLabel->SetTextSize(0.06);
  periodLabel->AddText(gAMSDataPeriod);
  periodLabel->Draw();

  TPaveText* liveTimeLabel = new TPaveText(0.1764368,0.2478814,0.8787356,0.4449153,"brNDC");
  liveTimeLabel->SetFillColor(kWhite);
  liveTimeLabel->SetBorderSize(1);
  liveTimeLabel->SetTextAlign(21);
  liveTimeLabel->SetTextSize(0.05);
  liveTimeLabel->AddText("Measuring time weighted with live-time fraction:");
  liveTimeLabel->AddText(liveTimeLabelText.str().c_str());
  liveTimeLabel->Draw();

  return canvasLiveTimeDistribution;
}

TCanvas* EffectiveMeasuringTime::CreateMeasuringTimeDecomposition(Utilities::ObjectManager& objectManager) const {

  TH1D* hEffectiveMeasuringTime = static_cast<TH1D*>(fTimeTool->IntegratedMeasuringTimeOverCutOff()->Clone());
  for (int bin = 1; bin <= hEffectiveMeasuringTime->GetNbinsX(); ++bin) {
    hEffectiveMeasuringTime->SetBinContent(bin, hEffectiveMeasuringTime->GetBinContent(bin) / sSecondsToDays);
    hEffectiveMeasuringTime->SetBinError(bin, hEffectiveMeasuringTime->GetBinError(bin) / sSecondsToDays);
  }

  TCanvas* canvasMeasuringTimeDecomposition = dynamic_cast<TCanvas*>(objectManager.Add(new TCanvas("canvasMeasuringTimeDecomposition", "Measuring time: decomposition"), sMeasuringTimeDirectory));
  canvasMeasuringTimeDecomposition->cd();
  gPad->SetLogx();
  gPad->SetLeftMargin(0.13);
  gPad->SetTopMargin(0.03);
  gPad->SetBottomMargin(0.14);

  auto* hMeasuringTimeVsLiveTime = fTimeTool->MeasuringTimeVsLiveTime();
  double totalMeasuringTime = hMeasuringTimeVsLiveTime->GetEntries() / sSecondsToDays;
  double weightedMeasuringTime = 0.0;
  for (int i = 1; i <= hMeasuringTimeVsLiveTime->GetNbinsX(); ++i)
    weightedMeasuringTime += hMeasuringTimeVsLiveTime->GetBinContent(i) * hMeasuringTimeVsLiveTime->GetBinCenter(i);
  weightedMeasuringTime /= sSecondsToDays;

  auto* hTimeStatistics = fTimeTool->Statistics();
  double totalDaysSurvived = hTimeStatistics->GetBinContent(1) / sSecondsToDays;
  double daysRemovedByBadRuns = hTimeStatistics->GetBinContent(2) / sSecondsToDays;
  double daysRemovedByRTI = hTimeStatistics->GetBinContent(3) / sSecondsToDays;
  double daysRemovedByTrdCali = hTimeStatistics->GetBinContent(4) / sSecondsToDays;
  double totalExposureTime = (totalDaysSurvived + daysRemovedByBadRuns + daysRemovedByRTI + daysRemovedByTrdCali);

  auto* hEnergyFrame = DrawEnergyFrame(gAMSStartShowEnergy, gAMSStopShowEnergy, 0, totalExposureTime * 1.05, "ECAL Energy / GeV", "Measuring time / days");
  hEnergyFrame->GetYaxis()->SetNoExponent(kTRUE);
  hEnergyFrame->GetXaxis()->SetLabelOffset(hEnergyFrame->GetXaxis()->GetLabelOffset() + 0.01);

  hEffectiveMeasuringTime->SetLineWidth(4);
  hEffectiveMeasuringTime->SetLineColor(kRed);
  hEffectiveMeasuringTime->SetFillColor(kGray + 2);
  hEffectiveMeasuringTime->SetFillStyle(3004);
  hEffectiveMeasuringTime->Draw("hist.same");

  std::stringstream rtiEfficiency;
  rtiEfficiency << std::setw(6) << std::fixed << std::setprecision(2) << (totalMeasuringTime / totalExposureTime * 100.0) << " %";

  std::stringstream effectiveLiveness;
  effectiveLiveness << std::setw(6) << std::fixed << std::setprecision(2) << (weightedMeasuringTime / totalMeasuringTime * 100.0) << " %";

  TPaveText* efficiencyLabel1 = new TPaveText(0, totalMeasuringTime, 60, totalExposureTime);
  efficiencyLabel1->SetFillColor(kWhite);
  efficiencyLabel1->SetFillStyle(4050);
  efficiencyLabel1->SetBorderSize(0);
  efficiencyLabel1->SetTextAlign(12);
  efficiencyLabel1->SetTextFont(62);
  efficiencyLabel1->AddText("#Downarrow Detector quality cuts:");
  efficiencyLabel1->Draw();

  TPaveText* efficiencyText1 = new TPaveText(60, totalMeasuringTime, gAMSStopShowEnergy, totalExposureTime);
  efficiencyText1->SetFillColor(kWhite);
  efficiencyText1->SetFillStyle(4050);
  efficiencyText1->SetBorderSize(0);
  efficiencyText1->SetTextAlign(12);
  efficiencyText1->SetTextFont(62);
  efficiencyText1->AddText(rtiEfficiency.str().c_str());
  efficiencyText1->Draw();

  TPaveText* efficiencyLabel2 = new TPaveText(0, weightedMeasuringTime, 60, totalMeasuringTime);
  efficiencyLabel2->SetFillColor(kWhite);
  efficiencyLabel2->SetFillStyle(4050);
  efficiencyLabel2->SetBorderSize(0);
  efficiencyLabel2->SetTextAlign(12);
  efficiencyLabel2->SetTextFont(62);
  efficiencyLabel2->AddText("#Downarrow Live-time weighting:");
  efficiencyLabel2->Draw();

  TPaveText* efficiencyText2 = new TPaveText(60, weightedMeasuringTime, gAMSStopShowEnergy, totalMeasuringTime);
  efficiencyText2->SetFillColor(kWhite);
  efficiencyText2->SetFillStyle(4050);
  efficiencyText2->SetBorderSize(0);
  efficiencyText2->SetTextAlign(12);
  efficiencyText2->SetTextFont(62);
  efficiencyText2->AddText(effectiveLiveness.str().c_str());
  efficiencyText2->Draw();

  TLine* totalExposureTimeMarker = new TLine(0, totalExposureTime, gAMSStopShowEnergy, totalExposureTime);
  totalExposureTimeMarker->SetLineColor(kBlue);
  totalExposureTimeMarker->SetLineWidth(3);
  totalExposureTimeMarker->SetLineStyle(2);
  totalExposureTimeMarker->Draw();

  TLine* totalMeasuringTimeMarker = new TLine(0, totalMeasuringTime, gAMSStopShowEnergy, totalMeasuringTime);
  totalMeasuringTimeMarker->SetLineColor(kGreen + 2);
  totalMeasuringTimeMarker->SetLineWidth(3);
  totalMeasuringTimeMarker->SetLineStyle(2);
  totalMeasuringTimeMarker->Draw();
  
  TLine* weightedMeasuringTimeMarker = new TLine(0, weightedMeasuringTime, gAMSStopShowEnergy, weightedMeasuringTime);
  weightedMeasuringTimeMarker->SetLineColor(kViolet);
  weightedMeasuringTimeMarker->SetLineWidth(3);
  weightedMeasuringTimeMarker->SetLineStyle(2);
  weightedMeasuringTimeMarker->Draw();

  return canvasMeasuringTimeDecomposition;
}

TCanvas* EffectiveMeasuringTime::CreateRtiControlPlots(Utilities::ObjectManager& objectManager) const {

  // RTI cut 'CutBadReconstructionPeriod' control plot
  TCanvas* canvasRTIParticlesVsTrigger = dynamic_cast<TCanvas*>(objectManager.Add(new TCanvas("canvasRTIParticlesVsTrigger", "RTI control plot for 'CutBadReconstructionPeriod'"), sMeasuringTimeDirectory));
  canvasRTIParticlesVsTrigger->cd();
  gPad->SetLogz();
  gPad->SetLeftMargin(0.13);
  gPad->SetRightMargin(0.18);
  gPad->SetTopMargin(0.03);
  gPad->SetBottomMargin(0.14);

  TH2D* fParticlesVsTriggers = fTimeTool->ParticlesVsTriggers();
  fParticlesVsTriggers->UseCurrentStyle();
  fParticlesVsTriggers->GetXaxis()->SetNoExponent(kTRUE);
  fParticlesVsTriggers->GetXaxis()->SetRangeUser(0, 0.25);

  fParticlesVsTriggers->GetXaxis()->SetTitle("Number of particles / trigger rate");
  fParticlesVsTriggers->GetXaxis()->SetLabelOffset(fParticlesVsTriggers->GetXaxis()->GetLabelOffset() + 0.01);

  fParticlesVsTriggers->GetYaxis()->SetTitle("Trigger rate");
  fParticlesVsTriggers->GetYaxis()->SetNoExponent(kTRUE);

  fParticlesVsTriggers->GetZaxis()->SetTitle("Entries");
  fParticlesVsTriggers->GetZaxis()->SetLabelSize(fParticlesVsTriggers->GetYaxis()->GetLabelSize());
  fParticlesVsTriggers->GetZaxis()->SetLabelOffset(fParticlesVsTriggers->GetYaxis()->GetLabelOffset());
  fParticlesVsTriggers->GetZaxis()->SetTitleSize(fParticlesVsTriggers->GetYaxis()->GetTitleSize());
  fParticlesVsTriggers->GetZaxis()->SetTitleOffset(fParticlesVsTriggers->GetYaxis()->GetTitleOffset());

  fParticlesVsTriggers->Draw("col.z");

  TF1* fParticlesVsTriggersCut = new TF1("fParticlesVsTriggersCut", "x * 1600 / 0.07", 0, 0.0888);
  fParticlesVsTriggersCut->SetNpx(100);
  fParticlesVsTriggersCut->SetLineColor(kRed);
  fParticlesVsTriggersCut->SetLineStyle(2);
  fParticlesVsTriggersCut->SetLineWidth(3);
  fParticlesVsTriggersCut->Draw("same");

  return canvasRTIParticlesVsTrigger;
}

TCanvas* EffectiveMeasuringTime::CreateTriggerRateVsISSPositionPlots(Utilities::ObjectManager& objectManager) const {

  TCanvas* canvasTriggerRateVsISSPosition = dynamic_cast<TCanvas*>(objectManager.Add(new TCanvas("canvasTriggerRateVsISSPosition", "Trigger rate vs. ISS position", gStyle->GetCanvasDefW() * 2, gStyle->GetCanvasDefH() * 2), sMeasuringTimeDirectory));
  canvasTriggerRateVsISSPosition->cd();
  SetupCanvasWithEarthImageBackground();
  gPad->SetLogz();

  TH3F* fTriggerRateVsISSPosition = fTimeTool->TriggerRateVsISSPosition();
  fTriggerRateVsISSPosition->UseCurrentStyle();

  TProfile2D* triggerRateVsISSPositionProjection = dynamic_cast<TProfile2D*>(fTriggerRateVsISSPosition->Project3DProfile("yx"));
  StyleISSPositionPlotAxes(fTriggerRateVsISSPosition, triggerRateVsISSPositionProjection);
  triggerRateVsISSPositionProjection->GetZaxis()->SetRangeUser(50, 3e3);
  triggerRateVsISSPositionProjection->Draw("col.z");

  return canvasTriggerRateVsISSPosition;
}

TCanvas* EffectiveMeasuringTime::CreateLiveTimeVsISSPositionPlots(Utilities::ObjectManager& objectManager) const {

  TCanvas* canvasLiveTimeVsISSPosition = dynamic_cast<TCanvas*>(objectManager.Add(new TCanvas("canvasLiveTimeVsISSPosition", "Live-time vs. ISS position", gStyle->GetCanvasDefW() * 2, gStyle->GetCanvasDefH() * 2), sMeasuringTimeDirectory));
  canvasLiveTimeVsISSPosition->cd();
  SetupCanvasWithEarthImageBackground();

  TH3F* fLiveTimeVsISSPosition = fTimeTool->LiveTimeVsISSPosition();
  fLiveTimeVsISSPosition->UseCurrentStyle();

  TProfile2D* liveTimeVsISSPositionProjection = dynamic_cast<TProfile2D*>(fLiveTimeVsISSPosition->Project3DProfile("yx"));
  StyleISSPositionPlotAxes(fLiveTimeVsISSPosition, liveTimeVsISSPositionProjection);
  liveTimeVsISSPositionProjection->GetZaxis()->SetRangeUser(0, 1);
  liveTimeVsISSPositionProjection->GetZaxis()->SetNoExponent(kTRUE);
  liveTimeVsISSPositionProjection->GetZaxis()->SetTitle("Live-time fraction");
  liveTimeVsISSPositionProjection->Draw("col.z");

  return canvasLiveTimeVsISSPosition;
}

TCanvas* EffectiveMeasuringTime::CreateCutOffRigidityVsISSPositionPlots(Utilities::ObjectManager& objectManager) const {

  TCanvas* canvasCutOffRigidityVsISSPosition = dynamic_cast<TCanvas*>(objectManager.Add(new TCanvas("canvasCutOffRigidityVsISSPosition", "Cut-off rigidity vs. ISS position", gStyle->GetCanvasDefW() * 2, gStyle->GetCanvasDefH() * 2), sMeasuringTimeDirectory));
  canvasCutOffRigidityVsISSPosition->cd();
  SetupCanvasWithEarthImageBackground();

  TH3F* fCutOffRigidityVsISSPosition = fTimeTool->CutOffRigidityVsISSPosition();
  fCutOffRigidityVsISSPosition->UseCurrentStyle();

  TProfile2D* cutOffRigidityVsISSPositionProjection = dynamic_cast<TProfile2D*>(fCutOffRigidityVsISSPosition->Project3DProfile("yx"));
  StyleISSPositionPlotAxes(fCutOffRigidityVsISSPosition, cutOffRigidityVsISSPositionProjection);
  cutOffRigidityVsISSPositionProjection->Draw("col.z");

  return canvasCutOffRigidityVsISSPosition;
}

void EffectiveMeasuringTime::StyleISSPositionPlotAxes(TH3F* histogram, TProfile2D* profilePlot) const {

  profilePlot->GetXaxis()->SetTitle(histogram->GetXaxis()->GetTitle());
  profilePlot->GetXaxis()->SetNoExponent(kTRUE);
  profilePlot->GetXaxis()->SetLabelOffset(profilePlot->GetXaxis()->GetLabelOffset() + 0.01);

  profilePlot->GetYaxis()->SetTitle(histogram->GetYaxis()->GetTitle());
  profilePlot->GetYaxis()->SetNoExponent(kTRUE);

  profilePlot->GetZaxis()->SetTitle(histogram->GetZaxis()->GetTitle());
  profilePlot->GetZaxis()->SetLabelSize(profilePlot->GetYaxis()->GetLabelSize());
  profilePlot->GetZaxis()->SetLabelOffset(profilePlot->GetYaxis()->GetLabelOffset());
  profilePlot->GetZaxis()->SetTitleSize(profilePlot->GetYaxis()->GetTitleSize());
  profilePlot->GetZaxis()->SetTitleOffset(profilePlot->GetYaxis()->GetTitleOffset());
}

void EffectiveMeasuringTime::SetupCanvasWithEarthImageBackground() const {

  std::string earthImageFile = "${LEPTONANALYSIS}/LookupFiles/earth.png";
  Environment::ExpandEnvironmentVariables(earthImageFile);
  auto* earthImage = TImage::Open(earthImageFile.c_str());
  assert(earthImage);
  earthImage->Draw("X");

  gPad->SetLeftMargin(0.13);
  gPad->SetRightMargin(0.18);
  gPad->SetTopMargin(0.03);
  gPad->SetBottomMargin(0.14);

  TPad* pad = new TPad("extraPad", "", 0, 0, 1, 1);
  pad->SetFillStyle(4000);
  pad->SetFrameFillStyle(4000);
  pad->Draw();
  pad->cd();

  gPad->SetLeftMargin(0.13);
  gPad->SetRightMargin(0.18);
  gPad->SetTopMargin(0.03);
  gPad->SetBottomMargin(0.14);
}

