#ifndef EffectiveMeasuringTime_hh
#define EffectiveMeasuringTime_hh

#include "CutOffMode.hh"

namespace RTI {
  class MeasuringTime;
}

namespace Utilities {
  class ConfigHandler;
  class ObjectManager;
}

class TCanvas;
class TH1D;
class TH3F;
class TImage;
class TProfile2D;

class EffectiveMeasuringTime {
public:
  EffectiveMeasuringTime(Utilities::ConfigHandler&, Utilities::ObjectManager&,
                         double cutOffSafetyFactor, RTI::CutOffMode, bool useIGRGCutOff);
  ~EffectiveMeasuringTime();

  void ComputeMeasuringTime();
  void AnalyzeResults();

  TCanvas* CreateMeasuringTimeDecomposition(Utilities::ObjectManager&) const;
  TCanvas* CreateLiveTimeDistribution(Utilities::ObjectManager&) const;
  TCanvas* CreateRtiControlPlots(Utilities::ObjectManager&) const;
  TCanvas* CreateTriggerRateVsISSPositionPlots(Utilities::ObjectManager&) const;
  TCanvas* CreateLiveTimeVsISSPositionPlots(Utilities::ObjectManager&) const;
  TCanvas* CreateCutOffRigidityVsISSPositionPlots(Utilities::ObjectManager&) const;

  TH1D* IntegratedMeasuringTimeOverCutOff() const;

private:
  void SetupCanvasWithEarthImageBackground() const;
  void StyleISSPositionPlotAxes(TH3F*, TProfile2D*) const;

  RTI::MeasuringTime* fTimeTool;
};

#endif


