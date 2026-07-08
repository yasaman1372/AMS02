#include "IssTree.hh"

#include "AnalysisEvent.hh"
#include "BadRunManager.hh"
#include "CoordinateSystems.hh"

#include "Event.h"
#include "AMSGeometry.h"
#include "Environment.hh"

#include "third_party/FrameTrans.h"

#ifdef HAVE_AMS_SUPPORT
#include "DisableWarnings.h"
#pragma GCC diagnostic ignored "-Wunused-function"
#include "root.h"
#include "EnableWarnings.h"
#endif

#define DEBUG_LEVEL 0
#define INFO_OUT_TAG "IssTree"
#include "debugging.hh"

static double issVelocityInGTOD(const Utilities::CoordinateSystems::ISSParameters& issParameters) {

  double x, y, z;
  double vx, vy, vz;
  Utilities::ThirdParty::FT_Angular2Cart(issParameters.distanceInKM, issParameters.latitudeInRadians, issParameters.longitudeInRadians, x, y, z);
  Utilities::ThirdParty::FT_Angular2Cart(1.0, issParameters.velocityLatitudeInRadians, issParameters.velocityLongitudeInRadians, vx, vy, vz);

  // One orbit every 92.6 minutes -> Speed = (Orbit Circumference / (92.6 * 60s))
  double vECI = issParameters.distanceInKM * 2 * M_PI / (92.6 * 60.0);

  double omega = -2 * M_PI / 86400.0;
  double p = 2 * (vx * omega * y - vy * omega * x);
  double q = (omega * y) * (omega * y) + (omega * x) * (omega * x) - vECI * vECI;

  double vGTOD = -p / 2. + std::sqrt(p * p / 4. - q);
  return vGTOD;
}


IssTree::IssTree()
  : IO::TreeInterface("IssTree", "Tree for the time dependent ISS position and orientation.") {

  RegisterBranches();

}


void IssTree::Fill(const Analysis::Event& event) {

  Time = event.TimeStamp().GetSec();
  UTCTime = event.RawEvent()->EventHeader().UTCTimeStamp().AsDouble();

  // ISS Parameters for coordinate transforms
  const Utilities::CoordinateSystems::ISSParameters issParameters(event.RawEvent()->EventHeader());

  double time = issParameters.utcTimeStamp;
  double posIssR = issParameters.distanceInKM;
  double posIssLong = issParameters.longitudeInRadians;
  double posIssLat = issParameters.latitudeInRadians;
  double posIssX, posIssY, posIssZ;
  Utilities::ThirdParty::FT_Angular2Cart(posIssR, posIssLat, posIssLong, posIssX, posIssY, posIssZ);
  double velIssR = issParameters.velocityInKmPerSecond > 0.0 ? issParameters.velocityInKmPerSecond : issVelocityInGTOD(issParameters);
  double velIssLong = issParameters.velocityLongitudeInRadians;
  double velIssLat = issParameters.velocityLatitudeInRadians;
  double velIssX, velIssY, velIssZ;
  Utilities::ThirdParty::FT_Angular2Cart(velIssR, velIssLat, velIssLong, velIssX, velIssY, velIssZ);
  Utilities::ThirdParty::FT_GTOD_to_TEME_Bugfix(posIssX, posIssY, posIssZ, velIssX, velIssY, velIssZ, time);

  double issYaw = issParameters.yawInRadians;
  double issPitch = issParameters.pitchInRadians;
  double issRoll = issParameters.rollInRadians;

  ISSParameters().push_back(posIssX);
  ISSParameters().push_back(posIssY);
  ISSParameters().push_back(posIssZ);
  ISSParameters().push_back(velIssX);
  ISSParameters().push_back(velIssY);
  ISSParameters().push_back(velIssZ);
  ISSParameters().push_back(issYaw);
  ISSParameters().push_back(issPitch);
  ISSParameters().push_back(issRoll);

  // Trigger

  const AC::Trigger& trigger = event.RawEvent()->Trigger();
  TriggerRateFT = trigger.TriggerRateFT();
  TriggerRateL1 = trigger.TriggerRateLV1();
  TriggerLiveTime = trigger.LiveTime();
}
