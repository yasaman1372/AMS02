#ifndef IssTree_hh
#define IssTree_hh

#include "TreeInterface.hh"

class IssTree : public IO::TreeInterface {
public:
  IssTree();

  IO::TreeBranch<unsigned int> Time { "Time", 0 };
  IO::TreeBranch<double> UTCTime { "UTCTime", 0 };
  IO::TreeBranch<float> TriggerRateFT { "TriggerRateFT", -100.0f };
  IO::TreeBranch<float> TriggerRateL1 { "TriggerRateL1", -100.0f };
  IO::TreeBranch<float> TriggerLiveTime { "TriggerLiveTime", 0.0f };
  IO::TreeBranch<std::vector<float>> ISSParameters { "ISSParameters", IO::TreeVectorSize(9) };

private:
  virtual void Fill(const Analysis::Event&);
  virtual IO::TreeInterface* Create() const { return new IssTree; }

};

#endif
