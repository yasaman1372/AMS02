#ifndef LeptonTreePreselection_hh
#define LeptonTreePreselection_hh

namespace Analysis {
  class Event;
  class EventFactory;
  class Particle;
}

namespace Cuts {
  class Selector;
}

namespace IO {
  class FileManager;
}

namespace Utilities {
  class ConfigHandler;
  class ObjectManager;
}

class LeptonTreePreselection {
public:
  LeptonTreePreselection(Utilities::ConfigHandler&, Utilities::ObjectManager& inputObjectManager, IO::FileManager* fileManager);

  // Only called when processing ACQt files to execute the preselection.
  bool ProcessEvent(Analysis::EventFactory&, Analysis::Event&);

  // Print summary of cut statistics for each selector.
  void PrintSummary() const;

private:
  void ApplyEventPreProcessing(Analysis::Event&);
  bool ShouldStoreISSEventWithoutEcalShower(const Analysis::Particle*);
  void PerformTrdTracking(Analysis::EventFactory&, Analysis::Event&);

  IO::FileManager* fFileManager;
  Cuts::Selector* fBadRunsSelector;
  Cuts::Selector* fRtiSelector;
  Cuts::Selector* fTrdCalibrationSelector;
  Cuts::Selector* fMcPreselectionSelector;
};

#endif // LeptonTreePreselection_hh

