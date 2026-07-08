#ifndef ExampleAnalysisTreeReader_hh
#define ExampleAnalysisTreeReader_hh

#include "TreeReader.hh"

class ExampleAnalysisTree;
class TH1;

namespace Cuts {
class Selector;
}

namespace IO {
class TreeWriter;
}

class ExampleAnalysisTreeReader : public IO::TreeReader {
public:
  ExampleAnalysisTreeReader();
  virtual ~ExampleAnalysisTreeReader();

private:
  ClassDef(ExampleAnalysisTreeReader, 0)

  virtual IO::TreeInterface* Initialize(Utilities::ObjectManager&);
  virtual void ParseOptions(Utilities::ConfigHandler&);
  virtual void TreeChanged(TTree*);
  virtual bool ProcessEvent();
  virtual void Finish();

private:
  ExampleAnalysisTree* fTree;                    //! Interface to our tree
  TH1* fEcalEnergyElectronHistogram;             //! Example histogram
  TH1* fTrackerChargeLayer9Histogram;            //! Example histogram
  TH1* fEoverAbsRHistogram;                      //! Example histogram
  IO::TreeWriter* fTreeAfterSelectionWriter;     //! Tree writer to write reduced tree
  Cuts::Selector* fSelector;                     //! Selector for 'selection cuts' based on config file
};

#endif
