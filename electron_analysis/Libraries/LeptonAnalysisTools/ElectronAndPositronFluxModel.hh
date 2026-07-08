#ifndef ElectronAndPositronFluxModel_hh
#define ElectronAndPositronFluxModel_hh

#include <utility>

#include "Model.hh"
#include "ModelParameter.hh"

class TF1;
class TH1D;

class CombinedElectronPositronModel : public Modelling::Model {
public:
  CombinedElectronPositronModel(double _E0, double _E1);
  virtual ~CombinedElectronPositronModel() { }

  virtual TF1* GetPredictionForIdentifier(int id) const {

    if (id == 0)
      return ElecFlux;
    if (id == 1)
      return PosiFlux;
    return 0;
  }

  double PosiFluxF(double* x, double* par);
  double ElecFluxF(double* x, double* par);

  double PosiPowerLawComponentF(double* x, double*);
  double ElecPowerLawComponentF(double* x, double*);
  double SourceTermPosiF(double* x, double*);
  double SourceTermElecF(double* x, double*);

  double PosiFluxE3F(double* x, double* par);
  double ElecFluxE3F(double* x, double* par);
  double PosiPowerLawComponentE3F(double* x, double*);
  double ElecPowerLawComponentE3F(double* x, double*);
  double SourceTermPosiE3F(double* x, double*);
  double SourceTermElecE3F(double* x, double*);

  TF1* PosiFlux;
  TF1* ElecFlux;

  TF1* PosiPowerLawComponent;
  TF1* ElecPowerLawComponent;
  TF1* SourceTermPosi;
  TF1* SourceTermElec;

  TF1* PosiFluxE3;
  TF1* ElecFluxE3;
  TF1* PosiPowerLawComponentE3;
  TF1* ElecPowerLawComponentE3;
  TF1* SourceTermPosiE3;
  TF1* SourceTermElecE3;

  double E0;
  double E1;
  double M;

  Modelling::ModelParameter Cminus;
  Modelling::ModelParameter gammaMinus;
  Modelling::ModelParameter deltaGamma;
  Modelling::ModelParameter b;
  Modelling::ModelParameter invBreakEnergy;
  Modelling::ModelParameter Cplus;
  Modelling::ModelParameter gammaPlus;
  Modelling::ModelParameter Csource;
  Modelling::ModelParameter gammaSource;
  Modelling::ModelParameter lambdaSource;
  Modelling::ModelParameter phiMinus;
  Modelling::ModelParameter phiPlus;
};

void FitFluxes(TH1D* electronFlux, TH1D* positronFlux, TF1*& electronModel, TF1*& positronModel);
std::pair<TF1*, TF1*> PredefinedElectronPositronModel();

#endif

