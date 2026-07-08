#include "MeasuringTimeParameterization.hh"

#include <cmath>

double MeasuringTimeParameterizationFunction(double* x, double* par) {

  double xBreak = par[8];
  if (x[0] > xBreak)
    return par[0] * par[9];

  return par[0] * (par[1] * std::pow(x[0], 1.0)
                +  par[2] * std::pow(x[0], 2.0)
                +  par[3] * std::pow(x[0], 3.0)
                +  par[4] * std::pow(x[0], 4.0)
                +  par[5] * std::pow(x[0], 5.0)
                +  par[6] * std::pow(x[0], 6.0)
                +  par[7] * std::pow(x[0], 7.0));

}


