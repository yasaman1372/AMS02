#!/usr/bin/env python3

import json
import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import uncertainties

from iminuit import Minuit
from iminuit.cost import LeastSquares

from tools.binnings import Binning
from tools.corrections import CORRECTION_FUNCTIONS, CORRECTION_GUESSES
from tools.histograms import WeightedHistogram, plot_histogram_1d
from tools.statistics import calculate_efficiency, calculate_efficiency_error
from tools.utilities import load_histograms, save_figure


def fit_correction(mc_values, iss_values, iss_errors, model, plot):
    fit_func = CORRECTION_FUNCTIONS[model]
    guess = CORRECTION_GUESSES[model]
    lsq = LeastSquares(mc_values, iss_values, iss_errors, fit_func)
    m = Minuit(lsq, **guess)
    m.migrad()
    m.minos()
    m.hesse()
    parameters = dict(zip(m.parameters, uncertainties.correlated_values(m.values, np.array(m.covariance))))
    print(parameters)
    parameter_values = {key: value.nominal_value for key, value in parameters.items()}
    fit_values = fit_func(mc_values, **parameter_values)
    chisq = np.sum((fit_values - iss_values)**2 / iss_errors**2)
    dof = len(mc_values) - len(guess)
    rchisq = chisq / dof
    print(f"χ²: {chisq:.2f}, χ²/dof: {rchisq:.2f}")
    labels = [f"{name}={value:P}" for name, value in parameters.items()]
    labels.append(f"χ²/dof={chisq:.2f}/{dof}")
    plot.plot(fit_values, mc_values, "-", label="\n".join(labels))
    return parameter_values



def calculate_correction(variable, iss_hist, mc_hist, model, plotdir="plots", outputprefix="MCCorrection"):
    assert iss_hist.dimensions == 1 and mc_hist.dimensions == 1 and iss_hist.binnings[0] == mc_hist.binnings[0]
    binning = iss_hist.binnings[0]
    bin_centers = binning.bin_centers[1:-1]
    iss_values = np.cumsum(iss_hist.values[1:-1])
    iss_total = iss_values[-1]
    iss_fraction = calculate_efficiency(iss_values, iss_total)
    iss_fraction_error = calculate_efficiency_error(iss_values, iss_total)
    mc_values = np.cumsum(mc_hist.values[1:-1])
    mc_total = mc_values[-1]
    mc_fraction = calculate_efficiency(mc_values, mc_total)
    mc_fraction_error = calculate_efficiency_error(mc_values, mc_total)
    interpolation_efficiencies = np.linspace(0.1, 0.9, 100)
    iss_interpolated_values = np.interp(interpolation_efficiencies, iss_fraction, bin_centers)
    mc_interpolated_values = np.interp(interpolation_efficiencies, mc_fraction, bin_centers)
    iss_interpolated_errors = np.interp(interpolation_efficiencies, iss_fraction, iss_fraction_error)
    mc_interpolated_errors = np.interp(interpolation_efficiencies, mc_fraction, mc_fraction_error)

    figure = plt.figure(figsize=(12, 6.15))
    plot = figure.subplots(1, 1)
    plot.set_title(f"MC correction {variable}")
    plot.set_xlabel(f"ISS {variable}")
    plot.set_ylabel(f"MC {variable}")
    plot.errorbar(iss_interpolated_values, mc_interpolated_values, xerr=iss_interpolated_errors, yerr=mc_interpolated_errors, fmt=".")

    parameters = fit_correction(mc_interpolated_values, iss_interpolated_values, iss_interpolated_errors, model, plot)

    plot.legend()
    figure.tight_layout()
    save_figure(figure, plotdir, f"{outputprefix}_{variable}")

    distribution_figure = plt.figure(figsize=(12, 6.15))
    distribution_plot_before, distribution_plot_after = distribution_figure.subplots(1, 2)
    distribution_figure.suptitle(variable)
    distribution_plot_before.set_title("Without correction")
    distribution_plot_after.set_title("With correction")

    mc_scale = iss_hist.values.sum() / mc_hist.values.sum()

    plot_histogram_1d(distribution_plot_before, iss_hist, style="iss", label="ISS")
    plot_histogram_1d(distribution_plot_before, mc_hist, style="mc", label="MC", scale=mc_scale)

    corr_function = CORRECTION_FUNCTIONS[model]
    mc_hist.binnings[0].edges = corr_function(mc_hist.binnings[0].edges, **parameters)
    mc_hist.binnings[0].bin_centers = corr_function(mc_hist.binnings[0].bin_centers, **parameters)

    plot_histogram_1d(distribution_plot_after, iss_hist, style="iss", label="ISS")
    plot_histogram_1d(distribution_plot_after, mc_hist, style="mc", label="MC", scale=mc_scale)

    distribution_plot_before.legend()
    distribution_plot_after.legend()

    distribution_figure.tight_layout()
    save_figure(distribution_figure, plotdir, f"{outputprefix}_{variable}_distribution")

    return model, parameters


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iss", required=True, help="File containing ISS histograms.")
    parser.add_argument("--mc", required=True, help="File containing ISS histograms.")
    parser.add_argument("--variable", action="append", dest="variables", required=True, nargs="+", help="Variable and correction model.")
    parser.add_argument("--outputprefix", default="MCCorrections") 
    parser.add_argument("--plotdir", default="plots")
    parser.add_argument("--resultdir", default="results")

    args = parser.parse_args()

    models = {}
    secondaries = {}
    variables = []
    for variable, model, *secondary_variables in args.variables:
        variables.append(variable)
        models[variable] = model
        for secondary in secondary_variables:
            secondaries[secondary] = variable

    os.makedirs(args.resultdir, exist_ok=True)
    os.makedirs(args.plotdir, exist_ok=True)

    iss_hists, iss_hists_per_rig = load_histograms(args.iss, variables)
    mc_hists, mc_hists_per_rig = load_histograms(args.mc, variables)

    corrections = {
        variable: calculate_correction(variable, iss_hists[variable], mc_hists[variable], models[variable], plotdir=args.plotdir, outputprefix=args.outputprefix)
        for variable in variables
    }
    for secondary, primary in secondaries.items():
        corrections[secondary] = corrections[primary]

    with open(os.path.join(args.resultdir, f"{args.outputprefix}.json"), "w") as corrections_file:
        json.dump(corrections, corrections_file)


if __name__ == "__main__":
    main()
