#!/usr/bin/env python3

import multiprocessing as mp
import os

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
import uncertainties
from uncertainties import ufloat
from uncertainties import umath

from tools.binnings import make_lin_binning, make_energy_binning_from_config, make_mc_dataset_edge_binning, combine_binnings, reduce_bins, make_hierarchical_binning, make_signed_binning
from tools.config import get_config
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d
from tools.roottree import read_tree
from tools.statistics import calculate_chisq, hist_mean_and_std, hist_percentile, gaussian as gaussian_pdf, gaussian_cdf, king_pdf_1d
from tools.utilities import plot_steps, round_up, save_figure, set_energy_ticks, set_plot_lim_x, set_plot_lim_y
from tools.constants import BastianPlots


BRANCHES = ["McEnergy", "ThetaX", "ThetaY", "McThetaX", "McThetaY"]


def calculate_resolution_from_standard_deviation(hist_2d, event_percentiles ,plotdir="plots", outputprefix="StdDev"):
    mean, std, mean_error = hist_mean_and_std(hist_2d, axis=1)
    return std, None

def calculate_resolution_from_event_level_percentile(hist_2d, event_percentiles ,plotdir="plots", outputprefix="StdDev"):
    return (event_percentiles[:,1] - event_percentiles[:,0]) / 2, None

def calculate_resolution_from_percentile(hist_2d, event_percentiles ,confidence_level=0.6827, plotdir="plots", outputprefix="Percentile"):
    low_edge = hist_percentile(hist_2d, axis=1, percentile=0.5 - confidence_level / 2, bin_point="low", interpolate=True)
    high_edge = hist_percentile(hist_2d, axis=1, percentile=0.5 + confidence_level / 2, bin_point="high", interpolate=True)
    return (high_edge[1:-1] - low_edge[1:-1]) / 2, None

def king_pdf(x, N, sigma, gamma):
    return N * king_pdf_1d(x, sigma, gamma)
king_pdf._parameters = dict(x=None, N=(0, None), sigma=(0, None), gamma=(1.5, 10))

def guess_king_parameters(hist_1d):
    _, sigma_guess, _ = hist_mean_and_std(hist_1d)
    return dict(
        N=hist_1d.values.sum(),
        sigma=sigma_guess / 30,
        gamma=5,
    )

def double_king_pdf(x, N, frac_core, sigma, gamma, sigma_ratio, gamma2):
    return N * (frac_core * king_pdf_1d(x, sigma, gamma) + (1 - frac_core) * king_pdf_1d(x, sigma * sigma_ratio, gamma2))
double_king_pdf._parameters = dict(x=None, N=(0, None), frac_core=(0.5, 1), sigma=(0, None), gamma=(1.5, 5), sigma_ratio=(1.5, 10), gamma2=(1.5, 10))

def derive_sigma_from_king_function(parameters):
    return parameters["sigma"]
    #return parameters["sigma"] * (parameters["gamma"] / (parameters["gamma"] - 1.5))**0.5

def guess_double_king_parameters(hist_1d):
    _, sigma_guess, _ = hist_mean_and_std(hist_1d)
    return dict(
        N=hist_1d.values.sum(),
        frac_core=0.9,
        sigma=sigma_guess / 30,
        gamma=5,
        sigma_ratio=2,
        gamma2=10,
    )

def double_centered_gaussian_pdf(x, N, frac_core, sigma, sigma_ratio):
    return N * (frac_core * gaussian_pdf(x, 0, sigma) + (1 - frac_core) * gaussian_pdf(x, 0, sigma * sigma_ratio))
double_centered_gaussian_pdf._parameters = dict(x=None, N=(0, None), frac_core=(0.6, 1), sigma=(0, None), sigma_ratio=(2, 15))

def guess_double_centered_gaussian_parameters(hist_1d):
    _, sigma_guess, _ = hist_mean_and_std(hist_1d)
    return dict(
        N=hist_1d.values.sum(),
        frac_core=0.9,
        sigma=sigma_guess / 30,
        sigma_ratio=2,
    )

def centered_gaussian_pdf(x, N, sigma):
    return N * gaussian_pdf(x, 0, sigma)
centered_gaussian_pdf._parameters = dict(x=None, N=(0, None), sigma=(0, None))

def guess_centered_gaussian_parameters(hist_1d):
    _, sigma_guess, _ = hist_mean_and_std(hist_1d)
    return dict(
        N=hist_1d.values.sum(),
        sigma=sigma_guess / 20,
    )

def derive_sigma_from_gaussian(parameters):
    return parameters["sigma"]

def king_plus_gaussian_pdf(x, N, frac_king, sigma, gamma, sigma_ratio):
    return N * (frac_king * king_pdf_1d(x, sigma, gamma) + (1 - frac_king) * gaussian_pdf(x, 0, sigma * sigma_ratio))
king_plus_gaussian_pdf._parameters = dict(x=None, N=(0, None), frac_king=(0.9, 1), sigma=(0, None), gamma=(1.5, None), sigma_ratio=(5, 50))

def guess_king_plus_gaussian_parameters(hist_1d):
    _, sigma_guess, _ = hist_mean_and_std(hist_1d)
    return dict(
        N=hist_1d.values.sum(),
        frac_king=0.975,
        sigma=sigma_guess / 30,
        gamma=2.5,
        sigma_ratio=5,
    )


FUNCTION_COMPONENTS = {
    double_centered_gaussian_pdf: (
        lambda x, N, frac_core, sigma, sigma_ratio: double_centered_gaussian_pdf(x, N * frac_core, 1, sigma, sigma_ratio),
        lambda x, N, frac_core, sigma, sigma_ratio: double_centered_gaussian_pdf(x, N * (1 - frac_core), 0, sigma, sigma_ratio),
    ),
    double_king_pdf: (
        lambda x, N, frac_core, sigma, gamma, sigma_ratio, gamma2: double_king_pdf(x, N * frac_core, 1, sigma, gamma, sigma * sigma_ratio, gamma2),
        lambda x, N, frac_core, sigma, gamma, sigma_ratio, gamma2: double_king_pdf(x, N * (1 - frac_core), 0, sigma, gamma, sigma * sigma_ratio, gamma2),
    ),
    king_plus_gaussian_pdf: (
        lambda x, N, frac_king, sigma, gamma, sigma_ratio: king_pdf(x, N * frac_king, sigma, gamma),
        lambda x, N, frac_king, sigma, gamma, sigma_ratio: centered_gaussian_pdf(x, N * (1 - frac_king), sigma * sigma_ratio),
    ),
}

def rasterize_function(function, bin_edges, points_per_bin=10):
    """
    Read a function value at multiple points in each bin

    When fitting a function to a histogram, reading the function the bin centers
    is correct only for linear functions.
    In general, the average of the bin has to be calculated using an integral.
    For functions that are not analytically integrable, a numeric integral can be
    approximated by averaging the function value at multiple points in each bin.
    """
    points_in_bin = (np.arange(points_per_bin) + 0.5) / points_per_bin
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    bin_start = bin_edges[:-1]
    grid = bin_start[:,None] + bin_widths[:,None] * points_in_bin[None,:]
    x = grid.reshape(-1)
    def _rasterized_function(_, *parameters):
        y = function(x, *parameters)
        y_grid = y.reshape((-1, points_per_bin))
        y_mean = np.mean(y_grid, axis=1)
        return y_mean * bin_widths
    if hasattr(function, "_parameters"):
        _rasterized_function._parameters = function._parameters
    return _rasterized_function


def fit_function(hist_1d, function, parameter_guess, plot_fit=False, plotdir="plots", outputprefix="FitPlot"):
    binning = hist_1d.binnings[0]
    bin_edges = binning.edges[1:-1]
    rasterized_function = rasterize_function(function, bin_edges)
    loss = LeastSquares(binning.bin_centers[1:-1], hist_1d.values[1:-1], np.maximum(hist_1d.get_errors()[1:-1], 1), rasterized_function)
    m = Minuit(loss, **parameter_guess)
    m.migrad()
    if m.valid:
        m.hesse()
        fit_parameters = dict(zip(m.parameters, uncertainties.correlated_values(m.values, np.array(m.covariance))))
    else:
        fit_parameters = None
    print(m)
    guess_values = rasterized_function(None, *parameter_guess.values())
    fit_param_values = dict(zip(m.parameters, m.values))
    fit_param_errors = dict(zip(m.parameters, m.errors))
    fit_values = rasterized_function(None, *m.values)
    chisq, dof, rchisq = calculate_chisq(hist_1d.values[1:-1], fit_values, hist_1d.get_errors()[1:-1], len(fit_param_values))
    if plot_fit:
        figure = plt.figure(figsize=(8, 4.2))
        plot = figure.subplots(1, 1)
        plot_histogram_1d(plot, hist_1d, style="iss")
        plot_steps(plot, bin_edges, fit_values, label=f"Fit $\\chi^2={chisq:.1f}/{dof:.0f}={rchisq:.2f}$", alpha=0.9)
        #plot_steps(plot, bin_edges, guess_values, label=f"Guess", alpha=0.9)
        if function in FUNCTION_COMPONENTS:
            for component_index, component in enumerate(FUNCTION_COMPONENTS[function]):
                rasterized_component = rasterize_function(component, bin_edges)
                component_values = rasterized_component(None, *m.values)
                plot_steps(plot, bin_edges, component_values, linestyle="dashed", label=f"Component {component_index}")
        plot.set_yscale("log")
        plot.set_ylim(bottom=0.1, top=round_up(np.max(hist_1d.values)) * 1.5)
        plot.legend()
        save_figure(figure, plotdir, f"{outputprefix}")
    return fit_param_values, fit_param_errors, fit_parameters, rchisq


def calculate_resolution_from_fit(hist_2d, pdf, guess_function, sigma_function, title="Fit", plotdir="plots", outputprefix="Fit"):
    resolution = []
    resolution_uncertainty = []
    energy_points = []
    parameters = {parameter: [] for parameter in pdf._parameters if parameter != "x"}
    parameter_errors = {parameter: [] for parameter in pdf._parameters if parameter != "x"}
    rchisqs = []
    for bin_index, (hist_1d, energy_min, energy_max) in enumerate(hist_2d.project_all()):
        if hist_1d.values.sum() < 10:
            resolution.append(np.nan)
            resolution_uncertainty.append(np.nan)
            continue
        guess_parameters = guess_function(hist_1d)
        fit_param_values, fit_param_errors, fit_parameters, fit_rchisq = fit_function(hist_1d, pdf, guess_parameters, plot_fit=True, outputprefix=f"{outputprefix}_fit_{bin_index}")
        if fit_parameters is not None:
            energy_points.append((energy_min + energy_max) / 2)
            for parameter_name in parameters:
                parameters[parameter_name].append(fit_parameters[parameter_name].nominal_value)
                parameter_errors[parameter_name].append(fit_parameters[parameter_name].std_dev)
            rchisqs.append(fit_rchisq)
            width = sigma_function(fit_parameters)
            resolution.append(width.nominal_value)
            resolution_uncertainty.append(width.std_dev)
        else:
            resolution.append(np.nan)
            resolution_uncertainty.append(np.nan)
    for parameter_name in parameters:
        parameter_figure = plt.figure(figsize=(8, 4.2))
        parameter_plot = parameter_figure.subplots(1, 1)
        values = np.array(parameters[parameter_name])
        errors = np.array(parameter_errors[parameter_name])
        mask = errors < np.abs(values)
        parameter_plot.errorbar(np.array(energy_points)[mask], values[mask], errors[mask], fmt=".")
        parameter_plot.set_title(title)
        parameter_plot.set_ylabel(parameter_name)
        parameter_plot.set_xlabel("$E_{MC}$/ GeV")
        parameter_plot.set_xscale("log")
        save_figure(parameter_figure, plotdir, f"{outputprefix}_parameter_{parameter_name}_lin", close_figure=False)
        parameter_plot.set_yscale("log")
        save_figure(parameter_figure, plotdir, f"{outputprefix}_parameter_{parameter_name}_log")
    chisq_figure = plt.figure(figsize=(8, 4.2))
    chisq_plot = chisq_figure.subplots(1, 1)
    chisq_plot.plot(np.array(energy_points), np.array(rchisqs), ".")
    chisq_plot.set_xscale("log")
    chisq_plot.set_xlabel("$E_{MC}$ / GeV")
    chisq_plot.set_ylabel("$\\chi^2/dof$")
    chisq_plot.axhline(1, color="darkgray", alpha=0.5)
    save_figure(chisq_figure, plotdir, f"{outputprefix}_rchisq")
    return np.array(resolution), np.array(resolution_uncertainty)

def calculate_resolution_from_gaussian_fit(hist_2d, event_percentiles ,plotdir="plots", outputprefix="GaussianFit"):
    return calculate_resolution_from_fit(hist_2d, centered_gaussian_pdf, guess_centered_gaussian_parameters, derive_sigma_from_gaussian, plotdir=plotdir, outputprefix=outputprefix)

def calculate_resolution_from_double_gaussian_fit(hist_2d, event_percentiles ,plotdir="plots", outputprefix="DoubleGaussianFit"):
    return calculate_resolution_from_fit(hist_2d, double_centered_gaussian_pdf, guess_double_centered_gaussian_parameters, derive_sigma_from_gaussian, plotdir=plotdir, outputprefix=outputprefix)

def calculate_resolution_from_king_fit(hist_2d, event_percentiles ,plotdir="plots", outputprefix="DoubleKingFit"):
    return calculate_resolution_from_fit(hist_2d, king_pdf, guess_king_parameters, derive_sigma_from_king_function, plotdir=plotdir, outputprefix=outputprefix)

def calculate_resolution_from_double_king_fit(hist_2d, event_percentiles ,plotdir="plots", outputprefix="DoubleKingFit"):
    return calculate_resolution_from_fit(hist_2d, double_king_pdf, guess_double_king_parameters, derive_sigma_from_king_function, plotdir=plotdir, outputprefix=outputprefix)

def calculate_resolution_from_king_plus_gaussian_fit(hist_2d, event_percentiles ,plotdir="plots", outputprefix="KingPlusGaussianFit"):
    return calculate_resolution_from_fit(hist_2d, king_plus_gaussian_pdf, guess_king_plus_gaussian_parameters, derive_sigma_from_king_function, plotdir=plotdir, outputprefix=outputprefix)


RESOLUTION_METHODS = {
    "std": calculate_resolution_from_standard_deviation,
    "percentile": calculate_resolution_from_percentile,
    "event_level_percentile": calculate_resolution_from_event_level_percentile,
    "double-king-fit": calculate_resolution_from_double_king_fit,
    "double-gaussian-fit": calculate_resolution_from_double_gaussian_fit,
    "king-fit": calculate_resolution_from_king_fit,
    "gaussian-fit": calculate_resolution_from_gaussian_fit,
    "king-plus-gaussian-fit": calculate_resolution_from_king_plus_gaussian_fit,
}


def resolution_parametrization(energy, a, b, c):
    return a / (energy + b) + c


def fit_resolution_parametrization(energy_values, resolution_values, resolution_errors=None, min_energy=None, max_energy=None):

    if resolution_errors is None:
        resolution_errors = np.ones_like(resolution_values)
    mask = np.ones_like(energy_values, dtype=bool)
    if min_energy is not None:
        mask = mask & (energy_values >= min_energy)
    if max_energy is not None:
        mask = mask & (energy_values <= max_energy)
    cost = LeastSquares(energy_values[mask], resolution_values[mask], resolution_errors[mask], resolution_parametrization)

    m = Minuit(cost, a=1, b=0.5, c=0.1)
    m.limits["a"] = (0, None)
    m.limits["b"] = (0, None)
    m.limits["c"] = (0, None)
    m.migrad()
    if m.valid:
        m.hesse()
    print(m)
    parameter_dict = dict(zip(m.parameters, m.values))
    parameter_error_dict = dict(zip(m.parameters, m.errors))
    return parameter_dict, parameter_error_dict


def handle_file(arg):
    filenames, treename, chunk_size, rank, nranks, kwargs = arg

    energy_binning = kwargs["energy_binning"]
    alpha_binning = kwargs["alpha_binning"]

    histogram = Histogram(energy_binning, alpha_binning, alpha_binning)
    Alpha_x = [np.array([]) for x in range(len(energy_binning.bin_centers[1:-1]))]
    Alpha_y = [np.array([]) for x in range(len(energy_binning.bin_centers[1:-1]))]

    for events in read_tree(filenames, treename, rank=rank, nranks=nranks, chunk_size=chunk_size, branches=BRANCHES, cache_file=False):
        alpha_x = np.rad2deg(events.ThetaX - events.McThetaX)
        alpha_y = np.rad2deg(events.ThetaY - events.McThetaY)
        histogram.fill(events.McEnergy, alpha_x, alpha_y)
        ids = energy_binning.get_indices(events.McEnergy)
        for i in range(len(Alpha_x)):
            valid_ids = ids == i
            Alpha_x[i] = np.concatenate((Alpha_x[i], ak.to_numpy(alpha_x[valid_ids])))
            Alpha_y[i] = np.concatenate((Alpha_y[i], ak.to_numpy(alpha_y[valid_ids])))
        
    return histogram, Alpha_x, Alpha_y


def make_args(filename, treename, chunk_size, nranks, **kwargs):
    for rank in range(nranks):
        yield (filename, treename, chunk_size, rank, nranks, kwargs)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs=2, required=True, help="Path to config file and working directory.")
    parser.add_argument("--tree", nargs="+", required=True, help="MC Gamma Tree to read.")
    parser.add_argument("--treename", default="GammaTree", help="Name of the tree in the ROOT file to read.")
    parser.add_argument("--dataset", required=True, help="Name of the MC dataset to calculate the point spread function for.")
    parser.add_argument("--outputprefix", default="PointSpreadFunction", help="Prefix for plots and result files.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Number of events to read each step.")
    parser.add_argument("--title", default="Point Spread Function", help="Title for plots.")
    parser.add_argument("--resolution-method", choices=list(RESOLUTION_METHODS.keys()), default="event_level_percentile", help="Method with which to calculate the width of the PSF.")
    parser.add_argument("--energy-range", nargs=2, default=(0.15, 1000), type=float, help="Minimum and maximum energy to include in the plots.")
    parser.add_argument("--fit-energy-range", nargs=2, type=float, help="Minimum and maximum energy to include in the resolution fit.")
    parser.add_argument("--comparison", choices=list(BastianPlots.keys()), help="Version of bastians data to compare to.")
    parser.add_argument("--n-alpha-bins", type=int, default=1000, help="Number of bins to use for the angular difference (e.g. 200 for ECAL, 1000 for Vertex).")
    parser.add_argument("--no-title", action='store_true', help='Use this to prevent titles from being generated.')
    parser.add_argument("--transparent", action="store_true", help='set this to make the background of the png transparent')

    args = parser.parse_args()
    config_filename, workdir = args.config
    config = get_config(config_filename)

    os.makedirs(args.resultdir, exist_ok=True)
    os.makedirs(args.plotdir, exist_ok=True)

    energy_binning = make_energy_binning_from_config(config)
    energy_binning_with_dataset_edges = combine_binnings((energy_binning, make_mc_dataset_edge_binning(config, args.dataset)))
    alpha_binning = make_lin_binning(-10, 10, args.n_alpha_bins)

    comparison = None
    if args.comparison is not None:
        comparison = BastianPlots[args.comparison]["psf"]

    min_energy, max_energy = None, None
    if args.energy_range is not None:
        min_energy, max_energy = args.energy_range
    min_fit_energy, max_fit_energy = None, None
    if args.fit_energy_range is not None:
        min_fit_energy, max_fit_energy = args.fit_energy_range

    event_histogram = Histogram(energy_binning, alpha_binning, alpha_binning, labels=("E / GeV", "$\\alpha_x/\\circ$", "$\\alpha_y/\\circ$"))
    Alpha_x = [np.array([]) for x in range(len(energy_binning.bin_centers[1:-1]))]
    Alpha_y = [np.array([]) for x in range(len(energy_binning.bin_centers[1:-1]))]
    Alpha_x_percentile = np.zeros((len(energy_binning.bin_centers[1:-1]), 2))
    Alpha_y_percentile = np.zeros((len(energy_binning.bin_centers[1:-1]), 2))

    with mp.Pool(args.nprocesses) as pool:
        pool_args = make_args(args.tree, args.treename, chunk_size=args.chunk_size, nranks=args.nprocesses, energy_binning=energy_binning, alpha_binning=alpha_binning)
        for hist, alpha_x, alpha_y in pool.imap_unordered(handle_file, pool_args):
            event_histogram.add(hist)
            for i in range(len(Alpha_x)):
                Alpha_x[i] = np.concatenate((Alpha_x[i], alpha_x[i]))
                Alpha_y[i] = np.concatenate((Alpha_y[i], alpha_y[i]))

    for i in range(len(Alpha_x)):
        print(Alpha_x[i])
        print(Alpha_y[i])
        if len(Alpha_x[i]) == 0:
            Alpha_x_percentile[i] = [np.nan, np.nan]
        else:
            Alpha_x_percentile[i] = np.percentile(np.array(Alpha_x[i]), np.array([50 -68.27/2, 50+68.27/2]))
        if len(Alpha_y[i]) == 0:
            Alpha_y_percentile[i] = [np.nan, np.nan]
        else:
            Alpha_y_percentile[i] = np.percentile(np.array(Alpha_y[i]), np.array([50 -68.27/2, 50+68.27/2]))

    results = {}
    event_histogram.add_to_file(results, "histogram_3d")
    results["energy_values"] = energy_binning.bin_centers
    results["Alpha_x_68Percentile"] = Alpha_x_percentile
    results["Alpha_y_68Percentile"] = Alpha_y_percentile

    figure_x = plt.figure(figsize=(8, 4.2))
    plot_x = figure_x.subplots(1, 1)
    hist_x = event_histogram.project_axis(axis=2)

    energy_bin_points = energy_binning.bin_centers[1:-1]
    alpha_x_resolution, alpha_x_resolution_uncertainty = RESOLUTION_METHODS[args.resolution_method](hist_x, Alpha_x_percentile, plotdir=args.plotdir, outputprefix=f"{args.outputprefix}_x")
    alpha_x_mask = np.isfinite(alpha_x_resolution)
    energy_fit_points = energy_bin_points
    if args.fit_energy_range is not None:
        alpha_x_mask = alpha_x_mask & (energy_bin_points >= min_fit_energy) & (energy_bin_points <= max_fit_energy)
        energy_fit_points = energy_bin_points[(energy_bin_points >= min_fit_energy) & (energy_bin_points <= max_fit_energy)]
    alpha_x_parameters, alpha_x_parameter_errors = fit_resolution_parametrization(energy_bin_points[alpha_x_mask], alpha_x_resolution[alpha_x_mask], alpha_x_resolution_uncertainty[alpha_x_mask] if alpha_x_resolution_uncertainty is not None else None, min_energy=min_energy, max_energy=max_energy)
    fit_label = f"$\\sigma={alpha_x_parameters['a']:.4f}^\\circ / (E/GeV + {alpha_x_parameters['b']:.4f}) + {alpha_x_parameters['c']:.4f}^\\circ$"
    plot_histogram_2d(plot_x, hist_x, scale=1 / hist_x.values.sum(axis=1), log=True, show_overflow=False, cmap="jet", mask_below=5e-4, max_value=0.35)
    plot_x.plot(energy_bin_points, +alpha_x_resolution, "^", markersize=1, color="black")
    plot_x.plot(energy_bin_points, -alpha_x_resolution, "v", markersize=1, color="black")
    plot_x.plot(energy_fit_points, resolution_parametrization(energy_fit_points, **alpha_x_parameters), "-", color="tab:red", linewidth=1, alpha=0.9, label=fit_label)
    if comparison is not None:
        fit_label_bastian = f'B. Beischer: $\\sigma={comparison["E_alphaX"]["fitpara"][0]:.4f}^\\circ / (E/GeV + {comparison["E_alphaX"]["fitpara"][1]:.4f}) + {comparison["E_alphaX"]["fitpara"][2]:.4f}^\\circ$'
        plot_x.plot(energy_fit_points, resolution_parametrization(energy_fit_points, comparison["E_alphaX"]["fitpara"][0], comparison["E_alphaX"]["fitpara"][1], comparison["E_alphaX"]["fitpara"][2]), "-", color="tab:green", linewidth=1, alpha=0.9, label=fit_label_bastian)
    set_energy_ticks(plot_x)
    if args.energy_range is None:
        plot_x.set_xlim(hist_x.binnings[0].edges[1], hist_x.binnings[0].edges[-2])
    else:
        plot_x.set_xlim(min_energy, max_energy)
    #plot_x.set_ylim(hist_x.binnings[1].edges[1], hist_x.binnings[1].edges[-2])
    plot_x.set_ylim(-7.5, 7.5)
    plot_x.legend()
    figure_x.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
    if args.no_title:
        save_figure(figure_x, args.plotdir, f"{args.outputprefix}_alpha_x_NoTitle", transparent=args.transparent)
    plot_x.set_title(args.title)
    save_figure(figure_x, args.plotdir, f"{args.outputprefix}_alpha_x", transparent=args.transparent)
    results["resolution_x"] = alpha_x_resolution
    results["resolution_parameters_x"] = np.array(list(alpha_x_parameters.values()))

    alpha_x_hist_1GeV, min_e, max_e = hist_x.project_by_value(1, axis=0, return_bin_edges=True)
    index_1GeV = (hist_x.binnings[0].get_indices([1], with_overflow=False) - 1)[0]
    fig_1_x = plt.figure(figsize=(8, 4.2))
    plot_1_x = fig_1_x.subplots(1,1)
    alpha_x_1_sigma_high = alpha_x_resolution[index_1GeV]
    alpha_x_1_sigma_low = -alpha_x_resolution[index_1GeV]
    plot_histogram_1d(plot_1_x, alpha_x_hist_1GeV, show_overflow=False, label_y='Events', color='black')
    plot_1_x.set_ylim(bottom=0)
    plot_1_x.set_xlim(-7.5, 7.5)
    #plot_1_x.axvline(0, color="red")
    plot_1_x.axvline(alpha_x_1_sigma_high, color="orange")
    plot_1_x.axvline(alpha_x_1_sigma_low, color="orange")
    plot_1_x.axvspan(alpha_x_1_sigma_low, alpha_x_1_sigma_high, color = 'orange', alpha=0.25, label = '68%')
    plot_1_x.legend()
    if args.no_title:
        save_figure(fig_1_x, args.plotdir, f"{args.outputprefix}_alpha_x_1_GeV_NoTitle", transparent=args.transparent)
    plot_1_x.set_title(f"$\\alpha_x$, ${min_e:.2f}<= E/GeV<{max_e:.2f}$")
    save_figure(fig_1_x, args.plotdir, f"{args.outputprefix}_alpha_x_1_GeV", transparent=args.transparent)

    alpha_x_hist_10GeV, min_e, max_e = hist_x.project_by_value(10, axis=0, return_bin_edges=True)
    index_10GeV = (hist_x.binnings[0].get_indices([10], with_overflow=False) - 1)[0]
    fig_10_x = plt.figure(figsize=(8, 4.2))
    plot_10_x = fig_10_x.subplots(1,1)
    alpha_x_10_sigma_high = alpha_x_resolution[index_10GeV]
    alpha_x_10_sigma_low = -alpha_x_resolution[index_10GeV]
    plot_histogram_1d(plot_10_x, alpha_x_hist_10GeV, show_overflow=False, label_y='Events', color='black')
    plot_10_x.set_ylim(bottom=0)
    plot_10_x.set_xlim(-2, 2)
    #plot_10_x.axvline(0, color="red")
    plot_10_x.axvline(alpha_x_10_sigma_high, color="orange")
    plot_10_x.axvline(alpha_x_10_sigma_low, color="orange")
    plot_10_x.axvspan(alpha_x_10_sigma_low, alpha_x_10_sigma_high, color = 'orange', alpha=0.25, label = '68%')
    plot_10_x.legend()
    if args.no_title:
        save_figure(fig_10_x, args.plotdir, f"{args.outputprefix}_alpha_x_10_GeV_NoTitle", transparent=args.transparent)
    plot_10_x.set_title(f"$\\alpha_x$, ${min_e:.2f}<= E/GeV<{max_e:.2f}$")
    save_figure(fig_10_x, args.plotdir, f"{args.outputprefix}_alpha_x_10_GeV", transparent=args.transparent)

    alpha_x_hist_100GeV, min_e, max_e = hist_x.project_by_value(100, axis=0, return_bin_edges=True)
    index_100GeV = (hist_x.binnings[0].get_indices([100], with_overflow=False) - 1)[0]
    fig_100_x = plt.figure(figsize=(8, 4.2))
    plot_100_x = fig_100_x.subplots(1,1)
    alpha_x_100_sigma_high = alpha_x_resolution[index_100GeV]
    alpha_x_100_sigma_low = -alpha_x_resolution[index_100GeV]
    plot_histogram_1d(plot_100_x, alpha_x_hist_100GeV, show_overflow=False, label_y='Events', color='black')
    plot_100_x.set_ylim(bottom=0)
    plot_100_x.set_xlim(-.5, .5)
    #plot_100_x.axvline(0, color="red")
    plot_100_x.axvline(alpha_x_100_sigma_high, color="orange")
    plot_100_x.axvline(alpha_x_100_sigma_low, color="orange")
    plot_100_x.axvspan(alpha_x_100_sigma_low, alpha_x_100_sigma_high, color = 'orange', alpha=0.25, label = '68%')
    plot_100_x.legend()
    if args.no_title:
        save_figure(fig_100_x, args.plotdir, f"{args.outputprefix}_alpha_x_100_GeV_NoTitle", transparent=args.transparent)
    plot_100_x.set_title(f"$\\alpha_x$, ${min_e:.2f}<= E/GeV<{max_e:.2f}$")
    save_figure(fig_100_x, args.plotdir, f"{args.outputprefix}_alpha_x_100_GeV", transparent=args.transparent)


    figure_y = plt.figure(figsize=(8, 4.2))
    plot_y = figure_y.subplots(1, 1)
    hist_y = event_histogram.project_axis(axis=1)
    alpha_y_resolution, alpha_y_resolution_uncertainty = RESOLUTION_METHODS[args.resolution_method](hist_y, Alpha_y_percentile, plotdir=args.plotdir, outputprefix=f"{args.outputprefix}_y")
    alpha_y_mask = np.isfinite(alpha_y_resolution)
    if args.fit_energy_range is not None:
        alpha_y_mask = alpha_y_mask & (energy_bin_points >= min_fit_energy) & (energy_bin_points <= max_fit_energy)
    alpha_y_parameters, alpha_y_parameter_errors = fit_resolution_parametrization(energy_bin_points[alpha_y_mask], alpha_y_resolution[alpha_y_mask], alpha_y_resolution_uncertainty[alpha_y_mask] if alpha_y_resolution_uncertainty is not None else None, min_energy=min_energy, max_energy=max_energy)
    fit_label = f"$\\sigma={alpha_y_parameters['a']:.4f}^\\circ / (E/GeV + {alpha_y_parameters['b']:.4f}) + {alpha_y_parameters['c']:.4f}^\\circ$"
    plot_histogram_2d(plot_y, hist_y, scale=1 / hist_y.values.sum(axis=1), log=True, show_overflow=False, cmap="jet", mask_below=5e-4, max_value=0.35)
    plot_y.plot(energy_bin_points, +alpha_y_resolution, "^", markersize=1, color="black")
    plot_y.plot(energy_bin_points, -alpha_y_resolution, "v", markersize=1, color="black")
    plot_y.plot(energy_fit_points, resolution_parametrization(energy_fit_points, **alpha_y_parameters), "-", color="tab:red", linewidth=1, label=fit_label, alpha=0.9)
    if comparison is not None:
        fit_label_bastian = f'B. Beischer: $\\sigma={comparison["E_alphaY"]["fitpara"][0]:.4f}^\\circ / (E/GeV + {comparison["E_alphaY"]["fitpara"][1]:.4f}) + {comparison["E_alphaY"]["fitpara"][2]:.4f}^\\circ$'
        plot_y.plot(energy_fit_points, resolution_parametrization(energy_fit_points, comparison["E_alphaY"]["fitpara"][0], comparison["E_alphaY"]["fitpara"][1], comparison["E_alphaY"]["fitpara"][2]), "-", color="tab:green", linewidth=1, label=fit_label_bastian, alpha=0.9)
    set_energy_ticks(plot_y)
    if args.energy_range is None:
        plot_y.set_xlim(hist_y.binnings[0].edges[1], hist_y.binnings[0].edges[-2])
    else:
        plot_y.set_xlim(min_energy, max_energy)
    #plot_y.set_ylim(hist_y.binnings[1].edges[1], hist_y.binnings[1].edges[-2])
    plot_y.set_ylim(-7.5, 7.5)
    plot_y.legend()
    figure_y.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
    if args.no_title:
        save_figure(figure_y, args.plotdir, f"{args.outputprefix}_alpha_y_NoTitle", transparent=args.transparent)
    plot_y.set_title(args.title)
    save_figure(figure_y, args.plotdir, f"{args.outputprefix}_alpha_y", transparent=args.transparent)
    results["resolution_y"] = alpha_y_resolution
    results["resolution_parameters_y"] = np.array(list(alpha_y_parameters.values()))

    alpha_y_hist_1GeV, min_e, max_e = hist_y.project_by_value(1, axis=0, return_bin_edges=True)
    fig_1_y = plt.figure(figsize=(8, 4.2))
    plot_1_y = fig_1_y.subplots(1, 1)
    alpha_y_1_sigma_high = alpha_y_resolution[index_1GeV]
    alpha_y_1_sigma_low = -alpha_y_resolution[index_1GeV]
    plot_histogram_1d(plot_1_y, alpha_y_hist_1GeV, show_overflow=False, label_y='Events', color='black')
    plot_1_y.set_ylim(bottom=0)
    plot_1_y.set_xlim(-7.5, 7.5)
    #plot_1_y.axvline(0, color = 'red')
    plot_1_y.axvline(alpha_y_1_sigma_high, color = 'orange')
    plot_1_y.axvline(alpha_y_1_sigma_low, color = 'orange')
    plot_1_y.axvspan(alpha_y_1_sigma_low, alpha_y_1_sigma_high, color = 'orange', alpha=0.25, label = '68%')
    plot_1_y.legend()
    if args.no_title:
        save_figure(fig_1_y, args.plotdir, f"{args.outputprefix}_alpha_y_1_GeV_NoTitle", transparent=args.transparent)
    plot_1_y.set_title(f"Alpha_Y, ${min_e:.2f}<= E/GeV<{max_e:.2f}$")
    save_figure(fig_1_y, args.plotdir, f"{args.outputprefix}_alpha_y_1_GeV", transparent=args.transparent)

    alpha_y_hist_10GeV, min_e, max_e = hist_y.project_by_value(10, axis=0, return_bin_edges=True)
    index_10GeV = (hist_y.binnings[0].get_indices([10], with_overflow=False) - 1)[0]
    fig_10_y = plt.figure(figsize=(8, 4.2))
    plot_10_y = fig_10_y.subplots(1,1)
    alpha_y_10_sigma_high = alpha_y_resolution[index_10GeV]
    alpha_y_10_sigma_low = -alpha_y_resolution[index_10GeV]
    plot_histogram_1d(plot_10_y, alpha_y_hist_10GeV, show_overflow=False, label_y='Events', color='black')
    plot_10_y.set_ylim(bottom=0)
    plot_10_y.set_xlim(-2, 2)
    #plot_10_y.axvline(0, color="red")
    plot_10_y.axvline(alpha_y_10_sigma_high, color="orange")
    plot_10_y.axvline(alpha_y_10_sigma_low, color="orange")
    plot_10_y.axvspan(alpha_y_10_sigma_low, alpha_y_10_sigma_high, color = 'orange', alpha=0.25, label = '68%')
    plot_10_y.legend()
    if args.no_title:
        save_figure(fig_10_y, args.plotdir, f"{args.outputprefix}_alpha_y_10_GeV_NoTitle", transparent=args.transparent)
    plot_10_y.set_title(f"$\\alpha_y$, ${min_e:.2f}<= E/GeV<{max_e:.2f}$")
    save_figure(fig_10_y, args.plotdir, f"{args.outputprefix}_alpha_y_10_GeV", transparent=args.transparent)

    alpha_y_hist_100GeV, min_e, max_e = hist_y.project_by_value(100, axis=0, return_bin_edges=True)
    index_100GeV = (hist_y.binnings[0].get_indices([100], with_overflow=False) - 1)[0]
    fig_100_y = plt.figure(figsize=(8, 4.2))
    plot_100_y = fig_100_y.subplots(1,1)
    alpha_y_100_sigma_high = alpha_y_resolution[index_100GeV]
    alpha_y_100_sigma_low = -alpha_y_resolution[index_100GeV]
    plot_histogram_1d(plot_100_y, alpha_y_hist_100GeV, show_overflow=False, label_y='Events', color='black')
    plot_100_y.set_ylim(bottom=0)
    plot_100_y.set_xlim(-.5, .5)
    #plot_100_y.axvline(0, color="red")
    plot_100_y.axvline(alpha_y_100_sigma_high, color="orange")
    plot_100_y.axvline(alpha_y_100_sigma_low, color="orange")
    plot_100_y.axvspan(alpha_y_100_sigma_low, alpha_y_100_sigma_high, color = 'orange', alpha=0.25, label = '68%')
    plot_100_y.legend()
    if args.no_title:
        save_figure(fig_100_y, args.plotdir, f"{args.outputprefix}_alpha_y_100_GeV_NoTitle", transparent=args.transparent)
    plot_100_y.set_title(f"$\\alpha_y$, ${min_e:.2f}<= E/GeV<{max_e:.2f}$")
    save_figure(fig_100_y, args.plotdir, f"{args.outputprefix}_alpha_y_100_GeV", transparent=args.transparent)

    np.savez_compressed(os.path.join(args.resultdir, f"{args.outputprefix}.npz"), **results)


    for index, (hist_2d, energy_min, energy_max) in enumerate(event_histogram.project_all(axis=0)):
        figure_2d = plt.figure(figsize=(6, 5))
        plot_2d = figure_2d.subplots(1, 1)
        plot_histogram_2d(plot_2d, hist_2d, show_overflow=False, cmap="jet")
        plot_2d.set_xlim(-7, 7)
        plot_2d.set_ylim(-7, 7)

        figure_2d.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
        if args.no_title:
            save_figure(figure_2d, args.plotdir, f"{args.outputprefix}_energy_{index}_NoTitle", transparent=args.transparent)
        plot_2d.set_title(f"{args.title} ${energy_min:.2f}<=E/GeV<{energy_max:.2f}$")
        save_figure(figure_2d, args.plotdir, f"{args.outputprefix}_energy_{index}", transparent=args.transparent)


if __name__ == "__main__":
   main()
