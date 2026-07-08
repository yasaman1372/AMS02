#!/usr/bin/env python3

import json
import os

import numpy as np
from scipy.special import erf, gamma, gammainc
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from iminuit import Minuit
from iminuit.cost import BinnedNLL
import uncertainties
import sympy as sp

from tools.histograms import WeightedHistogram, plot_histogram_1d, plot_histogram_2d
from tools.statistics import gaussian, hist_mean_and_std
from tools.utilities import plot_steps, round_down, round_up, save_figure, set_energy_ticks

def lowergamma(a, x):
    return gamma(a) * gammainc(a, x)

def gaussian_pdf(x, mu, sigma):
    return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def gaussian_cdf(x, mu, sigma):
    return (1 + erf(np.sqrt(2) * (x - mu) / (2 * sigma))) / 2

def double_sided_exp_pdf(x, sigma):
    return np.exp(-np.abs(x) / sigma) / (2 * sigma)

def double_sided_exp_cdf(x, sigma):
    result = np.zeros_like(x)
    result[x < 0] = np.exp(x[x < 0] / sigma) / 2
    result[x >= 0] = 1 - np.exp(-x[x >= 0] / sigma) / 2
    return result


def double_sided_mod_exp_pdf(x, sigma, alpha):
    return np.exp(-(x / sigma)**alpha) / (2 * sigma * gamma(1 + 1 / alpha))

def double_sided_mod_exp_cdf(x, sigma, alpha):
    result = np.zeros_like(x)
    result[x < 0] = 0.5 - lowergamma(1 / alpha, (-x[x < 0] / sigma)**alpha) / (2 * alpha * gamma(1 + 1 / alpha))
    result[x >= 0] = 0.5 + lowergamma(1 / alpha, (x[x >= 0] / sigma)**alpha) / (2 * alpha * gamma(1 + 1 / alpha))
    return result

def asymmetric_double_sided_mod_exp_pdf(x, sigma1, sigma2, alpha):
    result = np.zeros_like(x)
    f1 = 2 / (1 + sigma2 / sigma1)
    f2 = 2 / (1 + sigma1 / sigma2)
    result[x < 0] = f1 * double_sided_mod_exp_pdf(x[x < 0], sigma1, alpha)
    result[x >= 0] = f2 * double_sided_mod_exp_pdf(x[x >= 0], sigma2, alpha)
    return result

def asymmetric_double_sided_mod_exp_cdf(x, sigma1, sigma2, alpha):
    result = np.zeros_like(x)
    f1 = 2 / (1 + sigma2 / sigma1)
    f2 = 2 / (1 + sigma1 / sigma2)
    result[x < 0] = f1 * double_sided_mod_exp_cdf(x[x < 0], sigma1, alpha)
    result[x >= 0] = f1 / 2 + f2 * (double_sided_mod_exp_cdf(x[x >= 0], sigma2, alpha) - 0.5)
    return result


def two_double_sided_exp_cdf(x, sigma1, sigma2, f):
    w1 = 1 / (1 + f)
    w2 = f / (1 + f)
    return w1 * double_sided_exp_cdf(x, sigma1) + w2 * double_sided_exp_cdf(x, sigma2)

def three_double_sided_exp_cdf(x, sigma1, sigma2, sigma3, f1, f2):
    w1 = 1 / (1 + f1 + f2)
    w2 = f1 / (1 + f1 + f2)
    w3 = f2 / (1 + f1 + f2)
    return w1 * double_sided_exp_cdf(x, sigma1) + w2 * double_sided_exp_cdf(x, sigma2) + w3 * double_sided_exp_cdf(x, sigma3)


def two_double_sided_mod_exp_cdf(x, sigma1, alpha1, sigma2, alpha2, f):
    w1 = 1 / (1 + f)
    w2 = f / (1 + f)
    return w1 * double_sided_mod_exp_cdf(x, sigma1, alpha1) + w2 * double_sided_mod_exp_cdf(x, sigma2, alpha2)

def asymmetric_double_sided_mod_exp_plus_double_sided_mod_exp_cdf(x, sigma11, sigma12, alpha1, sigma2, alpha2, f):
    w1 = 1 / (1 + f)
    w2 = f / (1 + f)
    return w1 * asymmetric_double_sided_mod_exp_cdf(x, sigma11, sigma12, alpha1) + w2 * double_sided_mod_exp_cdf(x, sigma2, alpha2)

def two_asymmetric_double_sided_mod_exp_cdf(x, sigma11, sigma12, alpha1, sigma21, sigma22, alpha2, f):
    w1 = 1 / (1 + f)
    w2 = f / (1 + f)
    return w1 * asymmetric_double_sided_mod_exp_cdf(x, sigma11, sigma12, alpha1) + w2 * asymmetric_double_sided_mod_exp_cdf(x, sigma21, sigma22, alpha2)

def two_shifted_asymmetric_double_sided_mod_exp_cdf(x, mu, sigma11, sigma12, alpha1, sigma21, sigma22, alpha2, f):
    w1 = 1 / (1 + f)
    w2 = f / (1 + f)
    return w1 * asymmetric_double_sided_mod_exp_cdf(x - mu, sigma11, sigma12, alpha1) + w2 * asymmetric_double_sided_mod_exp_cdf(x - mu, sigma21, sigma22, alpha2)

    


class Dataset:
    def __init__(self, name, label, hists):
        self.name = name
        self.label = label
        self.hists = hists

    @staticmethod
    def load(name, label, hist_filename, pairs):
        with np.load(hist_filename) as file:
            hists = {
                pair: WeightedHistogram.from_file(file, f"pair_hist_{pair[0]}_{pair[1]}")
                for pair in pairs
            }
            return Dataset(name, label, hists)

DOUBLE_SIDED_EXP = (double_sided_exp_cdf, dict(sigma=10), dict(sigma=(0, None)))
TWO_DOUBLE_SIDED_EXP = (two_double_sided_exp_cdf, dict(sigma1=10, sigma2=0.1, f=0.5), dict(sigma1=(0, None), sigma2=(0, None), f=(0, None)))
THREE_DOUBLE_SIDED_EXP = (three_double_sided_exp_cdf, dict(sigma1=20, sigma2=0.3, sigma3=0.03, f1=0.95, f2=0.9), dict(sigma1=(0, None), sigma2=(0, None), sigma3=(0, None), f1=(0, None), f2=(0, None)))
DOUBLE_SIDED_MOD_EXP = (double_sided_mod_exp_cdf, dict(sigma=10, alpha=1), dict(sigma=(0, None), alpha=(0, None)))
TWO_DOUBLE_SIDED_MOD_EXP = (two_double_sided_mod_exp_cdf, dict(sigma1=50, sigma2=3e-4, alpha1=1.9, alpha2=0.2, f=0.7), dict(sigma1=(0, None), alpha1=(0, None), sigma2=(0, None), alpha2=(0, None), f=(0, None)))
ASYMMETRIC_PLUS_NORMAL_DOUBLE_SIDED_MOD_EXP = (asymmetric_double_sided_mod_exp_plus_double_sided_mod_exp_cdf, dict(sigma11=30, sigma12=50, alpha1=2, sigma2=3e-4, alpha2=0.2, f=0.7), dict(sigma11=(0, None), sigma12=(0, None), alpha1=(0, None), sigma2=(0, None), alpha2=(0, None), f=(0, None)))
TWO_ASYMMETRIC_DOUBLE_SIDED_MOD_EXP = (two_asymmetric_double_sided_mod_exp_cdf, dict(sigma11=30, sigma12=50, alpha1=2, sigma21=10e-3, sigma22=25e-3, alpha2=0.25, f=1), dict(sigma11=(0, None), sigma12=(0, None), alpha1=(0, None), sigma21=(0, None), sigma22=(0, None), alpha2=(0, None), f=(0, None)))
TWO_SHIFTED_ASYMMETRIC_DOUBLE_SIDED_MOD_EXP = (two_shifted_asymmetric_double_sided_mod_exp_cdf, dict(mu=0, sigma11=30, sigma12=50, alpha1=2, sigma21=10e-3, sigma22=25e-3, alpha2=0.25, f=1), dict(mu=(None, None), sigma11=(0, None), sigma12=(0, None), alpha1=(0, None), sigma21=(0, None), sigma22=(0, None), alpha2=(0, None), f=(0, None)))

MODELS = {
    "vertex_pr_vertex_coord_x": TWO_DOUBLE_SIDED_MOD_EXP,
    "vertex_pr_vertex_coord_y": TWO_SHIFTED_ASYMMETRIC_DOUBLE_SIDED_MOD_EXP,
    "vertex_pr_vertex_coord_z": TWO_ASYMMETRIC_DOUBLE_SIDED_MOD_EXP,
}


def derive_resolution(dataset, energy_estimator, resolution_variable, rigidity_range, plotdir, resultdir, prefix, title, fit_key):

    rigidity_min, rigidity_max = rigidity_range

    resolution_hist = dataset.hists[(energy_estimator, resolution_variable)]

    rigidity_binning = resolution_hist.binnings[0]
    rigidity_values = rigidity_binning.bin_centers[1:-1]
    res_mean, res_std, res_mean_error = hist_mean_and_std(resolution_hist)

    fitted_resolutions = []

    for bin, rigidity_low, rigidity_high, mean_value, std_value in zip(range(1, len(rigidity_values) + 1), rigidity_binning.edges[1:], rigidity_binning.edges[2:], res_mean, res_std):
        if rigidity_low >= rigidity_min and rigidity_high <= rigidity_max:
            projection = resolution_hist.project(bin)

            zoom_levels = (1, 0.1, 0.01, 0.001)

            figure_1d = plt.figure(figsize=(16, 8))
            figure_1d.suptitle(f"{title} {rigidity_low:.2f}<R/GV<{rigidity_high:.2f}")
            plots_1d, residual_plots = figure_1d.subplots(2, len(zoom_levels), height_ratios=(2, 1))
            for plot_1d in plots_1d:
                plot_histogram_1d(plot_1d, projection, log=True, label=f"{dataset.label}", draw_density=True)

            for zoom_level, plot_1d, plot_res in zip(zoom_levels, plots_1d, residual_plots):
                xlim_min, xlim_max = plot_1d.get_xlim()
                zoom_low = 0.5 - zoom_level / 2
                zoom_high = 0.5 + zoom_level / 2
                plot_1d.set_xlim(zoom_low * xlim_min + zoom_high * xlim_max, zoom_high * xlim_min + zoom_low * xlim_max)
                plot_res.set_xlim(zoom_low * xlim_min + zoom_high * xlim_max, zoom_high * xlim_min + zoom_low * xlim_max)

            last_parameters = None

            if fit_key in MODELS:
                fit_cdf, fit_params_guess, fit_params_limits = MODELS[fit_key]
                fit_params_fixed = None
                if last_parameters is not None:
                    fit_params_guess = last_parameters
                data_counts = np.stack((projection.values[1:-1], projection.squared_values[1:-1]), axis=1)
                data_values = projection.values[1:-1] / projection.values.sum()
                data_errors = np.maximum(projection.get_errors()[1:-1], 1) / projection.values.sum()
                binning = projection.binnings[0]
                loss = BinnedNLL(data_counts, binning.edges[1:-1], fit_cdf)
                m = Minuit(loss, **fit_params_guess)
                if fit_params_limits is not None:
                    for key, limits in fit_params_limits.items():
                        m.limits[key] = limits
                if fit_params_fixed is not None:
                    for key in fit_params_fixed:
                        m.fixed[key] = True
                m.migrad()
                if m.valid:
                    m.minos()
                print(m)

                try:
                    if m.covariance is not None:
                        fit_parameters = dict(zip(m.parameters, uncertainties.correlated_values(m.values, np.array(m.covariance))))
                except np.linalg.LinAlgError:
                    pass
                fit_param_names = m.parameters
                fit_param_values = m.values
                fit_param_errors = m.errors
                fit_param_dict = dict(zip(fit_param_names, fit_param_values))
                fit_param_error_dict = dict(zip(fit_param_names, fit_param_errors))
                fit_values = np.diff(fit_cdf(binning.edges[1:-1], **fit_param_dict))
                residuals = (data_values - fit_values) / data_errors
                nonzero = (data_values > 0) & (fit_values > 0)
                chisq = (residuals[nonzero]**2).sum()
                dof = nonzero.sum() - len(fit_param_values)
                rchisq = chisq / dof
                bin_centers = (binning.edges[2:-1] + binning.edges[1:-2]) / 2
                bin_widths = binning.edges[2:-1] - binning.edges[1:-2]
                last_parameters = fit_param_dict

                #fitted_resolutions.append(((rigidity_low + rigidity_high) / 2, fit_param_values["sigma"], fit_param_errors["sigma"]))

                fit_label = "\n".join(f"{name}={uncertainties.ufloat(value, error):P}" for name, value, error in zip(fit_param_names, fit_param_values, fit_param_errors))
                for plot_1d in plots_1d:
                    plot_steps(plot_1d, binning.edges[1:-1], fit_values * projection.values.sum() / bin_widths, label=f"Fit\n$\\chi^2/dof={chisq:.1f}/{dof}={rchisq:.2f}$\n{fit_label}")
                for plot_res in residual_plots:
                    plot_res.plot(bin_centers, (data_values - fit_values) / data_errors, ".")
                plots_1d[0].legend()

            save_figure(figure_1d, plotdir, f"{prefix}_r{bin}")


    resolution_figure = plt.figure(figsize=(12, 6.15))
    resolution_figure.suptitle(f"{title}")
    gridspec = GridSpec(2, 2, hspace=0, width_ratios=(9, 0.3), height_ratios=(2, 1))
    resolution_plot_2d = resolution_figure.add_subplot(gridspec[0,0])
    resolution_plot = resolution_figure.add_subplot(gridspec[1,0], sharex=resolution_plot_2d)
    colorbar_plot = resolution_figure.add_subplot(gridspec[0,1])

    plot_histogram_2d(resolution_plot_2d, resolution_hist, log=True, scale=1 / resolution_hist.values.sum(axis=1), colorbar_ax=colorbar_plot)
    resolution_plot_2d.plot(rigidity_values, res_mean, ".", markersize=1, color="black")
    resolution_plot_2d.plot(rigidity_values, res_mean + res_std, "^", markersize=1, color="black")
    resolution_plot_2d.plot(rigidity_values, res_mean - res_std, "v", markersize=1, color="black")
    resolution_plot_2d.axvline(rigidity_min, linewidth=0.5, color="red")
    resolution_plot_2d.axvline(rigidity_max, linewidth=0.5, color="red")
    y_binning = resolution_hist.binnings[1]
    resolution_plot_2d.set_ylim(y_binning.edges[1], y_binning.edges[-2])

    resolution_plot.plot(rigidity_values, res_std, label="Resolution")
    resolution_plot.plot(rigidity_values, res_mean, label="Bias")
    if fitted_resolutions:
        fitted_resolutions = np.array(fitted_resolutions)
        fit_rig = fitted_resolutions[:,0]
        fit_sigma = fitted_resolutions[:,1]
        fit_sigma_error = fitted_resolutions[:,2]
        resolution_plot.errorbar(fit_rig, fit_sigma, fit_sigma_error, fmt=".")
        print(f"{fit_key} resolution: {np.mean(fit_sigma):.3g}")
    resolution_plot.set_ylim(round_down(np.min(res_mean[np.isfinite(res_mean)])), round_up(np.max(res_mean[np.isfinite(res_mean)])))

    resolution_plot.legend()
    set_energy_ticks(resolution_plot)
    resolution_plot.set_xlabel(f"{energy_estimator} / GV")
    resolution_plot.set_ylabel(f"Resolution")
    save_figure(resolution_figure, plotdir, f"{prefix}_{energy_estimator}_{resolution_variable}")

    np.savez(os.path.join(resultdir, f"{prefix}.npz"), rigidity=rigidity_values, bias=res_mean, resolution=res_std)


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", nargs=3, required=True, help="Label and path to histogram filename.")
    parser.add_argument("--resolution", nargs=3, dest="resolutions", action="append", required=True, help="Resolution name, rigidity estimator and resolution variable")
    parser.add_argument("--rigidity-range", type=float, nargs=2, required=True, help="Rigidity range to derive resolution in.")
    parser.add_argument("--resolution-prefix", required=True, help="Prefix to select parametrizations.")
    parser.add_argument("--resultdir", default="results", help="Directory to store result files in.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--outputprefix", default="Resolution", help="Outputprefix for results and plots.")

    args = parser.parse_args()

    os.makedirs(args.resultdir, exist_ok=True)
    os.makedirs(args.plotdir, exist_ok=True)

    dataset_name, dataset_label, dataset_filename = args.dataset
    pairs = []
    for resolution_name, energy_estimator, resolution_variable in args.resolutions:
        pairs.append((energy_estimator, resolution_variable))

    dataset = Dataset.load(dataset_name, dataset_label, dataset_filename, pairs)

    for resolution_name, energy_estimator, resolution_variable in args.resolutions:
        derive_resolution(dataset, energy_estimator, resolution_variable, rigidity_range=args.rigidity_range, plotdir=args.plotdir, resultdir=args.resultdir, prefix=f"{args.outputprefix}_{resolution_name}", title=f"{dataset_label} {resolution_name}", fit_key=f"{args.resolution_prefix}_{resolution_name}")


if __name__ == "__main__":
    main()

    
