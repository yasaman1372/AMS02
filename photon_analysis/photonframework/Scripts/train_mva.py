#!/usr/bin/env python3

from fnmatch import filter as fnmatch
from glob import glob
import json
import multiprocessing as mp
import multiprocessing.pool as mp_pool
import os

import numpy as np
from numpy.lib import recfunctions
import matplotlib.pyplot as plt
from matplotlib.contour import ContourSet
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogLocator, LogFormatter
import xgboost as xgb
from iminuit import Minuit
from iminuit.cost import LeastSquares, BinnedNLL, ExtendedBinnedNLL, ExtendedUnbinnedNLL
from iminuit.util import make_func_code
import awkward as ak
import uncertainties
from scipy.stats import gaussian_kde, norm, chi2
from scipy.interpolate import interp1d
from scipy.integrate import quad
from gammapy.stats import WStatCountsStatistic

from tools.binnings import Binning, Binnings, make_lin_binning, make_lin_binning_with_known_edge, make_log_binning, reduce_bins
from tools.confidence import calculate_contour, calculate_confidence_interval
from tools.config import get_config
from tools.constants import MC_PARTICLE_IDS, MC_PARTICLE_MASSES
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d
from tools.selection import Selection, Cut
from tools.statistics import calculate_correlation_and_error, calculate_likelihood, calculate_efficiency_and_error_weighted, calculate_efficiency_and_rejection, calculate_signal_and_background_efficiency, calculate_cut_value_for_efficiency, draw_random_from_hist, smooth_additive, calculate_chisq
from tools.utilities import plot_steps, shaded_steps, rec_to_float, filter_branches, float_or_int, plot_feature_importance, round_down, round_up, save_figure, format_order_of_magnitude, make_tab_palette, create_histogram_contour, set_energy_ticks, load_weighted_mc_trigger_count, make_kde_template
from tools.variables import VariableLabels

SOME_PRIME_NUMBER = 94777357
ANOTHER_PRIME_NUMBER = 58445929

def _print_dict(data):
    for key, value in data.items():
        if isinstance(value, dict):
            print(key, "dict")
            _print_dict(value)
        else:
            print(type(key), type(value), key, value)

def increase_value(value):
    return value * (0.9 + (value > 0) * 0.2)

def decrease_value(value):
    return value * (1.1 - (value > 0) * 0.2)


def get_rebin_factor(binning, max_factor=5):
    if len(binning) <= 20:
        return 1
    for factor in range(max_factor, 1, -1):
        if (len(binning) - 2) % factor == 0:
            return factor
    return 1


def _normalize_selection_name(name):
    if name.endswith("_positive"):
        return name[:-len("_positive")]
    if name.endswith("_negative"):
        return name[:-len("_neagtive")]
    return name

def merge_selections(first, second):
    common_keys = set(first) & set(second)
    return {key: first[key] + second[key] for key in common_keys}
    #first_keys = set(first) - common_keys
    #second_keys = set(second) - common_keys
    #normalized_keys_first = {_normalize_selection_name(key): key for key in first_keys}
    #normalized_keys_second = {_normalize_selection_name(key): key for key in second_keys}
    #if set(normalized_keys_first) ^ set(normalized_keys_second):
    #    raise ValueError(f"Cannot merge selections with non-matching keys {first_keys!r} and {second_keys!r}")
    #selections = {key: first[key] + second[key] for key in common_keys}
    #for normalized_key in normalized_keys_first:
    #    selections[normalized_key] = first[normalized_keys_first[normalized_key]] + second[normalized_keys_second[normalized_key]]
    #return selections

def linear_without_cutoff(x, a, b):
    return a * x + b

def linear_with_cutoff(x, a, b):
    return np.maximum(a * x + b, 0)

def linear_with_minimum(x, a, b, c):
    return np.maximum(a * x + b, c)

def fit_linear(x, y, y_err, cutoff=True):
    if cutoff:
        fit_function = linear_with_cutoff
        guess = dict(a=1, b=0)
    else:
        fit_function = linear_without_cutoff
        guess = dict(a=1, b=0)
    loss = LeastSquares(x, y, y_err, fit_function)
    m = Minuit(loss, **guess)
    m.migrad()
    print(m)
    fit_param_values = dict(zip(m.parameters, m.values))
    fit_param_errors = dict(zip(m.parameters, m.errors))
    fit_values = fit_function(x, **fit_param_values)
    smooth_x = np.linspace(x[0], x[-1], 10 * len(x))
    smooth_fit_values = fit_function(smooth_x, **fit_param_values)
    return m.valid, fit_param_values, fit_param_errors, fit_values, smooth_x, smooth_fit_values



PALETTE = make_tab_palette()
DATA_COLOR = PALETTE.get_color()
BACKGROUND_COLOR = PALETTE.get_color()
SIGNAL_COLOR = PALETTE.get_color()
FIT_COLOR = PALETTE.get_color()
CUT_COLOR = PALETTE.get_color()

class Dataset:
    def __init__(self, name, label, hists, hists_all, events, selections, triggers=None, fraction=None):
        self.name = name
        self.label = label
        self.hists = hists
        self.hists_all = hists_all
        self.events = events
        self.selections = selections
        self.triggers = triggers
        self.fraction = fraction

    @staticmethod
    def load(args, variables, branches, rigidity_binning=None, mc_weighting=None):
        mc_only_variables = []
        if len(args) == 9:
            assert rigidity_binning is not None and mc_weighting is not None
            mc_trigger_filename, dataset_fraction, species = args[6:]
            weighting = mc_weighting.weightings[MC_PARTICLE_IDS[species]]
            mc_trigger_hist = load_weighted_mc_trigger_count(mc_trigger_filename, rigidity_binning, weighting)
            args = args[:6]
            mc_only_variables = ["McAbsRigidity"]
        else:
            mc_trigger_hist = None
            dataset_fraction = 1
        name, label, hist_filename, hist_all_filename, events_filename, selections_filename = args
        hists = Dataset.load_hists(hist_filename, variables + mc_only_variables)
        hists_all = Dataset.load_hists(hist_all_filename, variables + mc_only_variables)
        events = Dataset.load_events(events_filename, branches)
        selections = Dataset.load_selections(selections_filename)
        return Dataset(name, label, hists, hists_all, events, selections, triggers=mc_trigger_hist, fraction=float(dataset_fraction))

    @staticmethod
    def load_hists(filename, variables):
        hists = None
        for filename in glob(filename):
            with np.load(filename) as file:
                temphists = {
                    variable: WeightedHistogram.from_file(file, f"hist_{variable}")
                    for variable in variables
                }
            if hists is None:
                hists = temphists
            else:
                for variable in hists:
                    hists[variable] += temphists[variable]
        return hists

    @staticmethod
    def load_events(filename, branches):
        event_arrays = {var: [] for var in branches}
        for filename in glob(filename):
            with np.load(filename) as file:
                for var in branches:
                    event_arrays[var].append(file[var])
        return np.core.records.fromarrays([np.concatenate(event_arrays[var]) for var in branches], names=branches)

    @staticmethod
    def load_selections(filename_pattern):
        selections = {}
        for filename in glob(filename_pattern):
            with np.load(filename) as selections_file:
                for name in selections_file["selections"]:
                    if name in selections:
                        selections[name].add(Selection.from_file(selections_file, f"selection_{name}"))
                    else:
                        selections[name] = Selection.from_file(selections_file, f"selection_{name}")
        return selections

    def __add__(self, other):
        hists = {key: self.hists[key] + other.hists[key] for key in self.hists}
        hists_all = {key: self.hists_all[key] + other.hists_all[key] for key in self.hists_all}
        events = recfunctions.stack_arrays((self.events, other.events), asrecarray=True)
        selections = merge_selections(self.selections, other.selections)
        trigger_hist = None
        if self.triggers is not None and other.triggers is not None:
            if np.all(self.triggers == other.triggers):
                trigger_hist = self.triggers
            else:
                trigger_hist = self.triggers + other.triggers
        return Dataset(self.name, self.label, hists, hists_all, events, selections, triggers=trigger_hist, fraction=self.fraction + other.fraction)

    def draw_random(self, amount, seed=None):
        rng = np.random.default_rng(seed=seed)
        return Dataset(self.name, self.label, self.hists, self.hists_all, rng.choice(self.events, amount, replace=False), self.selections, triggers=self.triggers, fraction=self.fraction)

    def remove_weights(self):
        new_weights = np.ones_like(self.events.TotalWeight)
        events = recfunctions.rec_append_fields(recfunctions.rec_drop_fields(self.events, ("TotalWeight",)), "TotalWeight", new_weights)
        return Dataset(self.name, self.label, self.hists, self.hists_all, events, self.selections, triggers=self.triggers, fraction=self.fraction)

    def draw_toy(self, size, variables, seed=0):
        branch_values = []
        branch_names = []
        for branch_index, branch in enumerate(variables):
            branch_values.append(draw_random_from_hist(self.hists[branch], size, seed=seed * ANOTHER_PRIME_NUMBER + branch_index).astype(self.events[branch].dtype))
            branch_names.append(branch)
        branch_names.append("TotalWeight")
        branch_values.append(np.ones(size, dtype=np.float64))
        branch_names.append("TotalFlatWeight")
        branch_values.append(np.ones(size, dtype=np.float64))
        branch_names.append("RunNumber")
        branch_values.append(np.zeros(size, dtype=np.uint32))
        branch_names.append("EventNumber")
        branch_values.append(np.zeros(size, dtype=np.uint32))
        events = np.core.records.fromarrays(branch_values, names=branch_names)
        return Dataset(name=f"{self.name} Toy", label=f"{self.label} Toy", hists=self.hists, hists_all=self.hists_all, events=events, selections=self.selections, triggers=self.triggers, fraction=self.fraction)

    def get_most_signal_like(self, predictions, amount):
        sorted_predictions = np.sort(predictions)
        min_pred = sorted_predictions[-amount]
        indices = predictions >= min_pred
        return self.events[indices], predictions[indices]

    def get_most_background_like(self, predictions, amount):
        sorted_predictions = np.sort(predictions)
        max_pred = sorted_predictions[amount]
        indices = predictions <= max_pred
        return self.events[indices], predictions[indices]

    def apply_selections(self, selections, prediction_values=None):
        events = self.events
        for selection in selections:
            passes = events[f"PassesSelection{selection}"]
            events = events[passes]
            if prediction_values is not None:
                prediction_values = prediction_values[passes]
        return Dataset(self.name, self.label, self.hists, self.hists_all, events, self.selections), prediction_values


def predict_bdt(bdt, dataset, variables):
    return bdt.predict(xgb.DMatrix(rec_to_float(filter_branches(dataset.events, variables)), feature_names=variables))

def prediction_hist(dataset, predictions, binning, title):
    return WeightedHistogram.fill_direct((binning,), predictions, weights=dataset.events.TotalWeight, labels=(title,))


def derive_mass_cut(mass_values, weights, signal_values, target_efficiency, precision=1e-4):
    def _calc_efficiency(delta):
        passed = np.any([np.abs(mass_values - signal_value) < delta for signal_value in signal_values], axis=0)
        failed = ~passed
        total_passed = np.sum(weights[passed])
        total_failed = np.sum(weights[failed])
        return total_passed / (total_passed + total_failed)

    cut_delta = 0.1
    step = 0.05
    direction = 1
    while True:
        efficiency = _calc_efficiency(cut_delta)
        if np.abs(efficiency - target_efficiency) < precision:
            return cut_delta
        if efficiency < target_efficiency:
            if direction < 0:
                direction = 1
                step /= 2
            cut_delta += step
        elif efficiency > target_efficiency:
            if direction > 0:
                direction = -1
                step /= 2
            if step > cut_delta:
                step = cut_delta / 2
            cut_delta -= step


def draw_contour(minuit, loss, fit_parameters, plotdir, prefix, title, dof=2, sample_size=None):
    print(f"calculating contour", flush=True)
    contour_figure = plt.figure(figsize=(12, 6.15))
    contour_figure.suptitle(f"{title} Template Fit Contour")
    contour_plot = contour_figure.subplots(1, 1)
    contour_plot.set_xlabel("Signal Events")
    contour_plot.set_ylabel("Background Events")

    signal_value = max(minuit.values["signal_counts"], 0)
    background_value = max(minuit.values["background_counts"], 0)
    if minuit.valid:
        signal_error_lower = -minuit.merrors["signal_counts"].lower
        signal_error_upper = minuit.merrors["signal_counts"].upper
        background_error_lower = -minuit.merrors["background_counts"].lower
        background_error_upper = minuit.merrors["background_counts"].upper
    else:
        signal_error_lower = -minuit.errors["signal_counts"]
        signal_error_upper = minuit.errors["signal_counts"]
        background_error_lower = -minuit.errors["background_counts"]
        background_error_upper = minuit.errors["background_counts"]

    contour_points = []
    contour_levels = []
    for sigmas in range(1, 4):
        cl = chi2(dof).ppf(norm.cdf(sigmas) - norm.cdf(-sigmas))
        points = calculate_contour(loss, (fit_parameters["signal_counts"], fit_parameters["background_counts"]), cl)
        points.append(points[0])
        points = np.array(points)
        negative_points = points[:,0] < 0
        points[:,0][negative_points] = points[:,0][negative_points] / (1 - points[:,0][negative_points])
        contour_points.append(points)
        contour_levels.append(sigmas)
    contour_points = np.array(contour_points)
    contour_levels = np.array(contour_levels)

    min_signal = -1
    max_signal = max(signal_value + 6 * max(signal_error_upper, 1), np.max(contour_points[:,:,0]))
    min_background = background_value - 6 * background_error_lower
    max_background = background_value + 6 * background_error_upper
    if np.isnan(min_signal) or np.isnan(max_signal) or np.isnan(min_background) or np.isnan(max_background):
        print("Warning: Contour plot limits are nan, not plotting.")
        return
    contour_plot.set_xlim(min_signal, max_signal)
    contour_plot.set_ylim(min_background, max_background)

    contour_plot.errorbar([signal_value], [background_value], [[background_error_lower], [background_error_upper]], [[signal_error_lower], [signal_error_upper]], fmt="o", label="Best Fit")
    contour_plot.axvline(0, color="gray", alpha=0.5)
    contour_plot.add_patch(Rectangle((min_signal, min_background), -min_signal, max_background - min_background, linewidth=0, fill=False, hatch="//"))

    if sample_size is not None:
        sample_size_x = np.arange(max_signal + 1)
        sample_size_y = sample_size - sample_size_x
        contour_plot.plot(sample_size_x, sample_size_y, "--", color="gray", label="Sample Size")

    contour = ContourSet(contour_plot, contour_levels, contour_points[:,np.newaxis])
    plt.clabel(contour, fmt=lambda s: f"{s}σ")

    max_signal = max(np.max(contour_points[:,:,0]) * 1.1, max_signal)
    if np.isfinite(min_signal) and np.isfinite(max_signal):
        contour_plot.set_xlim(min_signal, max_signal)
    else:
        print("Warning: Invalid limits {min_signal!r} and {max_signal!r}")
    contour_plot.legend()
    save_figure(contour_figure, plotdir, f"{prefix}_contour")

    return


def draw_template_fit(data_hist, signal_template, background_template, template_binning, fit_parameters, fit_param_dict, fit_valid, title, chisq, dof, rchisq, plotdir, prefix, subtitle=None, data_label="Data", signal_template_label="Signal Template", background_template_label="Background Template", n_hypothetical=1, log=True, postfix=None, show_empty_components=True, additional_labels=None, show_errors_on_model=False, label_x=None, target_efficiency=None, target_cut_value=None):
    data_color = DATA_COLOR
    background_color = BACKGROUND_COLOR
    signal_color = SIGNAL_COLOR
    fit_color = FIT_COLOR
    cut_color = CUT_COLOR

    data_hist_bins = len(data_hist.binnings[0]) - 2
    template_bins = len(template_binning) - 2
    template_binning_factor = template_bins / data_hist_bins
    figure = plt.figure(figsize=(12, 6.15))
    plot = figure.subplots(1, 1)
    figure.suptitle(title)
    if subtitle is not None:
        plot.set_title(subtitle)
    fit_label_postfix = "" if fit_valid else " INVALID"

    plot_histogram_1d(plot, data_hist, style="iss", log=log, label=f"{data_label} ({data_hist.values.sum():.0f} Events)", color=data_color, use_approximate_poisson_errors=True, draw_zeros=False)
    plot_steps(plot, template_binning.edges, signal_template * fit_param_dict["signal_counts"] * template_binning_factor, label=f"{fit_parameters['signal_counts']:P} $\\times$ {signal_template_label}", color=signal_color)
    plot_steps(plot, template_binning.edges, background_template * fit_param_dict["background_counts"] * template_binning_factor, label=f"{fit_parameters['background_counts']:P} $\\times$ {background_template_label}", color=background_color)
    plot_steps(plot, template_binning.edges, (signal_template * fit_param_dict["signal_counts"] + background_template * fit_param_dict["background_counts"]) * template_binning_factor, label=f"Fit $\\chi^2={chisq:.1f}/{dof:.0f}={rchisq:.2f}${fit_label_postfix}", alpha=0.9, color=fit_color)
    if show_errors_on_model:
        shaded_steps(plot, template_binning.edges, values=background_template * fit_param_dict["background_counts"] * template_binning_factor, errors=np.sqrt(background_template * fit_param_dict["background_counts"] * template_binning_factor), alpha=0.5, color=background_color)
    plot_steps(plot, template_binning.edges, signal_template * template_binning_factor * n_hypothetical, linestyle="dotted", label=f"Distribution for {n_hypothetical:.0f} signal events", color=signal_color)
    if target_efficiency is not None and target_cut_value is not None:
        #print(f"{100*target_efficiency:.0f}% cut value: {target_cut_value:.1f}")
        plot.axvline(target_cut_value, color=cut_color, alpha=0.85, label=f"{100*target_efficiency:.0f}% efficient cut", linewidth=1, linestyle="dashed")

    if additional_labels is not None:
        for label in additional_labels:
            plot.plot([], [], label=label, color="white")

    if label_x is not None:
        plot.set_xlabel(label_x)

    plot.set_xlim(data_hist.binnings[0].edges[1], data_hist.binnings[0].edges[-2])
    if log:
        plot.set_ylim(0.5, plot.get_ylim()[1])
    else:
        plot.set_ylim(0, plot.get_ylim()[1])
    plot.legend()
    postfix = postfix or ""
    if postfix:
        postfix = f"_{postfix}"
    save_figure(figure, plotdir, f"{prefix}_template_fit{postfix}")


def integrate_events_below_cut_value(prediction_values, cut_value):
    return np.sum(prediction_values <= cut_value)

def integrate_events_above_cut_value(prediction_values, cut_value, weights=None):
    if weights is None:
        weights = np.ones_like(prediction_values)
    return np.sum(weights[prediction_values > cut_value])

def integrate_histogram_below_cut_value(hist, cut_value):
    max_bin = hist.binnings[0].get_indices(np.array(cut_value)).item()
    return hist.values[:max_bin+1].sum()

def integrate_histogram_above_cut_value(hist, cut_value):
    min_bin = hist.binnings[0].get_indices(np.array(cut_value)).item()
    return hist.values[min_bin:].sum()


def make_kde_pdf(signal_pdf, background_pdf):
    def _kde_pdf(arg_points, signal_counts, background_counts):
        penalty = np.exp(np.minimum(signal_counts / np.maximum(2 * background_counts, 1), 0))
        p = signal_pdf(arg_points) * np.maximum(signal_counts, 0) + background_pdf(arg_points) * np.maximum(background_counts, 0)
        return np.maximum(signal_counts, 0) + np.maximum(background_counts, 0), p * penalty
    return _kde_pdf


def make_unmodified_kde_pdf(signal_pdf, background_pdf, min_p=1e-16):
    pdf_test_points = np.linspace(-25, 25, 10000)
    s_pdf_val = signal_pdf(pdf_test_points)
    b_pdf_val = background_pdf(pdf_test_points)
    pdf_ratio = b_pdf_val / np.maximum(s_pdf_val, min_p)
    def _kde_pdf(arg_points, signal_counts, background_counts):
        p = np.maximum(signal_pdf(arg_points) * signal_counts + background_pdf(arg_points) * background_counts, min_p)
        sb_ratio = signal_counts / background_counts
        g0sel = sb_ratio > -pdf_ratio
        ptest = signal_counts * s_pdf_val + background_counts * b_pdf_val
        corrf = ptest.sum() / (ptest[g0sel]).sum()
        #norm, sig_norm = quad(lambda x: np.maximum(signal_pdf(x) * signal_counts + background_pdf(x) * background_counts, min_p), -50, 50, limit=1000)
        #corr = (signal_counts + background_counts) / norm
        #print(signal_counts, background_counts, signal_counts + background_counts, norm, sig_norm, corr, corrf)
        return signal_counts + background_counts, p * corrf
    return _kde_pdf


def make_kde_pdf_with_deficit(signal_pdf, background_pdf):
    def _kde_pdf(arg_points, signal_counts, background_counts):
        p = signal_pdf(arg_points) * signal_counts + background_pdf(arg_points) * background_counts
        return np.max(signal_counts + background_counts, 0), np.maximum(p, 0)
    return _kde_pdf


def make_kde_pdf_with_limited_deficit(signal_pdf, background_pdf, test_points):
    def _kde_pdf(arg_points, signal_counts, background_counts):
        p_sig = signal_pdf(arg_points)
        p_bkg = background_pdf(arg_points)
        p_sig_test = signal_pdf(test_points)
        p_bkg_test = background_pdf(test_points)
        n_sig_min = np.max((-background_counts * p_bkg_test[p_sig_test > 0] / p_sig_test[p_sig_test > 0]))
        n_bkg_min = np.max((-signal_counts * p_sig_test[p_bkg_test > 0] / p_bkg_test[p_bkg_test > 0]))
        penalty = np.exp(np.minimum((signal_counts - n_sig_min) / np.maximum(2 * background_counts, 1), 0)) #* np.exp(np.minimum((background_counts - n_bkg_min) / np.maximum(2 * signal_counts, 1), 0))
        p_total = p_sig * np.maximum(signal_counts, n_sig_min) + p_bkg * np.maximum(background_counts, n_bkg_min)
        if not np.all(np.isfinite(p_total)) or not np.all(np.isfinite(penalty)):
            print("Loss not finite:", p_total, penalty, p_sig, p_bkg, n_sig_min, n_bkg_min, signal_counts, background_counts)
        return np.maximum(signal_counts, n_sig_min) + np.maximum(background_counts, n_bkg_min), np.maximum(p_total, 1e-10) * penalty
    return _kde_pdf
        

def select_kde_bandwidth(histogram, predictions, weights, tolerance=0.01, min_bin_width_ratio=0.25, max_bin_width_ratio=1.0):
    print(f"{histogram.values.sum()} entries in template histogram")
    nonzero = histogram.values > 0
    scale = 1 / histogram.values.sum()
    bin_width = np.mean(histogram.binnings[0].edges[2:-1] - histogram.binnings[0].edges[1:-2])
    step = 0.1
    kde = gaussian_kde(predictions, weights=weights)
    print("f", kde.factor, "c", kde.covariance)
    bandwidth_guess = kde.factor
    bandwidth = kde.factor
    direction = None
    cov_factor = np.sqrt(kde.covariance[0, 0])
    print("bw guess", bandwidth, cov_factor)
    for _ in range(100):
        kde.set_bandwidth(bandwidth)
        kde_template = make_kde_template(kde, histogram.binnings[0])
        chisq, dof, rchisq = calculate_chisq(data=histogram.values * scale, model=kde_template, errors=histogram.get_errors() * scale, n_parameters=1)
        #residuals = (histogram.values[nonzero] * scale - kde_template[nonzero])**2 / (histogram.get_errors()[nonzero] * scale)**2
        #rchisq = residuals.sum() / (nonzero.sum() - 1)
        if abs(rchisq - 1) <= tolerance:
            print(f"Optimal bandwidth {bandwidth:.4f}")
            if bandwidth * cov_factor < min_bin_width_ratio * bin_width:
                bandwidth = min_bin_width_ratio * bin_width / cov_factor
                print(f"Bandwidth is below minimum ratio to bin width, setting to {bandwidth:.4f}")
            if bandwidth * cov_factor > max_bin_width_ratio * bin_width:
                bandwidth = max_bin_width_ratio * bin_width / cov_factor
                print(f"Bandwidth is above maximum ratio to bin width, setting to {bandwidth:.4f}")
            return bandwidth
        if rchisq < 1:
            bandwidth += step
            if direction is not None and direction < 0:
                step /= 2
            direction = 1
        elif rchisq > 1:
            bandwidth -= step
            if direction is not None and direction > 0:
                step /= 2
            elif bandwidth <= 2 * step:
                step = bandwidth / 2
            direction = -1
    print("Failed to fit KDE bandwidth!")
    return bandwidth_guess


def combine_pdfs(*pdfs):
    def _combined_pdf(arg_points, signal_counts, background_counts):
        norm_all = 0
        p_all = 1
        for index, pdf in enumerate(pdfs):
            norm, p = pdf(arg_points[:,index], signal_counts, background_counts)
            norm_all += norm
            p_all *= p
        return norm_all / len(pdfs), p_all
    return _combined_pdf


def make_templates_for_variable(signal_events, background_events, variable, nbins=100, fine_binning_factor=10, signal_template_smoothing_window=3):
    min_value = decrease_value(min(np.min(signal_events[variable]), np.min(background_events[variable])))
    max_value = increase_value(max(np.max(signal_events[variable]), np.max(background_events[variable])))
    var_binning = make_lin_binning(min_value, max_value, nbins)
    fine_var_binning = make_lin_binning(min_value, max_value, nbins * fine_binning_factor)

    signal_template_hist = WeightedHistogram.fill_direct((var_binning,), signal_events[variable], weights=signal_events.TotalWeight, labels=(f"{variable}",))
    signal_template_fine_hist = WeightedHistogram.fill_direct((fine_var_binning,), signal_events[variable], weights=signal_events.TotalWeight, labels=(f"{variable}",))
    background_template_hist = WeightedHistogram.fill_direct((var_binning,), background_events[variable], weights=background_events.TotalWeight, labels=(f"{variable}",))
    background_template_fine_hist = WeightedHistogram.fill_direct((fine_var_binning,), background_events[variable], weights=background_events.TotalWeight, labels=(f"{variable}",))

    signal_template = smooth_additive(signal_template_fine_hist.values / signal_template_fine_hist.values.sum(), window=signal_template_smoothing_window)
    coarse_signal_template = signal_template_hist.values / signal_template_hist.values.sum()
    background_kde = gaussian_kde(background_events[variable], weights=background_events.TotalWeight)
    background_template = make_kde_template(background_kde, fine_var_binning)
    coarse_background_template = make_kde_template(background_kde, var_binning)

    def _signal_pdf(points):
        indices = fine_var_binning.get_indices(points)
        return signal_template[indices] / fine_var_binning.bin_widths[indices]

    def _background_pdf(points):
        indices = fine_var_binning.get_indices(points)
        return background_template[indices] / fine_var_binning.bin_widths[indices]

    def _fit_function(signal_counts, background_counts):
        return signal_template * np.maximum(signal_counts, 0) + background_template * np.maximum(background_counts, 0)

    def _coarse_fit_function(signal_counts, background_counts):
        return coarse_signal_template * np.maximum(signal_counts, 0) + coarse_background_template * np.maximum(background_counts, 0)

    return var_binning, fine_var_binning, signal_template, signal_template_hist, background_template, background_template_hist, _signal_pdf, _background_pdf, _fit_function, _coarse_fit_function


def data_derived_secondary_fit(positive_data, negative_data, positive_predictions, negative_predictions, variables, n_candidates, binnings, plotdir, prefix, title):
    print("Performing candidate fits with data derived templates")
    n_background = len(negative_data.events) // 2
    print(len(negative_data.events), n_background, n_candidates)
    if n_candidates > len(negative_data.events) - n_background:
        n_candidates = len(negative_data.events) - n_background
    assert len(negative_data.events) - n_background >= n_candidates
    negative_candidate_data, negative_candidate_predictions = negative_data.get_most_signal_like(negative_predictions, n_candidates)
    negative_background_data, negative_background_predictions = negative_data.get_most_background_like(negative_predictions, n_background)

    result_data = {}

    for variable in variables:
        print(variable)
        fine_binning_factor = 10
        try:
            var_binning, fine_var_binning, signal_template, signal_template_hist, background_template, background_template_hist, _signal_pdf, _background_pdf, _fit_function, _coarse_fit_function = make_templates_for_variable(positive_data.events, negative_background_data, variable, fine_binning_factor=fine_binning_factor)
        except np.linalg.LinAlgError:
            continue

        candidate_hist = WeightedHistogram.fill_direct((var_binning,), negative_candidate_data[variable], weights=negative_candidate_data.TotalWeight, labels=(f"{variable}",))

        signal_template_figure = plt.figure(figsize=(12, 6.15))
        signal_template_figure.suptitle(f"{title}")
        signal_template_plot = signal_template_figure.subplots(1, 1)
        signal_template_plot.set_title(f"{variable} Signal Template")
        plot_histogram_1d(signal_template_plot, signal_template_hist, style="iss", label=f"{positive_data.label} Positive", log=True)
        plot_steps(signal_template_plot, fine_var_binning.edges, signal_template * signal_template_hist.values.sum() * fine_binning_factor, label="Signal Template")
        #ylim_max = round_up(np.max((signal_template_hist.values + signal_template_hist.get_errors()) * 1.1))
        #ylim_min = 0 #ylim_max / 1e4
        #signal_template_plot.set_ylim(ylim_min, ylim_max)
        signal_template_plot.set_ylim(bottom=0)
        signal_template_plot.legend()
        save_figure(signal_template_figure, plotdir, f"{prefix}_{variable}_signal_template")

        background_template_figure = plt.figure(figsize=(12, 6.15))
        background_template_figure.suptitle(f"{title}")
        background_template_plot = background_template_figure.subplots(1, 1)
        background_template_plot.set_title(f"{variable} Background Template")
        plot_histogram_1d(background_template_plot, background_template_hist, style="iss", label=f"{negative_data.label} Background", log=True)
        plot_steps(background_template_plot, fine_var_binning.edges, background_template * background_template_hist.values.sum() * fine_binning_factor, label="Background Template")
        #ylim_max = round_up(np.max(background_template_hist.values + background_template_hist.get_errors()))
        #ylim_min = 0 #ylim_max / 1e4
        #background_template_plot.set_ylim(ylim_min, ylim_max)
        background_template_plot.set_ylim(bottom=0)
        background_template_plot.legend()
        save_figure(background_template_figure, plotdir, f"{prefix}_{variable}_background_template")

        _kde_pdf = make_kde_pdf(_signal_pdf, _background_pdf)

        guess = dict(signal_counts=1, background_counts=len(negative_candidate_data) - 1)
        loss = ExtendedUnbinnedNLL(negative_candidate_data[variable], _kde_pdf)
        m = Minuit(loss, **guess)
        m.migrad()
        if m.valid:
            m.minos()
        print(m)

        fit_parameters = dict(zip(m.parameters, uncertainties.correlated_values(m.values, np.array(m.covariance))))
        fit_param_names = m.parameters
        fit_param_values = m.values
        fit_param_errors = m.errors
        fit_param_dict = dict(zip(fit_param_names, fit_param_values))
        fit_param_error_dict = dict(zip(fit_param_names, fit_param_errors))
        fit_values = _fit_function(**fit_param_dict)
        coarse_fit_values = _coarse_fit_function(**fit_param_dict)
        chisq, dof, rchisq = calculate_chisq(data=candidate_hist.values, model=coarse_fit_values, errors=np.sqrt(coarse_fit_values), n_parameters=len(guess))
        #nonzero = candidate_hist.values > 0
        #residuals = ((candidate_hist.values[nonzero] - coarse_fit_values[nonzero]) / candidate_hist.get_errors()[nonzero])**2
        #chisq = residuals.sum()
        #dof = nonzero.sum() - len(guess)
        #rchisq = chisq / dof
        print(f"chi² = {chisq:.1f} / {dof:.0f} = {rchisq:.2f}")

        result_data[variable] = dict(fit_parameters=dict(values=fit_param_dict, errors=fit_param_error_dict), chisq=dict(chisq=chisq, dof=int(dof), rchisq=rchisq))

        draw_template_fit(data_hist=candidate_hist, signal_template=signal_template, background_template=background_template, template_binning=fine_var_binning, fit_parameters=fit_parameters, fit_param_dict=fit_param_dict, fit_valid=m.valid, title=f"{title} Template Fit", chisq=chisq, dof=dof, rchisq=rchisq, plotdir=plotdir, prefix=f"{prefix}_{variable}", data_label=f"Most signal like negative events")

    return result_data


def perform_isotope_fits(data_events, data_predictions, signal_template_datasets, background_template_dataset, isotope_variables, binnings, plotdir, prefix, title, data_label, mva_templates=None):
    for variable in isotope_variables:
        print(variable)
        binning = binnings.variable_binnings[variable]

        signal_template_hists = {}
        for name, dataset in signal_template_datasets.items():
            values = dataset.events[variable]
            selection = (values >= binning.edges[1]) & (values <= binning.edges[-2])
            if not np.all(selection):
                removed = np.sum(~selection)
                total = len(selection)
                print(f"WARNING: Removing {removed}/{total} out-of-range events in signal template dataset {name}:")
                print(values[~selection])
            values = values[selection]
            weights = dataset.events.TotalWeight[selection]
            signal_template_hists[name] = WeightedHistogram.fill_direct((binning,), values, weights=weights, labels=(variable,))

        background_template_values = background_template_dataset.events[variable]
        background_template_selection = (background_template_values >= binning.edges[1]) & (background_template_values <= binning.edges[-2])
        if not np.all(background_template_selection):
            removed = np.sum(~background_template_selection)
            total = len(background_template_selection)
            print(f"WARNING: Removing {removed}/{total} out-of-range events in background template dataset:")
            print(background_template_values[~background_template_selection])
        background_template_values = background_template_values[background_template_selection]
        background_template_weights = background_template_dataset.events.TotalWeight[background_template_selection]
        background_template_hist = WeightedHistogram.fill_direct((binning,), background_template_values, weights=background_template_weights, labels=(variable,))

        background_kde = gaussian_kde(background_template_values, weights=background_template_weights)
        background_template = make_kde_template(background_kde, binning)
        signal_templates = {name: smooth_additive(hist.values / hist.values.sum(), window=2) for name, hist in signal_template_hists.items()}
        signal_template_names = [name for name in signal_template_hists]

        data_values = data_events[variable]
        data_selection = (data_values >= binning.edges[1]) & (data_values <= binning.edges[-2])
        if not np.all(data_selection):
            removed = np.sum(~data_selection)
            total = len(data_selection)
            print(f"WARNING: Removing {removed}/{total} out-of-range events in data sample:")
            print(data_values[~data_selection])
        data_values = data_values[data_selection]
        data_weights = data_events.TotalWeight[data_selection]
        predictions = data_predictions[data_selection]

        data_hist = WeightedHistogram.fill_direct((binning,), data_values, weights=data_weights, labels=(variable,))

        print(f"Isotope KDE bandwidth: {background_kde.factor:.2f}")
        print(background_kde.neff, background_kde.n)
        print(background_kde.covariance)

        def _make_signal_pdf(name):
            def _signal_pdf(points):
                indices = binning.get_indices(points)
                return signal_templates[name][indices] / binning.bin_widths[indices]
            return _signal_pdf

        signal_pdfs = {name: _make_signal_pdf(name) for name in signal_template_names}

        def _background_pdf(points):
            indices = binning.get_indices(points)
            return background_template[indices] / binning.bin_widths[indices]

        def _isotope_fit_function(background_counts, *signal_counts, **kwargs):
            signal_count_dict = {f"{signal_template_name}_signal_counts": signal_count for (signal_template_name, signal_count) in zip(signal_template_names, signal_counts)}
            for key, value in kwargs.items():
                signal_count_dict[key] = value
            result = np.maximum(background_counts, 0) * background_template
            for signal_template_name, signal_template in signal_templates.items():
                result += np.maximum(signal_count_dict[f"{signal_template_name}_signal_counts"], 0) * signal_template
            return result

        def _isotope_pdf(arg_points, background_counts, *signal_counts):
            p = np.maximum(background_counts, 0) * _background_pdf(arg_points)
            penalty_norm = np.maximum(np.maximum(2 * background_counts, 0) + np.sum(np.maximum(2 * signal_counts, 0)), 1)
            penalty = np.exp(np.minimum(background_counts / penalty_norm, 0))
            for isotope_signal_counts, signal_template_name in zip(signal_counts, signal_template_names):
                signal_pdf = signal_pdfs[signal_template_name]
                p += np.maximum(isotope_signal_counts, 0) * signal_pdf(arg_points)
                penalty *= np.exp(np.minimum(isotope_signal_counts / penalty_norm, 0))
            total = np.maximum(background_counts, 0) + sum((np.maximum(counts, 0) for counts in signal_counts))
            return total, p * penalty

        _isotope_pdf.func_code = make_func_code(["arg_points", "background_counts"] + [f"{signal_template_name}_signal_counts" for signal_template_name in signal_template_names])

        isotope_loss = ExtendedUnbinnedNLL(data_values, _isotope_pdf)
        loss = isotope_loss

        if mva_templates is not None:
            print("Using MVA for 2D isotope template fit")
            mva_signal_template, mva_background_template, mva_binning = mva_templates
            def _mva_signal_pdf(points):
                indices = mva_binning.get_indices(points)
                return mva_signal_template[indices] / mva_binning.bin_widths[indices]
            def _mva_background_pdf(points):
                indices = mva_binning.get_indices(points)
                return mva_background_template[indices] / mva_binning.bin_widths[indices]
            def _mva_pdf(arg_points, background_counts, *signal_counts):
                p = np.maximum(background_counts, 0) * _mva_background_pdf(arg_points)
                penalty_norm = np.maximum(np.maximum(background_counts, 0) + np.sum(np.maximum(signal_counts, 0)), 1)
                penalty = np.exp(np.minimum(background_counts / penalty_norm, 0))
                for isotope_signal_counts, signal_template_name in zip(signal_counts, signal_template_names):
                    p += np.maximum(isotope_signal_counts, 0) * _mva_signal_pdf(arg_points)
                    penalty *= np.exp(np.minimum(isotope_signal_counts / penalty_norm, 0))
                total = np.maximum(background_counts, 0) + sum((np.maximum(counts, 0) for counts in signal_counts))
                return total, p * penalty
            _mva_pdf.func_code = make_func_code(["arg_points", "background_counts"] + [f"{signal_template_name}_signal_counts" for signal_template_name in signal_template_names])
            mva_loss = ExtendedUnbinnedNLL(predictions, _mva_pdf)
            loss = isotope_loss + mva_loss

        default_guess = np.sum(data_weights) / (len(signal_template_datasets) + 1)
        guess = dict(background_counts=default_guess)
        for signal_template_name in signal_template_names:
            guess[f"{signal_template_name}_signal_counts"] = default_guess
        miso = Minuit(loss, **guess)
        miso.migrad()
        if miso.valid:
            miso.minos()
        print(miso)

        if np.any(np.isnan(miso.values)) or np.any(np.isnan(miso.covariance)):
            print(f"Error: Fit returned NaN result: {miso.values} {miso.covariance}")
            continue

        fit_parameters = dict(zip(miso.parameters, uncertainties.correlated_values(miso.values, np.array(miso.covariance))))
        fit_param_names = miso.parameters
        fit_param_values = miso.values
        fit_param_errors = miso.errors
        fit_param_dict = dict(zip(fit_param_names, fit_param_values))
        fit_param_error_dict = dict(zip(fit_param_names, fit_param_errors))
        fit_values = _isotope_fit_function(**fit_param_dict)

        chisq, dof, rchisq = calculate_chisq(data=data_hist.values, model=fit_values, errors=np.sqrt(fit_values), n_parameters=len(guess))
        #nonzero = data_hist.values > 0
        #residuals = ((data_hist.values[nonzero] - fit_values[nonzero]) / data_hist.get_errors()[nonzero])**2
        #chisq = residuals.sum()
        #dof = nonzero.sum() - len(guess)
        #rchisq = chisq / dof

        isotope_fit_figure = plt.figure(figsize=(12, 6.15))
        isotope_fit_figure.suptitle(title)
        isotope_fit_plot = isotope_fit_figure.subplots(1, 1)
        plot_histogram_1d(isotope_fit_plot, data_hist, style="iss", log=True, label=data_label)
        for signal_template_name, signal_template in zip(signal_template_names, signal_templates.values()):
            var_name = f"{signal_template_name}_signal_counts"
            signal_label = signal_template_datasets[signal_template_name].label
            plot_steps(isotope_fit_plot, binning.edges, signal_template * fit_param_dict[var_name], label=f"{fit_parameters[var_name]:P} $\\times$ {signal_label} Positive")
        plot_steps(isotope_fit_plot, binning.edges, background_template * fit_param_dict["background_counts"], label=f"{fit_parameters['background_counts']:P} $\\times$ {background_template_dataset.label} Negative")
        fit_label_postfix = "" if miso.valid else " INVALID"
        plot_steps(isotope_fit_plot, binning.edges, fit_values, label=f"Fit $\\chi^2={chisq:.1f}/{dof:.0f}={rchisq:.2f}${fit_label_postfix}", alpha=0.9)
        isotope_fit_plot.set_ylim(1e-3, isotope_fit_plot.get_ylim()[1])
        isotope_fit_plot.legend()
        save_figure(isotope_fit_figure, plotdir, f"{prefix}_{variable}_fit")

        for signal_template_name, signal_template_hist in signal_template_hists.items():
            isotope_signal_template_figure = plt.figure(figsize=(12, 6.15))
            isotope_signal_template_figure.suptitle(f"{title} Signal Template {signal_template_name}")
            isotope_signal_template_plot = isotope_signal_template_figure.subplots(1, 1)
            signal_label = signal_template_datasets[signal_template_name].label
            plot_histogram_1d(isotope_signal_template_plot, signal_template_hist, style="mc", log=True, scale=1 / signal_template_hist.values.sum(), label=signal_label)
            plot_steps(isotope_signal_template_plot, binning.edges, signal_templates[signal_template_name], label="Template")
            isotope_signal_template_plot.legend()
            save_figure(isotope_signal_template_figure, plotdir, f"{prefix}_{variable}_template_signal_{signal_template_name}")

        isotope_background_template_figure = plt.figure(figsize=(12, 6.15))
        isotope_background_template_figure.suptitle(f"{title} Background Template")
        isotope_background_template_plot = isotope_background_template_figure.subplots(1, 1)
        background_label = background_template_dataset.label
        plot_histogram_1d(isotope_background_template_plot, background_template_hist, style="mc", log=True, scale=1 / background_template_hist.values.sum(), label=background_label)
        plot_steps(isotope_background_template_plot, binning.edges, background_template, label="Template")
        isotope_background_template_plot.legend()
        save_figure(isotope_background_template_figure, plotdir, f"{prefix}_{variable}_template_background")





def perform_template_fit(negative_data, positive_data, signal_template_data, background_template_data, mva_binning, bdt, variables, plotdir="plots", prefix="template_fit", title="TemplateFit", rig_title="", bdt_title="BDT", smooth=False, method="llh", use_kde=False, kde_bandwidth=None, do_contour=False, template_cache=None, attempt_2d_fit=None, efficiency_data=None, target_efficiency=None, allow_deficit=False, signal_template_label="MC $R>0$", background_template_label="MC $R<0$", data_label="Data", prediction_cache=None, store_templates=True, reduce_number_of_plots=False):
    if prediction_cache is None:
        prediction_cache = {}

    def _get_from_cache(key, function):
        if key in prediction_cache:
            return prediction_cache[key]
        else:
            result = function()
            prediction_cache[key] = result
            return result
    data_predictions = _get_from_cache("data_predictions", lambda: predict_bdt(bdt, negative_data, variables))
    data_hist = _get_from_cache("data_hist", lambda: prediction_hist(negative_data, data_predictions, mva_binning, bdt_title))
    positive_predictions = _get_from_cache("positive_predictions", lambda: predict_bdt(bdt, positive_data, variables))
    positive_data_hist = _get_from_cache("positive_data_hist", lambda: prediction_hist(positive_data, positive_predictions, mva_binning, bdt_title))
    signal_template_predictions = _get_from_cache("signal_template_predictions", lambda: predict_bdt(bdt, signal_template_data, variables))
    background_template_predictions = _get_from_cache("background_template_predictions", lambda: predict_bdt(bdt, background_template_data, variables))
    signal_template_hist = _get_from_cache("signal_template_hist", lambda: prediction_hist(signal_template_data, signal_template_predictions, mva_binning, bdt_title))
    background_template_hist = _get_from_cache("background_template_hist", lambda: prediction_hist(background_template_data, background_template_predictions, mva_binning, bdt_title))

    positive_data_event_count = float(positive_data.events.TotalWeight.sum())
    print("positive data", positive_data_event_count, positive_data_hist.values.sum())

    data_counts = np.stack((data_hist.values, data_hist.squared_values), axis=1)
    bin_edges = data_hist.binnings[0].edges
    signal_template = signal_template_hist.values / signal_template_hist.values.sum()
    signal_template_original_hist = signal_template_hist
    background_template = background_template_hist.values / background_template_hist.values.sum()
    background_template_original_hist = background_template_hist
    lsq_mask = data_hist.values > 0
    plot_template_mva_binning = mva_binning

    result_data = {}

    result_data["positive_events"] = positive_data_event_count

    cut_value_signal = None
    count_limit_cut = None
    ratio_limit_cut = None
    positive_events_in_signal_region = None
    negative_events_in_signal_region = None

    if efficiency_data is not None and target_efficiency is not None:
        result_data["cut_limit"] = {}
        signal_cut = target_efficiency
        background_cut = 0.99
        cut_value_signal = efficiency_data["target_efficiency_data"][signal_cut]["cut_value"]
        cut_value_background = efficiency_data["target_efficiency_data"][background_cut]["cut_value"]

        mc_to_iss_factor = integrate_events_below_cut_value(data_predictions, cut_value_background) / integrate_events_below_cut_value(background_template_predictions, cut_value_background)
        mc_events_in_signal_region = integrate_events_above_cut_value(background_template_predictions, cut_value_signal)
        negative_events_in_signal_region = integrate_events_above_cut_value(data_predictions, cut_value_signal)
        positive_events_in_signal_region = integrate_events_above_cut_value(positive_predictions, cut_value_signal, weights=positive_data.events.TotalWeight)
        mc_events_in_signal_region_binned = integrate_histogram_above_cut_value(background_template_hist, cut_value_signal)
        negative_events_in_signal_region_binned = integrate_histogram_above_cut_value(data_hist, cut_value_signal)
        positive_events_in_signal_region_binned = integrate_histogram_above_cut_value(positive_data_hist, cut_value_signal)
        print("MC/ISS factor:", mc_to_iss_factor)
        print(f"N_data = {negative_events_in_signal_region}, N_mc = {mc_events_in_signal_region}, alpha = {mc_to_iss_factor}")
        stats = WStatCountsStatistic(n_on=negative_events_in_signal_region, n_off=mc_events_in_signal_region, alpha=mc_to_iss_factor)
        count_excess_cut = max(stats.n_sig, 0)
        count_limit_cut = count_excess_cut + stats.compute_errp(2)
        ratio_limit_cut = count_limit_cut / positive_events_in_signal_region
        print(f"Cut Excess {count_excess_cut:.2f} Limit ({signal_cut:.2f}, {background_cut:.2f}) = {count_limit_cut:.2f} / {positive_events_in_signal_region:.0f} = {ratio_limit_cut:.2g}")

        result_data["cut_limit"][f"{signal_cut},{background_cut}"] = dict(negative_events=float(negative_events_in_signal_region), positive_events=float(positive_events_in_signal_region), mc_events=float(mc_events_in_signal_region), mc_to_iss_factor=float(mc_to_iss_factor), cut_value_signal=float(cut_value_signal), cut_value_background=float(cut_value_background), limit_count=float(count_limit_cut), limit_ratio=float(ratio_limit_cut))


    if template_cache is None:
        template_cache = {}

    fine_mva_binning_factor = 10
    fine_mva_binning = make_lin_binning(mva_binning.edges[1], mva_binning.edges[-2], fine_mva_binning_factor * (len(mva_binning.edges) - 3))
    reduced_mva_binning = make_lin_binning(mva_binning.edges[1], mva_binning.edges[-2], 12)
    #fine_mva_binning = mva_binning
    if smooth:
        signal_template = smooth_additive(signal_template, window=2)
        background_template = smooth_additive(background_template, window=2)
    if use_kde:
        if kde_bandwidth is None:
            print("Fitting KDE bandwidth", flush=True)
            kde_bandwidth = select_kde_bandwidth(background_template_original_hist, background_template_predictions, background_template_data.events.TotalWeight)
        template_cache_key = (kde_bandwidth,)
        if template_cache_key not in template_cache:
            background_kde = gaussian_kde(background_template_predictions, bw_method=kde_bandwidth, weights=background_template_data.events.TotalWeight)
            print("KDE bandwidth:", kde_bandwidth)
            result_data["kde_bandwidth"] = kde_bandwidth
            print(f"Making background template from {len(background_template_predictions)} events")
            coarse_kde_background_template = make_kde_template(background_kde, mva_binning)
            kde_background_template = make_kde_template(background_kde, fine_mva_binning)
            reduced_kde_background_template = make_kde_template(background_kde, reduced_mva_binning)

            print(f"Making signal template from {len(signal_template_predictions)} events")
            signal_template_fine_hist = WeightedHistogram.fill_direct((fine_mva_binning,), signal_template_predictions, weights=signal_template_data.events.TotalWeight, labels=(bdt_title,))
            signal_template_reduced_hist = WeightedHistogram.fill_direct((reduced_mva_binning,), signal_template_predictions, weights=signal_template_data.events.TotalWeight, labels=(bdt_title,))
            kde_signal_template = smooth_additive(signal_template_fine_hist.values / signal_template_fine_hist.values.sum(), window=fine_mva_binning_factor // 2)
            coarse_kde_signal_template = signal_template_original_hist.values / signal_template_original_hist.values.sum()
            reduced_kde_signal_template = signal_template_reduced_hist.values / signal_template_reduced_hist.values.sum()

            template_cache[template_cache_key] = (kde_bandwidth, mva_binning, fine_mva_binning, reduced_mva_binning, coarse_kde_background_template, kde_background_template, reduced_kde_background_template, coarse_kde_signal_template, kde_signal_template, reduced_kde_signal_template)
        else:
            print("Taking templates from cache")
            kde_bandwidth, mva_binning, fine_mva_binning, reduced_mva_binning, coarse_kde_background_template, kde_background_template, reduced_kde_background_template, coarse_kde_signal_template, kde_signal_template, reduced_kde_signal_template = template_cache[template_cache_key]
        background_template = kde_background_template
        signal_template = kde_signal_template

        def _signal_pdf(points):
            indices = fine_mva_binning.get_indices(points)
            return kde_signal_template[indices] / fine_mva_binning.bin_widths[indices]
        def _binned_kde_background_pdf(points):
            indices = fine_mva_binning.get_indices(points)
            return kde_background_template[indices] / fine_mva_binning.bin_widths[indices]
        #signal_template = fine_signal_template
        plot_template_mva_binning = fine_mva_binning
        coarse_data_hist = data_hist
        data_hist = prediction_hist(negative_data, data_predictions, fine_mva_binning, f"BDT {title}")
        reduced_data_hist = prediction_hist(negative_data, data_predictions, reduced_mva_binning, f"BDT {title}")
        kde_background_template_hist = Histogram(fine_mva_binning, values=kde_background_template, labels=(f"$\\Lambda_{{CC}}$"))
        negative_events_in_template_in_signal_region = integrate_histogram_above_cut_value(kde_background_template_hist, cut_value_signal) * len(background_template_predictions) * mc_to_iss_factor
        print(f"Events above cut (in template): {negative_events_in_template_in_signal_region:.2f}")
        if mc_events_in_signal_region is not None:
            print(f"Template sensitivity reduction: {negative_events_in_template_in_signal_region:.2f} - {mc_events_in_signal_region*mc_to_iss_factor:.2f} = {negative_events_in_template_in_signal_region - mc_events_in_signal_region * mc_to_iss_factor:.2f}")

    if store_templates:
        result_data["templates"] = dict(mva=dict(edges=list(map(float, fine_mva_binning.edges)), signal_values=list(map(float, signal_template)), background_values=list(map(float, background_template)), data_values=list(map(float, data_predictions))))
        result_data["event_data"] = dict(mva=dict(edges=list(map(float, mva_binning.edges)), data_values=list(map(float, data_predictions)), data_weights=list(map(float, negative_data.events.TotalWeight)), mc_background_values=list(map(float, background_template_predictions)), mc_background_weights=list(map(float, background_template_data.events.TotalWeight))))

    if not reduce_number_of_plots:
        signal_template_figure = plt.figure(figsize=(12, 6.15))
        signal_template_plot = signal_template_figure.subplots(1, 1)
        signal_template_figure.suptitle(f"{title}")
        plot_histogram_1d(signal_template_plot, signal_template_original_hist, label=signal_template_label, style="iss", use_approximate_poisson_errors=True, zorder=3, color=DATA_COLOR)
        plot_steps(signal_template_plot, plot_template_mva_binning.edges, signal_template * signal_template_original_hist.values.sum() * fine_mva_binning_factor, label=f"Signal Template", color=SIGNAL_COLOR)
        #ymax = round_up(np.max(signal_template_original_hist.values))
        #ymax = 10**(round_up(np.log10(np.max(signal_template_original_hist.values) / signal_template_original_hist.values.sum() / fine_mva_binning_factor)))
        #ymin = 0 #ymax / 1e4
        #signal_template_plot.set_ylim(ymin, ymax)
        #signal_template_plot.set_ylabel("Events")
        signal_template_plot.set_ylim(bottom=0)
        signal_template_plot.set_xlabel("$\\Lambda_{CC}$")
        signal_template_plot.legend()
        save_figure(signal_template_figure, plotdir, f"{prefix}_signal_template")

        background_template_figure = plt.figure(figsize=(12, 6.15))
        background_template_plot = background_template_figure.subplots(1, 1)
        background_template_figure.suptitle(f"{title}")
        background_template_scale = 1 / background_template_original_hist.values.sum()
        background_template_chisq, background_template_dof, background_template_rchisq = calculate_chisq(data=background_template_original_hist.values * background_template_scale, model=coarse_kde_background_template, errors=background_template_original_hist.get_errors() * background_template_scale, n_parameters=1)

        plot_histogram_1d(background_template_plot, background_template_original_hist, label=background_template_label, style="iss", zorder=3, color=DATA_COLOR)#, use_approximate_poisson_errors=True)
        plot_steps(background_template_plot, plot_template_mva_binning.edges, kde_background_template * background_template_original_hist.values.sum() * fine_mva_binning_factor, label=f"Background Template\n$\\chi^2/\\mathrm{{dof}}={background_template_chisq:.1f}/{background_template_dof:.0f}={background_template_rchisq:.2f}$", color=BACKGROUND_COLOR)
        #background_template_plot.set_ylim(10**(round_down(np.log10(np.min(background_template_original_hist.values[background_template_original_hist.values > 0]) / background_template_original_hist.values.sum() / fine_mva_binning_factor))), 10**(round_up(np.log10(np.max(background_template_original_hist.values) / background_template_original_hist.values.sum() / fine_mva_binning_factor))))
        background_template_plot.set_xlabel("$\\Lambda_{CC}$")
        background_template_plot.set_ylim(bottom=0)
        background_template_plot.legend()
        save_figure(background_template_figure, plotdir, f"{prefix}_background_template")

        coarse_background_template_figure = plt.figure(figsize=(12, 6.15))
        coarse_background_template_plot = coarse_background_template_figure.subplots(1, 1)
        coarse_background_template_figure.suptitle(f"{title}")
        background_template_scale = 1 / background_template_original_hist.values.sum()
        background_template_chisq, background_template_dof, background_template_rchisq = calculate_chisq(data=background_template_original_hist.values * background_template_scale, model=coarse_kde_background_template, errors=background_template_original_hist.get_errors() * background_template_scale, n_parameters=1)

        plot_histogram_1d(coarse_background_template_plot, background_template_original_hist, label=background_template_label, style="iss", zorder=3, color=DATA_COLOR)#, use_approximate_poisson_errors=True)
        plot_steps(coarse_background_template_plot, mva_binning.edges, coarse_kde_background_template * background_template_original_hist.values.sum(), label=f"Background Template\n$\\chi^2/\\mathrm{{dof}}={background_template_chisq:.1f}/{background_template_dof:.0f}={background_template_rchisq:.2f}$", color=BACKGROUND_COLOR)
        #coarse_background_template_plot.set_ylim(10**(round_down(np.log10(np.min(background_template_original_hist.values[background_template_original_hist.values > 0]) / background_template_original_hist.values.sum() / fine_mva_binning_factor))), 10**(round_up(np.log10(np.max(background_template_original_hist.values) / background_template_original_hist.values.sum() / fine_mva_binning_factor))))
        coarse_background_template_plot.set_xlabel("$\\Lambda_{CC}$")
        coarse_background_template_plot.set_ylim(bottom=0)
        coarse_background_template_plot.legend()
        save_figure(coarse_background_template_figure, plotdir, f"{prefix}_background_template_coarse")

        result_data["background_template_data"] = dict(chisq=background_template_chisq, dof=int(background_template_dof), rchisq=background_template_rchisq)

    print("Template Normalization", signal_template.sum(), background_template.sum())
    def _fit_function(signal_counts, background_counts):
        return signal_counts * signal_template + background_counts * background_template
    def _coarse_fit_function(signal_counts, background_counts):
        return signal_counts * coarse_kde_signal_template + background_counts * coarse_kde_background_template
    def _reduced_fit_function(signal_counts, background_counts):
        return signal_counts * reduced_kde_signal_template + background_counts * reduced_kde_background_template
    def _lsq_fit_function(_, signal_counts, background_counts):
        return (signal_counts * signal_template + background_counts * background_template)[lsq_mask]
    def _cdf(arg_edges, signal_counts, background_counts):
        pdf = _fit_function(signal_counts, background_counts)
        cdf = np.concatenate(([0], np.cumsum(pdf)))
        return cdf
    def _binned_pdf(arg_points, signal_counts, background_counts):
        pdf = _fit_function(signal_counts, background_counts) / (bin_edges[1:] - bin_edges[:-1])
        arg_bins = data_hist.binnings[0].get_indices(arg_points)
        return signal_counts + background_counts, pdf[arg_bins]

    if not allow_deficit:
        _kde_pdf = make_kde_pdf(_signal_pdf, _binned_kde_background_pdf)
    else:
        _kde_pdf = make_kde_pdf_with_limited_deficit(_signal_pdf, _binned_kde_background_pdf, data_hist.binnings[0].bin_centers)

    sample_size = len(data_predictions)
    print("size", sample_size, len(signal_template_predictions), len(background_template_predictions))

    guess = dict(signal_counts=1, background_counts=data_hist.values.sum() - 1)
    #errordef = Minuit.LIKELIHOOD
    if method == "llh":
        loss = ExtendedBinnedNLL(data_counts, bin_edges, _cdf)
    elif method == "lsq":
        loss = LeastSquares(data_hist.binnings[0].bin_centers, data_hist.values, data_hist.get_errors(), _lsq_fit_function)
        loss.mask = lsq_mask
        #errordef = Minuit.LEAST_SQUARES
    elif method == "ubllh":
        if use_kde:
            loss = ExtendedUnbinnedNLL(data_predictions, _kde_pdf)
        else:
            loss = ExtendedUnbinnedNLL(data_predictions, _binned_pdf)
    else:
        raise ValueError(f"Unknown fit method {method!r}")
    m = Minuit(loss, **guess)
    #m.errordef = errordef
    #if not do_contour:
    m.limits["signal_counts"] = (0, None)
    m.limits["background_counts"] = (0, None)
    m.migrad()
    if m.valid:
        m.minos()
    print(m)
    print("Minimum likelihood:", m.fval)
    result_data["minimum_likelihood"] = m.fval
    result_data["fit_valid"] = m.valid
    #print(m.merrors)
    print(m.parameters, m.values, m.covariance)
    fit_parameters = dict(zip(m.parameters, uncertainties.correlated_values(m.values, np.array(m.covariance))))
    fit_param_names = m.parameters
    fit_param_values = m.values
    fit_param_errors = m.errors
    fit_param_dict = dict(zip(fit_param_names, fit_param_values))
    fit_param_error_dict = dict(zip(fit_param_names, fit_param_errors))
    fit_param_errors_asymm = {key: (error.lower, error.upper) for key, error in m.merrors.items()} if m.valid else None
    print("fpea", fit_param_errors_asymm)
    fit_values = _fit_function(**fit_param_dict)
    coarse_fit_values = _coarse_fit_function(**fit_param_dict)
    chisq, dof, rchisq = calculate_chisq(data=coarse_data_hist.values, model=coarse_fit_values, errors=np.sqrt(coarse_fit_values), n_parameters=len(guess))
    print(f"χ²/dof = {chisq:.1f} / {dof:.0f} = {rchisq:.2f}")
    reduced_fit_values = _reduced_fit_function(**fit_param_dict)
    reduced_chisq, reduced_dof, reduced_rchisq = calculate_chisq(data=reduced_data_hist.values, model=reduced_fit_values, errors=np.sqrt(reduced_fit_values), n_parameters=len(guess))
    print(f"χ²/dof = {reduced_chisq:.1f} / {reduced_dof:.0f} = {reduced_rchisq:.2f} (reduced binning)")
    #nonzero = coarse_data_hist.values > 0
    #residuals = ((coarse_data_hist.values[nonzero] - coarse_fit_values[nonzero]) / coarse_data_hist.get_errors()[nonzero])**2
    #chisq = residuals.sum()
    #dof = nonzero.sum() - len(guess)
    #rchisq = chisq / dof

    result_data["template_fit"] = dict(fit_param_values=fit_param_dict, fit_param_errors=fit_param_error_dict, fit_param_errors_asymm=fit_param_errors_asymm, chisq=dict(chisq=chisq, dof=int(dof), rchisq=rchisq))

    confidence_intervals_constant = {}
    confidence_intervals_sample_size = {}
    for sigmas in range(1, 4):
        confidence_intervals_constant[sigmas] = calculate_confidence_interval(sigmas, loss, fit_param_values, adjust_lower=False, compensation=0)
        confidence_intervals_sample_size[sigmas] = calculate_confidence_interval(sigmas, loss, fit_param_values, adjust_lower=False, compensation=1)
    confidence_intervals_count = dict(constant=confidence_intervals_constant, sample_size=confidence_intervals_sample_size)
    confidence_intervals_ratio = {key: {sigma: (lower_count / positive_data_event_count, upper_count / positive_data_event_count) for sigma, (lower_count, upper_count) in d.items()} for key, d in confidence_intervals_count.items()}
    count_limit = confidence_intervals_sample_size[2][1]
    count_limit_3_sigma = confidence_intervals_sample_size[3][1]
    ratio_limit = confidence_intervals_ratio["sample_size"][2][1]

    result_data["confidence_intervals"] = dict(event_count=confidence_intervals_count, ratio=confidence_intervals_ratio)
    print(result_data["confidence_intervals"])

    
    cl_labels = [f"95% Confidence Limit:", f"Fit: {count_limit:.2f} / {format_order_of_magnitude(positive_data_event_count)} = {format_order_of_magnitude(ratio_limit)}"]
    cv_labels = []
    if efficiency_data is not None:
        cl_labels.append(f"Cut: {count_limit_cut:.2f} / {format_order_of_magnitude(positive_events_in_signal_region)} = {format_order_of_magnitude(ratio_limit_cut)}")
        cv_labels.append(f"Positive Events: {format_order_of_magnitude(positive_events_in_signal_region)}")
        cv_labels.append(f"Negative Events: {format_order_of_magnitude(negative_events_in_signal_region)}")

    n_hypothetical = 10
    if coarse_data_hist.values.sum() > 200:
        n_hypothetical = 20
    if coarse_data_hist.values.sum() > 1000:
        n_hypothetical = 100

    title_no_ccmva = title.replace("CCMVA ", "")

    draw_template_fit(data_hist=coarse_data_hist, signal_template=coarse_kde_signal_template, background_template=coarse_kde_background_template, template_binning=mva_binning, fit_parameters=fit_parameters, fit_param_dict=fit_param_dict, fit_valid=m.valid, chisq=chisq, dof=dof, rchisq=rchisq, title=f"Result from Unbinned Maximum Likelihood Template Fit", subtitle=f"{title}", background_template_label=f"Background Template", plotdir=plotdir, prefix=prefix, log=False, postfix="presentation_coarse_lin", n_hypothetical=n_hypothetical, additional_labels=cv_labels + cl_labels, show_empty_components=False, label_x="$\\Lambda_{CC}$", target_efficiency=target_efficiency, target_cut_value=cut_value_signal, data_label=data_label)
    if not reduce_number_of_plots:
        draw_template_fit(data_hist=coarse_data_hist, signal_template=signal_template, background_template=background_template, template_binning=plot_template_mva_binning, fit_parameters=fit_parameters, fit_param_dict=fit_param_dict, fit_valid=m.valid, chisq=chisq, dof=dof, rchisq=rchisq, title=f"{title} Template Fit", background_template_label=f"Background Template\nBandwidth={kde_bandwidth:.2f}", plotdir=plotdir, prefix=prefix, data_label=data_label)
        draw_template_fit(data_hist=coarse_data_hist, signal_template=signal_template, background_template=background_template, template_binning=plot_template_mva_binning, fit_parameters=fit_parameters, fit_param_dict=fit_param_dict, fit_valid=m.valid, chisq=chisq, dof=dof, rchisq=rchisq, title=f"Result from Unbinned Maximum Likelihood Template Fit", subtitle=f"{title}", background_template_label=f"Background Template", plotdir=plotdir, prefix=prefix, log=False, postfix="presentation_lin", n_hypothetical=n_hypothetical, additional_labels=cl_labels, show_empty_components=False, label_x="$\\Lambda_{CC}$", target_efficiency=target_efficiency, target_cut_value=cut_value_signal, data_label=data_label)
        draw_template_fit(data_hist=coarse_data_hist, signal_template=signal_template, background_template=background_template, template_binning=plot_template_mva_binning, fit_parameters=fit_parameters, fit_param_dict=fit_param_dict, fit_valid=m.valid, chisq=chisq, dof=dof, rchisq=rchisq, title=f"Result from Unbinned Maximum Likelihood Template Fit", subtitle=f"{title}", background_template_label=f"Background Template", plotdir=plotdir, prefix=prefix, log=True, postfix="presentation_log", n_hypothetical=n_hypothetical, additional_labels=cl_labels, show_empty_components=False, label_x="$\\Lambda_{CC}$", target_efficiency=target_efficiency, target_cut_value=cut_value_signal, data_label=data_label)
        draw_template_fit(data_hist=coarse_data_hist, signal_template=coarse_kde_signal_template, background_template=coarse_kde_background_template, template_binning=mva_binning, fit_parameters=fit_parameters, fit_param_dict=fit_param_dict, fit_valid=m.valid, chisq=chisq, dof=dof, rchisq=rchisq, title=f"Result from Unbinned Maximum Likelihood Template Fit", subtitle=f"{title}", background_template_label=f"Background Template", plotdir=plotdir, prefix=prefix, log=True, postfix="presentation_coarse_log", n_hypothetical=n_hypothetical, additional_labels=cv_labels + cl_labels, show_empty_components=False, label_x="$\\Lambda_{CC}$", target_efficiency=target_efficiency, target_cut_value=cut_value_signal, data_label=data_label)
        draw_template_fit(data_hist=coarse_data_hist, signal_template=coarse_kde_signal_template, background_template=coarse_kde_background_template, template_binning=mva_binning, fit_parameters=fit_parameters, fit_param_dict=fit_param_dict, fit_valid=m.valid, chisq=chisq, dof=dof, rchisq=rchisq, title=f"Result from Unbinned Maximum Likelihood Template Fit", subtitle=f"{title_no_ccmva}", background_template_label=f"Background Template", plotdir=plotdir, prefix=prefix, log=False, postfix="presentation_coarse_nolimit_lin", n_hypothetical=n_hypothetical, additional_labels=cv_labels, show_empty_components=False, label_x="$\\Lambda_{CC}$", target_efficiency=target_efficiency, target_cut_value=cut_value_signal, data_label=data_label)
        draw_template_fit(data_hist=coarse_data_hist, signal_template=coarse_kde_signal_template, background_template=coarse_kde_background_template, template_binning=mva_binning, fit_parameters=fit_parameters, fit_param_dict=fit_param_dict, fit_valid=m.valid, chisq=chisq, dof=dof, rchisq=rchisq, title=f"Result from Unbinned Maximum Likelihood Template Fit", subtitle=f"{title_no_ccmva}", background_template_label=f"Background Template", plotdir=plotdir, prefix=prefix, log=True, postfix="presentation_coarse_nolimit_log", n_hypothetical=n_hypothetical, additional_labels=cv_labels, show_empty_components=False, label_x="$\\Lambda_{CC}$", target_efficiency=target_efficiency, target_cut_value=cut_value_signal, data_label=data_label)
        draw_template_fit(data_hist=coarse_data_hist, signal_template=signal_template, background_template=background_template, template_binning=plot_template_mva_binning, fit_parameters=fit_parameters, fit_param_dict=fit_param_dict, fit_valid=m.valid, chisq=chisq, dof=dof, rchisq=rchisq, title=f"Result from Unbinned Maximum Likelihood Template Fit", subtitle=f"{title}", background_template_label=f"Background Template", plotdir=plotdir, prefix=prefix, log=False, postfix="presentation_modelerrors", n_hypothetical=n_hypothetical, additional_labels=cl_labels, show_empty_components=False, show_errors_on_model=True, label_x="$\\Lambda_{CC}$", target_efficiency=target_efficiency, target_cut_value=cut_value_signal, data_label=data_label)
        draw_template_fit(data_hist=reduced_data_hist, signal_template=reduced_kde_signal_template, background_template=reduced_kde_background_template, template_binning=reduced_mva_binning, fit_parameters=fit_parameters, fit_param_dict=fit_param_dict, fit_valid=m.valid, chisq=reduced_chisq, dof=reduced_dof, rchisq=reduced_rchisq, title=f"Result from Unbinned Maximum Likelihood Template Fit", subtitle=f"{title}", background_template_label=f"Background Template", plotdir=plotdir, prefix=prefix, log=False, postfix="presentation_reduced", n_hypothetical=n_hypothetical, additional_labels=cl_labels, show_empty_components=False, label_x="$\\Lambda_{CC}$", target_efficiency=target_efficiency, target_cut_value=cut_value_signal, data_label=data_label)

    if do_contour:
        for contour_dof in (1, 2):
            draw_contour(m, loss, fit_param_dict, plotdir=plotdir, prefix=f"{prefix}_dof{contour_dof}", title=title, dof=contour_dof, sample_size=sample_size)

    if not reduce_number_of_plots:
        linearity_figure = plt.figure(figsize=(12, 6.15))
        linearity_plot = linearity_figure.subplots(1, 1)
        linearity_figure.suptitle(f"{title} Signal region Cumulative events")
        linearity_plot.set_xlabel("Fit result")
        linearity_plot.set_ylabel("Data")
        max_point = 3 * max(fit_param_dict["signal_counts"], 1)
        linearity_plot.set_xlim(0, max_point)
        linearity_plot.set_ylim(0, max_point)
        linear_values = np.linspace(0, max_point, 10)

        cumulative_fit_values = np.cumsum(fit_values[::-1])
        cumulative_data_values = np.cumsum(data_hist.values[::-1])
        cumulative_data_errors = np.sqrt(np.cumsum(data_hist.get_errors()[::-1]**2))
        linearity_plot.errorbar(cumulative_fit_values, cumulative_data_values, cumulative_data_errors, fmt=".")
        linearity_plot.plot(linear_values, linear_values, "-", color="gray", alpha=0.5)

        save_figure(linearity_figure, plotdir, f"{prefix}_linearity")

    if attempt_2d_fit is not None:
        print(f"Attemping 2D template fit with {attempt_2d_fit}")
        second_variable = attempt_2d_fit
        background_like_negative_events, background_like_negative_predictions = negative_data.get_most_background_like(data_predictions, len(negative_data.events) // 2)
        fine_binning_factor = 10
        second_binning, fine_second_binning, second_signal_template, second_signal_template_hist, second_background_template, second_background_template_hist, _signal_pdf_second, _background_pdf_second, _fit_function_second, _coarse_fit_function_second = make_templates_for_variable(positive_data.events, background_like_negative_events, second_variable, fine_binning_factor=fine_binning_factor)

        _second_pdf = make_kde_pdf(_signal_pdf_second, _background_pdf_second)
        second_data_hist = WeightedHistogram.fill_direct((second_binning,), negative_data.events[second_variable], weights=negative_data.events.TotalWeight, labels=(second_variable,))

        guess = dict(signal_counts=1, background_counts=len(negative_data.events) - 1)
        second_loss = ExtendedUnbinnedNLL(negative_data.events[second_variable], _second_pdf)
        loss2d = loss + second_loss
        m2d = Minuit(loss2d, **guess)
        m2d.migrad()
        if m2d.valid:
            m2d.minos()
        print(m2d)
        fit_parameters_2d = dict(zip(m2d.parameters, uncertainties.correlated_values(m2d.values, np.array(m2d.covariance))))
        fit_param_names_2d = m2d.parameters
        fit_param_values_2d = m2d.values
        fit_param_errors_2d = m2d.errors
        fit_param_dict_2d = dict(zip(fit_param_names, fit_param_values))
        fit_param_error_dict_2d = dict(zip(fit_param_names, fit_param_errors))

        fit_values_2d = _fit_function(**fit_param_dict_2d)
        coarse_fit_values_2d = _coarse_fit_function(**fit_param_dict_2d)
        second_fit_values_2d = _fit_function_second(**fit_param_dict_2d)
        coarse_second_fit_values_2d = _coarse_fit_function_second(**fit_param_dict_2d)

        chisq_2d_mva, dof_2d_mva, rchisq_2d_mva = calculate_chisq(data=coarse_data_hist.values, model=coarse_fit_values_2d, errors=np.sqrt(coarse_fit_values_2d), n_parameters=len(guess))
        #nonzero_2d_mva = coarse_data_hist.values > 0
        #residuals_2d_mva = ((coarse_data_hist.values[nonzero_2d_mva] - coarse_fit_values_2d[nonzero_2d_mva]) / coarse_data_hist.get_errors()[nonzero_2d_mva])**2
        #chisq_2d_mva = residuals_2d_mva.sum()
        #dof_2d_mva = nonzero_2d_mva.sum() - len(guess)
        #rchisq_2d_mva = chisq_2d_mva / dof_2d_mva

        chisq_2d_second, dof_2d_second, rchisq_2d_second = calculate_chisq(data=second_data_hist.values, model=coarse_second_fit_values_2d, errors=np.sqrt(coarse_second_fit_values_2d), n_parameters=len(guess))
        #nonzero_2d_second = second_data_hist.values > 0
        #residuals_2d_second = ((second_data_hist.values[nonzero_2d_second] - coarse_second_fit_values_2d[nonzero_2d_second]) / second_data_hist.get_errors()[nonzero_2d_second])**2
        #chisq_2d_second = residuals_2d_second.sum()
        #dof_2d_second = nonzero_2d_second.sum() - len(guess)
        #rchisq_2d_second = chisq_2d_second / dof_2d_second

        confidence_intervals_2d = {sigmas: calculate_confidence_interval(sigmas, loss2d, fit_param_values_2d, adjust_lower=False, compensation=1) for sigmas in range(1, 4)}
        confidence_intervals_ratio_2d = {sigmas: (lower / positive_data_event_count, upper / positive_data_event_count) for sigmas, (lower, upper) in confidence_intervals_2d.items()}
        print("CI2D", confidence_intervals_2d, confidence_intervals_ratio_2d)

        result_data["fit_2d"] = {
            "template_fit": dict(fit_param_values=fit_param_dict_2d, fit_param_errors=fit_param_error_dict_2d, chisq_mva=dict(chisq=chisq_2d_mva, dof=int(dof_2d_mva), rchisq=rchisq_2d_mva), chisq_second=dict(chisq=chisq_2d_second, dof=int(dof_2d_second), rchisq=rchisq_2d_second), variable=second_variable),
            "confidence_intervals": dict(event_count=confidence_intervals_2d, ratio=confidence_intervals_ratio_2d),
        }

        draw_template_fit(data_hist=coarse_data_hist, signal_template=signal_template, background_template=background_template, template_binning=plot_template_mva_binning, fit_parameters=fit_parameters_2d, fit_param_dict=fit_param_dict_2d, fit_valid=m2d.valid, chisq=chisq_2d_mva, dof=dof_2d_mva, rchisq=rchisq_2d_mva, title=f"{title} 2D Template Fit", plotdir=plotdir, prefix=f"{prefix}_2d_mva")
        draw_template_fit(data_hist=second_data_hist, signal_template=second_signal_template, background_template=second_background_template, template_binning=fine_second_binning, fit_parameters=fit_parameters_2d, fit_param_dict=fit_param_dict_2d, fit_valid=m2d.valid, chisq=chisq_2d_second, dof=dof_2d_second, rchisq=rchisq_2d_second, title=f"{title} 2D Template Fit", plotdir=plotdir, prefix=f"{prefix}_2d_{second_variable}")
    
        draw_contour(m2d, loss2d, fit_param_dict_2d, plotdir=plotdir, prefix=f"{prefix}_2d", title=title, dof=2, sample_size=sample_size)

        combined_pdf = combine_pdfs(_kde_pdf, _second_pdf)
        combined_data = np.stack((data_predictions, negative_data.events[second_variable]), axis=1)
        combined_loss = ExtendedUnbinnedNLL(combined_data, combined_pdf)
        mcombined = Minuit(combined_loss, **guess)
        mcombined.migrad()
        if mcombined.valid:
            mcombined.minos()
        print(mcombined)
        fit_parameters_combined = dict(zip(mcombined.parameters, uncertainties.correlated_values(mcombined.values, np.array(mcombined.covariance))))
        fit_param_names_combined = mcombined.parameters
        fit_param_values_combined = mcombined.values
        fit_param_errors_combined = mcombined.errors
        fit_param_dict_combined = dict(zip(fit_param_names, fit_param_values))
        fit_param_error_dict_combined = dict(zip(fit_param_names, fit_param_errors))

        confidence_intervals_combined = {sigmas: calculate_confidence_interval(sigmas, combined_loss, fit_param_values_combined, adjust_lower=False, compensation=1) for sigmas in range(1, 4)}
        confidence_intervals_ratio_combined = {sigmas: (lower / positive_data_event_count, upper / positive_data_event_count) for sigmas, (lower, upper) in confidence_intervals_combined.items()}
        print("CI2D_combined", confidence_intervals_combined, confidence_intervals_ratio_combined)


    return result_data, template_cache, prediction_cache


def define_signal_region(hist, min_fraction=0.99):
    assert hist.dimensions == 1
    indices = np.arange(len(hist.binnings[0].edges) - 1)
    fractions = hist.values / hist.values.sum()
    pairs = sorted(zip(indices, fractions), key=lambda t: t[1], reverse=True)
    cum_fraction = 0
    signal_bins = []
    for index, fraction in pairs:
        cum_fraction += fraction
        signal_bins.append(index)
        if cum_fraction >= min_fraction:
            break
    signal_bins = sorted(signal_bins)


def calculate_rejection(signal_histogram, background_histogram, plotdir, prefix, title, label, rig_title="", target_efficiency=0.9, comparison=None):
    efficiency, efficiency_error, rejection, rejection_error, cut_values = calculate_efficiency_and_rejection(signal_histogram, background_histogram)
    highest_rejection_over = np.max(rejection[efficiency > 0.5])
    efficiency_rejection_figure = plt.figure(figsize=(12, 6.15))
    efficiency_rejection_figure.suptitle(f"{title} efficiency{rig_title}")
    efficiency_rejection_plot = efficiency_rejection_figure.subplots(1, 1)
    efficiency_rejection_plot.grid()
    efficiency_rejection_plot.errorbar(efficiency * 100, rejection, rejection_error, efficiency_error * 100, fmt=".", label=label)

    try:
        if comparison is not None:
            comparison_label, comparison_data = comparison
            efficiency_rejection_plot.errorbar(np.array(comparison_data["efficiency"]) * 100, np.array(comparison_data["rejection"]), np.array(comparison_data["rejection_error"]), np.array(comparison_data["efficiency_error"]) * 100, fmt=".", label=comparison_label, alpha=0.5)
    except ValueError as error:
        print(error)

    efficiency_rejection_plot.set_xlabel("Signal efficiency / %")
    efficiency_rejection_plot.set_ylabel("Background rejection")
    efficiency_rejection_plot.set_yscale("log")
    efficiency_rejection_plot.set_xlim(50, 101)
    efficiency_rejection_plot.set_ylim(1, 1.1 * highest_rejection_over)
    efficiency_rejection_plot.legend()
    save_figure(efficiency_rejection_figure, plotdir, f"{prefix}_efficiency_rejection")

    rejection_at = np.interp(target_efficiency, efficiency, rejection)
    return dict(rejection_at=rejection_at, efficiency=list(efficiency), rejection=list(rejection), efficiency_error=list(efficiency_error), rejection_error=list(rejection_error), cut_values=list(cut_values), label=label)


def calculate_efficiency(signal_histogram, background_histogram, plotdir, prefix, title, label, rig_title="", target_efficiencies=(0.9,), comparison=None, log=True):
    signal_efficiency, signal_efficiency_error, background_efficiency, background_efficiency_error, cut_values = calculate_signal_and_background_efficiency(signal_histogram, background_histogram)
    efficiency_figure = plt.figure(figsize=(12, 6.15))
    efficiency_figure.suptitle(f"{title} efficiency{rig_title}")
    efficiency_plot = efficiency_figure.subplots(1, 1)
    efficiency_plot.grid()
    efficiency_plot.errorbar(signal_efficiency * 100, background_efficiency * 100, background_efficiency_error * 100, signal_efficiency_error * 100, fmt=".", label=label)

    if comparison is not None:
        comparison_label, comparison_data = comparison
        efficiency_plot.errorbar(np.array(comparison_data["signal_efficiency"]) * 100, np.array(comparison_data["background_efficiency"]) * 100, np.array(comparison_data["background_efficiency_error"]) * 100, np.array(comparison_data["signal_efficiency_error"]) * 100, fmt=".", label=comparison_label, alpha=0.5)

    efficiency_plot.set_xlabel("Signal efficiency / %")
    efficiency_plot.set_ylabel("Background efficiency / %")
    efficiency_plot.set_xlim(50, 101)
    #efficiency_plot.set_ylim(1e-5, 10)
    if log:
        efficiency_plot.set_yscale("log")
        efficiency_plot.set_ylim(1e-1, 1e2)
    else:
        efficiency_plot.set_ylim(0, 101)
    efficiency_plot.legend()
    save_figure(efficiency_figure, plotdir, f"{prefix}_efficiency")

    target_efficiency_data = {}
    for target_efficiency in target_efficiencies:
        background_efficiency_at = interp1d(signal_efficiency, background_efficiency, fill_value="extrapolate")(target_efficiency)
        cut_value_at = interp1d(signal_efficiency, cut_values, fill_value="extrapolate")(target_efficiency)
        target_efficiency_data[target_efficiency] = dict(background_efficiency=float(background_efficiency_at), cut_value=float(cut_value_at))

    return dict(target_efficiency_data=target_efficiency_data, signal_efficiency=list(signal_efficiency), background_efficiency=list(background_efficiency), signal_efficiency_error=list(signal_efficiency_error), background_efficiency_error=list(background_efficiency_error), cut_values=list(cut_values), label=label)


def compare_selection_efficiency(datasets, title, rig_title, plotdir, prefix, rig_min, rig_max):
    result_data = {}
    selection_names = [name for name in datasets[0].selections if all(name in dataset.selections for dataset in datasets)]
    for selection_name in selection_names:
        print(selection_name)
        selection_efficiency_figure = plt.figure(figsize=(12, 6.15))
        selection_efficiency_plot = selection_efficiency_figure.subplots(1, 1)
        selection_efficiency_plot.set_title(selection_name)
        selection_efficiency_plot.set_xlabel("Rigidity / GV")
        selection_efficiency_plot.set_ylabel("Efficiency")
        selection_efficiency_plot.set_ylim(0, 1)
        selection_efficiency_plot.set_xscale("log")
        selection_efficiency_plot.set_xlim(rig_min, rig_max)
        for dataset in datasets:
            selection = dataset.selections[selection_name]
            passed = selection.total_passed_histogram
            failed = selection.total_failed_histogram
            rigidities = passed.binnings[0].bin_centers
            bin_min, bin_max = passed.binnings[0].get_indices(np.array([rig_min, rig_max]))
            passed_in_range = passed.values[bin_min:bin_max+1].sum()
            passed_squared_in_range = passed.squared_values[bin_min:bin_max+1].sum()
            failed_in_range = failed.values[bin_min:bin_max+1].sum()
            failed_squared_in_range = failed.squared_values[bin_min:bin_max+1].sum()
            efficiency, efficiency_error = calculate_efficiency_and_error_weighted(passed.values, failed.values, passed.squared_values, failed.squared_values)
            overall_efficiency, overall_efficiency_error = calculate_efficiency_and_error_weighted(passed_in_range, failed_in_range, passed_squared_in_range, failed_squared_in_range)
            print(f"Overall efficiency {selection_name} ({dataset.name}, {rig_min:.1f}<R<{rig_max:.1f}): {uncertainties.ufloat(overall_efficiency, overall_efficiency_error)*100:P} %")
            selection_efficiency_plot.errorbar(rigidities, efficiency, efficiency_error, label=dataset.label, fmt=".")
        set_energy_ticks(selection_efficiency_plot)
        selection_efficiency_plot.legend()
        save_figure(selection_efficiency_figure, plotdir, f"{prefix}_selection_{selection_name}")
        for cut_variable, ref_cut in datasets[0].selections[selection_name].cuts.items():
            cut_efficiency_figure = plt.figure(figsize=(12, 6.15))
            cut_efficiency_plot = cut_efficiency_figure.subplots(1, 1)
            cut_efficiency_plot.set_title(ref_cut.label)
            cut_efficiency_plot.set_xlabel("Rigidity / GV")
            cut_efficiency_plot.set_ylabel("Efficiency")
            cut_efficiency_plot.set_ylim(0, 1)
            cut_efficiency_plot.set_xscale("log")
            cut_efficiency_plot.set_xlim(rig_min, rig_max)
            for dataset in datasets:
                cut = dataset.selections[selection_name].cuts[cut_variable]
                passed = cut.passed_histogram_per_rig.project(0, len(cut.passed_histogram_per_rig.binnings[1]), axis=1)
                failed = cut.failed_histogram_per_rig.project(0, len(cut.failed_histogram_per_rig.binnings[1]), axis=1)
                rigidities = passed.binnings[0].bin_centers
                efficiency, efficiency_error = calculate_efficiency_and_error_weighted(passed.values, failed.values, passed.squared_values, failed.squared_values)
                bin_min, bin_max = passed.binnings[0].get_indices(np.array([rig_min, rig_max]))
                overall_efficiency, overall_efficiency_error = calculate_efficiency_and_error_weighted(passed.values[bin_min:bin_max+1].sum(), failed.values[bin_min:bin_max+1].sum(), passed.squared_values[bin_min:bin_max+1].sum(), failed.squared_values[bin_min:bin_max+1].sum())
                print(f"Overall efficiency {selection_name} {cut_variable} ({dataset.name}, {rig_min:.1f}<R<{rig_max:.1f}): {uncertainties.ufloat(overall_efficiency, overall_efficiency_error)*100:P} %")
                cut_efficiency_plot.errorbar(rigidities, efficiency, efficiency_error, label=dataset.label, fmt=".")
            set_energy_ticks(cut_efficiency_plot)
            cut_efficiency_plot.legend()
            save_figure(cut_efficiency_figure, plotdir, f"{prefix}_cut_{selection_name}_{cut_variable}")
            
    acceptance_figure = plt.figure(figsize=(12, 6.15))
    acceptance_plot = acceptance_figure.subplots(1, 1)
    acceptance_plot.set_title(f"{title}{rig_title}")
    acceptance_plot.set_xlabel(f"Rigidity / GV")
    acceptance_plot.set_ylabel(f"Effective Acceptance / $cm^{2}sr$")
    acceptance_plot.set_xscale("log")
    acceptance_data = {}
    for dataset in datasets:
        mc_rig_hist = dataset.hists_all["McAbsRigidity"]
        mc_trigger_hist = dataset.triggers
        assert mc_rig_hist.binnings[0] == mc_trigger_hist.binnings[0]
        effective_acceptance = mc_rig_hist.values / mc_trigger_hist.values * 3.9**2 * np.pi / dataset.fraction
        plot_steps(acceptance_plot, mc_rig_hist.binnings[0].edges, effective_acceptance * 1e4, label=dataset.label)
        bin_min, bin_max = mc_trigger_hist.binnings[0].get_indices(np.array([rig_min, rig_max]))
        events_in_range = mc_rig_hist.values[bin_min:bin_max + 1].sum()
        triggers_in_range = mc_trigger_hist.values[bin_min:bin_max + 1].sum()
        overall_acceptance = events_in_range / triggers_in_range * 3.9**2 * np.pi / dataset.fraction
        acceptance_data[dataset.name] = dict(events=events_in_range, triggers=triggers_in_range, acceptance=overall_acceptance, rig_binning=list(mc_rig_hist.binnings[0].edges), mc_event_values=list(mc_rig_hist.values), mc_event_squared_values=list(mc_rig_hist.squared_values), mc_trigger_values=list(mc_trigger_hist.values), rig_min=rig_min, rig_max=rig_max)
    set_energy_ticks(acceptance_plot)
    acceptance_plot.legend()
    save_figure(acceptance_figure, plotdir, f"{prefix}_acceptance")
    result_data["acceptance"] = acceptance_data
    return result_data


def show_spectrum_2d(positive_data, negative_data, positive_predictions, negative_predictions, positive_mc, negative_mc, positive_mc_predictions, negative_mc_predictions, positive_monitors, variable, binnings, labelling, mva_binning, mva_label, title, rig_title, plotdir, prefix, cut_value_mva=None, n_signallike=10, target_efficiency=None, signal_masses=None, mass_target_efficiency=None):
    var_binning = binnings.variable_binnings[variable]
    is_log = var_binning.log
    edges = var_binning.edges[1:-1]
    if np.any(edges < 0):
        edges = edges[edges >= 0]
        if edges[0] > 0:
            edges = np.concatenate(([0], edges))
    raw_edges = edges
    transform = lambda e: e + 1e-7
    inverse = lambda e: e - 1e-7
    var_label = f"{variable}"
    if is_log:
        transform = lambda e: np.log(e + 1e-7)
        inverse = lambda e: np.exp(e) - 1e-7
        edges = transform(edges)
        var_label = f"log({var_label})"
    offset = -edges[0] + 1e-7
    edges = edges + offset
    all_edges = np.array(sorted(set(edges) | set(-edges)))
    combined_binning = Binning(all_edges)

    combined_hist = WeightedHistogram(combined_binning, mva_binning, labels=(f"sign(R)*{labelling.get_label(variable)}", f"MVA {mva_label}"))
    if target_efficiency is not None:
        positive_hist = WeightedHistogram(combined_binning, mva_binning, labels=(f"sign(R)*{labelling.get_label(variable)}", f"MVA {mva_label}"))
        negative_hist = WeightedHistogram(combined_binning, mva_binning, labels=(f"sign(R)*{labelling.get_label(variable)}", f"MVA {mva_label}"))
    positive_values = transform(np.abs(positive_data.events[variable])) + offset
    if np.any(positive_values < 0):
        print(f"Warning: {var_label}: clamping {np.sum(positive_values < 0)} positive values to zero")
        print(positive_values[positive_values < 0])
        positive_values[positive_values < 0] = 0
    negative_values = -(transform(np.abs(negative_data.events[variable])) + offset)
    if np.any(negative_values > 0):
        print(f"Warning: {var_label}: clamping {np.sum(negative_values > 0)} negative values to zero")
        print(negative_values[negative_values > 0])
        negative_values[negative_values > 0] = 0
    combined_hist.fill(positive_values, positive_predictions, weights=positive_data.events.TotalWeight)
    combined_hist.fill(negative_values, negative_predictions, weights=negative_data.events.TotalWeight)
    if target_efficiency is not None:
        positive_hist.fill(positive_values, positive_predictions, weights=positive_data.events.TotalWeight)
        negative_hist.fill(negative_values, negative_predictions, weights=negative_data.events.TotalWeight)

    combined_figure = plt.figure(figsize=(12, 6.15))
    combined_figure.suptitle(f"{title}{rig_title}")
    combined_plot = combined_figure.subplots(1, 1)
    plot_histogram_2d(combined_plot, combined_hist, log=True)

    if target_efficiency is not None:
        for contour_coords_x, contour_coords_y in create_histogram_contour(positive_hist, target_efficiency):
            combined_plot.add_line(plt.Line2D(contour_coords_x, contour_coords_y, color="tab:red"))
            combined_plot.add_line(plt.Line2D(-contour_coords_x, contour_coords_y, color="tab:red"))

    if cut_value_mva is not None:
        if signal_masses is not None and mass_target_efficiency is not None:
            mass_cut = derive_mass_cut(positive_data.events[variable], positive_data.events.TotalWeight, signal_masses, mass_target_efficiency)
            mass_cut_min = min(signal_masses) - mass_cut
            mass_cut_max = max(signal_masses) + mass_cut
            mva_window_top = mva_binning.edges[-2]
            print("mass cut value", mass_cut_min, mass_cut_max)
            rect_positive = Rectangle((mass_cut_min, cut_value_mva), mass_cut_max - mass_cut_min, mva_window_top - cut_value_mva, edgecolor="orange", facecolor="none", linewidth=1)
            rect_negative = Rectangle((-mass_cut_max, cut_value_mva), mass_cut_max - mass_cut_min, mva_window_top - cut_value_mva, edgecolor="orange", facecolor="none", linewidth=1)
            combined_plot.add_patch(rect_positive)
            combined_plot.add_patch(rect_negative)
            positive_window_selection = (positive_predictions >= cut_value_mva) & (positive_data.events[variable] >= mass_cut_min) & (positive_data.events[variable] <= mass_cut_max)
            positive_events_in_window = positive_data.events[positive_window_selection]
            negative_window_selection = (negative_predictions >= cut_value_mva) & (negative_data.events[variable] >= mass_cut_min) & (negative_data.events[variable] <= mass_cut_max)
            candidates_in_window = negative_data.events[negative_window_selection]
            n_positive_events_in_window = np.sum(positive_events_in_window.TotalWeight)
            n_candidates_in_window = np.sum(candidates_in_window.TotalWeight)
            combined_plot.text(mass_cut_min, mva_window_top, f" {format_order_of_magnitude(n_positive_events_in_window)} events", ha="left", va="top")
            combined_plot.text(-mass_cut_max, mva_window_top, f"{n_candidates_in_window:.0f} events ", ha="left", va="top")
        else:
            positive_events_above_cut = np.sum(positive_data.events[positive_predictions >= cut_value_mva].TotalWeight)
            negative_events_above_cut = np.sum(negative_data.events[negative_predictions >= cut_value_mva].TotalWeight)
            combined_plot.axhline(cut_value_mva, color="orange", alpha=0.75, linewidth=1)
            combined_plot.text(0, mva_binning.edges[-2], f"{negative_events_above_cut:.0f} events", ha="right", va="top")
            combined_plot.text(0, mva_binning.edges[-2], f"{format_order_of_magnitude(positive_events_above_cut)} events", ha="left", va="top")


    combined_plot.axvline(0, color="gray", alpha=0.5, linewidth=0.5)

    if is_log:
        major_ticks = 10**np.arange(np.floor(np.log10(raw_edges[0])), np.ceil(np.log10(raw_edges[-1])))
        minor_ticks = [factor * tick for tick in major_ticks for factor in range(1, 10)]
        all_ticks = [-tick for tick in major_ticks[::-1] if tick > raw_edges[0]] + [tick for tick in major_ticks if tick >= raw_edges[0]]
        all_minor_ticks = [-tick for tick in minor_ticks[::-1] if tick > raw_edges[0]] + [tick for tick in minor_ticks if tick >= raw_edges[0]]

        formatter = LogFormatter()
        formatter.create_dummy_axis()
        tick_labels = [f"{'-' if tick < 0 else ''}{formatter.format_data(tick)}" for tick in all_ticks]
        tick_values = [np.sign(tick) * (transform(np.abs(tick)) + offset) for tick in all_ticks]
        minor_tick_values = [np.sign(tick) * (transform(np.abs(tick)) + offset) for tick in all_minor_ticks]
        combined_plot.set_xticks(tick_values)
        combined_plot.set_xticks(minor_tick_values, minor=True)
        combined_plot.set_xticklabels(tick_labels)

    save_figure(combined_figure, plotdir, f"{prefix}_mva_vs_{variable}")


    indices = np.arange(len(negative_data.events))
    indices_and_scores = zip(indices, negative_predictions)
    sorted_indices = np.array([t[0] for t in sorted(indices_and_scores, key=lambda t: t[1])])
    signal_like_indices = sorted_indices[:n_signallike]
    signal_like_events = negative_data.events[signal_like_indices]

    var_figure = plt.figure(figsize=(12, 6.15))
    var_figure.suptitle(f"{title}{rig_title}")
    var_plot = var_figure.subplots(1, 1)

    plot_histogram_1d(var_plot, WeightedHistogram.fill_direct((var_binning,), np.abs(positive_data.events[variable]), weights=positive_data.events.TotalWeight, labels=(f"{variable}",)), label=f"{positive_data.label} Positive", style="iss", log=True)
    plot_histogram_1d(var_plot, WeightedHistogram.fill_direct((var_binning,), np.abs(negative_data.events[variable]), weights=negative_data.events.TotalWeight, labels=(f"{variable}",)), label=f"{negative_data.label} Negative", style="iss", log=True)
    plot_histogram_1d(var_plot, WeightedHistogram.fill_direct((var_binning,), np.abs(positive_mc.events[variable]), weights=positive_mc.events.TotalWeight, labels=(f"{variable}",)), scale=positive_data.events.TotalWeight.sum() / positive_mc.events.TotalWeight.sum(), label=f"{positive_mc.label} Positive", style="mc", log=True)
    plot_histogram_1d(var_plot, WeightedHistogram.fill_direct((var_binning,), np.abs(negative_mc.events[variable]), weights=negative_mc.events.TotalWeight, labels=(f"{variable}",)), scale=negative_data.events.TotalWeight.sum() / negative_mc.events.TotalWeight.sum(), label=f"{negative_mc.label} Negative", style="mc", log=True)
    for monitor_dataset in positive_monitors:
        plot_histogram_1d(var_plot, WeightedHistogram.fill_direct((var_binning,), np.abs(monitor_dataset.events[variable]), weights=monitor_dataset.events.TotalWeight, labels=(f"{variable}",)), scale=positive_data.events.TotalWeight.sum() / monitor_dataset.events.TotalWeight.sum(), label=f"{monitor_dataset.label} Positive", style="mc", log=True)

    candidate_hist = WeightedHistogram.fill_direct((var_binning,), np.abs(signal_like_events[variable]), weights=signal_like_events.TotalWeight, labels=(f"{variable}",))
    plot_histogram_1d(var_plot, candidate_hist, label=f"{n_signallike} most signal like negative events", style="iss", log=True, markersize=12, markeredgecolor="black", markeredgewidth=0.5)

    signal_like_bin_indices = var_binning.get_indices(signal_like_events[variable])
    signal_like_bin_count = candidate_hist.values[signal_like_bin_indices]
    candidate_indices = np.arange(len(signal_like_events))
    for index, event, height in zip(candidate_indices, signal_like_events, signal_like_bin_count):
        bin_index = signal_like_bin_indices[index]
        others_indices = candidate_indices[signal_like_bin_indices==bin_index]
        if index == others_indices[0]:
            s = ",".join(map(str, others_indices))
            var_plot.text(event[variable], height * 1.5, s, fontsize=6, ha="center")

    var_plot.set_xlabel(labelling.get_label(variable))
    var_plot.legend()
    save_figure(var_figure, plotdir, f"{prefix}_candidate_comparison_{variable}")


def check_mva_correlations(positive_data, negative_data, positive_predictions, negative_predictions, positive_mc, negative_mc, positive_mc_predictions, negative_mc_predictions, variables, binnings, mva_binning, mva_label, title, rig_title, plotdir, prefix):
    for variable in variables:
        figure = plt.figure(figsize=(12, 6.15))
        figure.suptitle(f"{title}{rig_title}")
        plots = figure.subplots(2, 2).flatten()
        binning = binnings.variable_binnings[variable]
        for plot, dataset, predictions, label in zip(plots, (positive_data, negative_data, positive_mc, negative_mc), (positive_predictions, negative_predictions, positive_mc_predictions, negative_mc_predictions), (f"{positive_data.label} Positive", f"{negative_data.label} Negative", f"{positive_mc.label} Positive", f"{negative_mc.label} Negative")):
            values = dataset.events[variable]
            correlation, correlation_error = calculate_correlation_and_error(values, predictions)
            hist = WeightedHistogram.fill_direct((binning, mva_binning), values, predictions, weights=dataset.events.TotalWeight, labels=(f"{variable}", f"{mva_label}"))
            plot_histogram_2d(plot, hist, log=True)
            plot.set_title(f"{label} $\\rho={uncertainties.ufloat(correlation, correlation_error):P}$")
        save_figure(figure, plotdir, f"{prefix}_{variable}")


def perform_injection_study(signal_template_data, background_template_data, mva_binning, bdt, background_dataset, background_prediction_hist, injection_dataset, injection_events_min, injection_events_max, background_events, injection_steps, draw_toy, plotdir, prefix, title, rig_title, variables, prediction_cache, template_cache, kde_bandwidth, mva_function, draw_from_template=False, efficiency_data=None, target_efficiency=None, allow_deficit=False):
    injection_results = []
    injection_base_dataset = background_dataset
    for injection_amount in range(injection_events_min, injection_events_max + 1):
        for injection_index in range(injection_steps):
            print("Injection", injection_base_dataset.name, injection_dataset.name, injection_amount, injection_index)
            if draw_toy:
                injection_base_dataset = background_dataset.draw_toy(background_events, variables=variables, seed=injection_amount * SOME_PRIME_NUMBER + injection_index)
            injected_events = injection_dataset.draw_random(injection_amount, seed=injection_amount * SOME_PRIME_NUMBER + injection_index)
            injected_events.events["TotalWeight"] = 1
            injected_dataset = injection_base_dataset + injected_events
            injected_dataset.label = f"{injected_dataset.label} +{injection_amount}x{injection_dataset.label}"
            injected_dataset = injected_dataset.remove_weights()
            if draw_toy:
                if not draw_from_template:
                    toy_predictions = draw_random_from_hist(background_prediction_hist, background_events)
                    toy_hist = WeightedHistogram.fill_direct((prediction_cache["data_hist"].binnings[0],), toy_predictions, weights=np.ones(background_events), labels=prediction_cache["data_hist"].labels)
                else:
                    cache_entry = template_cache[(kde_bandwidth,)]
                    toy_template_hist = Histogram(cache_entry[2], values=cache_entry[5])
                    toy_predictions = draw_random_from_hist(toy_template_hist, background_events)
                    toy_hist = WeightedHistogram.fill_direct((prediction_cache["data_hist"].binnings[0],), toy_predictions, weights=np.ones(background_events), labels=prediction_cache["data_hist"].labels)
                if injection_amount > 0:
                    injected_predictions = mva_function(injected_events)
                    toy_hist.fill(injected_predictions, weights=np.ones_like(injected_predictions))
                else:
                    injected_predictions = np.zeros(0)
                prediction_cache["data_predictions"] = np.concatenate((toy_predictions, injected_predictions))
                prediction_cache["data_hist"] = toy_hist
            else:
                prediction_cache.pop("data_predictions")
                prediction_cache.pop("data_hist")
            try:
                injection_result, _, _ = perform_template_fit(negative_data=injected_dataset, positive_data=signal_template_data, signal_template_data=signal_template_data, background_template_data=background_template_data, mva_binning=mva_binning, bdt=bdt, variables=variables, plotdir=plotdir, prefix=f"{prefix}_{injection_dataset.name}_{injection_amount}_{injection_index}", title=f"{title}{rig_title} ({injection_amount} injected signal events)", rig_title=rig_title, use_kde=True, kde_bandwidth=kde_bandwidth, method="ubllh", template_cache=template_cache, prediction_cache=prediction_cache, signal_template_label="Data $R>0$", efficiency_data=efficiency_data, target_efficiency=target_efficiency, data_label="Data", allow_deficit=allow_deficit, store_templates=False, reduce_number_of_plots=True)
                if injection_result["fit_valid"]:
                    injection_results.append((injection_amount, injection_index, injection_result))
            except RuntimeError:
                pass
    injection_result = dict(fits=injection_results)
    injected_amounts = []
    injected_offset = []
    fitted_signal_values = []
    fitted_signal_errors = []
    injection_amount_lin = np.arange(injection_events_min, injection_events_max + 1)
    injection_signal_values = {number: [] for number in injection_amount_lin}
    for amount, index, injection in injection_results:
        injected_amounts.append(amount)
        injected_offset.append((index / injection_steps - 0.5) / 3)
        fitted_signal_values.append(injection["template_fit"]["fit_param_values"]["signal_counts"])
        fitted_signal_errors.append(injection["template_fit"]["fit_param_errors"]["signal_counts"])
        injection_signal_values[amount].append(injection["template_fit"]["fit_param_values"]["signal_counts"])
    injection_signal_means = np.array([np.mean(injection_signal_values[amount]) for amount in injection_amount_lin])
    injection_signal_stds = np.array([np.std(injection_signal_values[amount], ddof=1) for amount in injection_amount_lin])
    injection_signal_mean_error = injection_signal_stds / np.sqrt(injection_steps)
    injection_signal_mean_error[injection_signal_mean_error < 0.1] = 0.1
    injection_linear_fit_valid, injection_linear_fit_parameters, injection_linear_fit_errors, injection_linear_fit_values, injection_linear_smooth_x, injection_linear_smooth_y  = fit_linear(injection_amount_lin, injection_signal_means, injection_signal_mean_error, cutoff=not allow_deficit)
    print(f"Deficit: {uncertainties.ufloat(injection_linear_fit_parameters['b'], injection_linear_fit_errors['b']):P} (valid={injection_linear_fit_valid})")

    injection_figure = plt.figure(figsize=(12, 6.15))
    injection_plot = injection_figure.subplots(1, 1)
    injection_figure.suptitle(f"{title}{rig_title} {injection_dataset.label} Signal Injection")
    injection_plot.set_xlabel("Injected Signal Events")
    injection_plot.set_ylabel("Fit Signal Counts")
    injection_plot.errorbar(np.array(injected_amounts) + np.array(injected_offset), np.array(fitted_signal_values), np.array(fitted_signal_errors), fmt=".", alpha=0.5, label="Fits")
    injection_plot.errorbar(injection_amount_lin, injection_signal_means, injection_signal_mean_error, fmt=".", zorder=3, label="Mean")
    injection_plot.plot(injection_amount_lin, injection_amount_lin, "-", label="Linear")
    injection_plot.plot(injection_linear_smooth_x, injection_linear_smooth_y, "-", label=f"a={uncertainties.ufloat(injection_linear_fit_parameters['a'], injection_linear_fit_errors['a']):P}\nb={uncertainties.ufloat(injection_linear_fit_parameters['b'], injection_linear_fit_errors['b']):P}")
    injection_plot.legend()
    save_figure(injection_figure, plotdir, f"{prefix}_{injection_dataset.name}_injection")
    injection_result["average"] = dict(injected=list(map(int, injection_amount_lin)), mean=list(injection_signal_means), mean_error=list(injection_signal_mean_error), cutoff=not allow_deficit)
    injection_result["linearity_fit"] = dict(parameters=injection_linear_fit_parameters, errors=injection_linear_fit_errors, valid=injection_linear_fit_valid)
    return injection_result




def train_estimator(variables, mc_train_signal, mc_train_background, mc_predict_signal, mc_predict_background, iss_signal, iss_background, monitors_signal, monitors_background, prefix, title, label, resultdir, plotdir, rig_title="", rig_range=None, comparison=None, max_depth=5, ntrees=30, eta=1, binnings=None, labelling=None, nprocesses=os.cpu_count(), compare_2d=None, use_train_weights=True, do_injection_study=False, injection_events_min=0, injection_events_max=10, injection_steps=3, injection_use_mc=True, injection_use_data=False, injection_use_monitors=False, injection_use_template=False, do_sensitivity_study=False, sensitivity_steps=100, do_kde_bandwidth_study=False, do_alternative_signal_template_study=False, data_derived_candidate_fits=None, attempt_2d_fit=None, isotope_fit_args=None, target_efficiency=0.9, mass_selection=None, mass_variable=None, signal_masses=None, reload_bdt=False, allow_deficit=False, kde_bandwidths=None):

    os.makedirs(plotdir, exist_ok=True)
    os.makedirs(resultdir, exist_ok=True)

    mc_train_signal_events = filter_branches(mc_train_signal.events, variables)
    mc_train_background_events = filter_branches(mc_train_background.events, variables)
    mc_predict_signal_events = filter_branches(mc_predict_signal.events, variables)
    mc_predict_background_events = filter_branches(mc_predict_background.events, variables)
    iss_signal_events = filter_branches(iss_signal.events, variables)
    iss_background_events = filter_branches(iss_background.events, variables)

    mc_train_signal_labels = np.ones(len(mc_train_signal_events))
    mc_train_background_labels = np.zeros(len(mc_train_background_events))
    mc_predict_signal_labels = np.ones(len(mc_predict_signal_events))
    mc_predict_background_labels = np.zeros(len(mc_predict_background_events))

    mc_train_events = recfunctions.stack_arrays((mc_train_signal_events, mc_train_background_events), asrecarray=True)
    mc_predict_events = recfunctions.stack_arrays((mc_predict_signal_events, mc_predict_background_events), asrecarray=True)
    mc_train_labels = np.concatenate((mc_train_signal_labels, mc_train_background_labels))
    mc_predict_labels = np.concatenate((mc_predict_signal_labels, mc_predict_background_labels))

    if use_train_weights:
        mc_train_signal_weights = mc_train_signal.events.TotalFlatWeight / mc_train_signal.events.TotalFlatWeight.mean()
        mc_train_background_weights = mc_train_background.events.TotalFlatWeight / mc_train_background.events.TotalFlatWeight.mean()
        mc_predict_signal_weights = mc_predict_signal.events.TotalFlatWeight / mc_predict_signal.events.TotalFlatWeight.mean()
        mc_predict_background_weights = mc_predict_background.events.TotalFlatWeight / mc_predict_background.events.TotalFlatWeight.mean()
    else:
        mc_train_signal_weights = np.ones_like(mc_train_signal_labels)
        mc_train_background_weights = np.ones_like(mc_train_background_labels)
        mc_predict_signal_weights = np.ones_like(mc_predict_signal_labels)
        mc_predict_background_weights = np.ones_like(mc_predict_background_labels)
    mc_train_weights = np.concatenate((mc_train_signal_weights, mc_train_background_weights))
    mc_predict_weights = np.concatenate((mc_predict_signal_weights, mc_predict_background_weights))

    dtrain = xgb.DMatrix(rec_to_float(mc_train_events), label=mc_train_labels, weight=mc_train_weights, feature_names=variables)
    dpredict = xgb.DMatrix(rec_to_float(mc_predict_events), label=mc_predict_labels, weight=mc_predict_weights, feature_names=variables)

    evallist = [(dpredict, "eval"), (dtrain, "train")]

    train_params = {"max_depth": max_depth, "eta": eta, "objective": "binary:logitraw", "nthread": 4, "eval_metric": "auc"}


    os.makedirs(resultdir, exist_ok=True)
    bdt_model_path = os.path.abspath(os.path.join(resultdir, f"{prefix}_model.json"))
    if not reload_bdt or not os.path.exists(bdt_model_path):
        print("Starting training", flush=True)
        bdt = xgb.train(train_params, dtrain, num_boost_round=ntrees, evals=evallist)#, early_stopping_rounds=3)
        print(f"Saving model to {bdt_model_path!r}", flush=True)
        bdt.save_model(bdt_model_path)
    else:
        print(f"Loading model from {bdt_model_path!r}", flush=True)
        bdt = xgb.Booster()
        bdt.load_model(bdt_model_path)

    def _predict(dataset):
        return predict_bdt(bdt, dataset, variables)

    print("Making predictions")

    mc_train_signal_predictions = _predict(mc_train_signal)
    mc_train_background_predictions = _predict(mc_train_background)
    mc_predict_signal_predictions = _predict(mc_predict_signal)
    mc_predict_background_predictions = _predict(mc_predict_background)

    min_score = min(np.min(mc_train_signal_predictions), np.min(mc_train_background_predictions))
    max_score = max(np.max(mc_train_signal_predictions), np.max(mc_train_background_predictions))
    score_range = max_score - min_score
    min_score -= score_range / 20
    max_score += score_range / 20
    mva_nbins = min(max(len(iss_background_events) // 6, 25), 100)

    if target_efficiency is None:
        print(f"Using MVA binning with {mva_nbins} bins")
        mva_binning = make_lin_binning(min_score, max_score, mva_nbins)
        very_fine_mva_binning = make_Lin_binning(min_score, max_score, 2000)
    else:
        very_fine_binning = make_lin_binning(min_score, max_score, 2000)
        very_fine_signal_spectrum = prediction_hist(mc_predict_signal, mc_predict_signal_predictions, very_fine_binning, title)
        target_cut_value = calculate_cut_value_for_efficiency(very_fine_signal_spectrum, target_efficiency)
        mva_binning = make_lin_binning_with_known_edge(min_score, max_score, mva_nbins, target_cut_value)
        very_fine_mva_binning = make_lin_binning_with_known_edge(min_score, max_score, 2000, target_cut_value)

    def _prediction_hist(dataset, binning=mva_binning):
        return prediction_hist(dataset, _predict(dataset), binning, f"{title}")

    mc_train_signal_histogram = _prediction_hist(mc_train_signal)
    mc_train_background_histogram = _prediction_hist(mc_train_background)
    mc_predict_signal_histogram = _prediction_hist(mc_predict_signal)
    mc_predict_background_histogram = _prediction_hist(mc_predict_background)
    very_fine_mc_predict_signal_histogram = _prediction_hist(mc_predict_signal, binning=very_fine_mva_binning)
    very_fine_mc_predict_background_histogram = _prediction_hist(mc_predict_background, binning=very_fine_mva_binning)
    iss_signal_histogram = _prediction_hist(iss_signal)
    iss_background_histogram = _prediction_hist(iss_background)

    iss_signal_values = _predict(iss_signal)
    iss_background_values = _predict(iss_background)
    mc_predict_signal_values = _predict(mc_predict_signal)
    mc_predict_background_values = _predict(mc_predict_background)

    result_data = {}

    rejection_comparison = None
    efficiency_comparison = None
    comparison_label = None
    if comparison is not None:
        comparison_data, comparison_label = comparison
        rejection_comparison = (comparison_label, comparison_data["rejection"])
        efficiency_comparison = (comparison_label, comparison_data["efficiency"])
    rejection_data = calculate_rejection(mc_predict_signal_histogram, mc_predict_background_histogram, plotdir=plotdir, prefix=prefix, title=title, label=label, rig_title=rig_title, comparison=rejection_comparison)
    target_efficiencies = [0.5, 0.8, 0.9, 0.95, 0.99]
    if target_efficiency is not None and target_efficiency not in target_efficiencies:
        target_efficiencies.append(target_efficiency)
    efficiency_data = calculate_efficiency(very_fine_mc_predict_signal_histogram, very_fine_mc_predict_background_histogram, plotdir=plotdir, prefix=prefix, title=title, label=label, rig_title=rig_title, comparison=efficiency_comparison, target_efficiencies=target_efficiencies)
    result_data["rejection"] = rejection_data
    result_data["efficiency"] = efficiency_data

    result_data["hists"] = {
        "binning": list(mva_binning.edges),
        "mc_train_signal": (list(mc_train_signal_histogram.values), list(mc_train_signal_histogram.squared_values)),
        "mc_train_background": (list(mc_train_background_histogram.values), list(mc_train_background_histogram.squared_values)),
        "mc_predict_signal": (list(mc_predict_signal_histogram.values), list(mc_predict_signal_histogram.squared_values)),
        "mc_predict_background": (list(mc_predict_background_histogram.values), list(mc_predict_background_histogram.squared_values)),
        "iss_signal": (list(iss_signal_histogram.values), list(iss_signal_histogram.squared_values)),
        "iss_background": (list(iss_background_histogram.values), list(iss_background_histogram.squared_values)),
    }

    spectrum_figure = plt.figure(figsize=(12, 6.15))
    spectrum_plot = spectrum_figure.subplots(1, 1)
    spectrum_plot.set_title(f"{title} spectrum{rig_title}")
    spectrum_plot.set_yscale("log")
    mc_signal_scale = iss_signal_histogram.values.sum() / mc_predict_signal_histogram.values.sum()
    mc_background_scale = iss_background_histogram.values.sum() / mc_predict_background_histogram.values.sum()
    plot_histogram_1d(spectrum_plot, mc_predict_signal_histogram, scale=mc_signal_scale, style="mc", label=f"{mc_predict_signal.label} Signal × {mc_signal_scale:.1f}", color=SIGNAL_COLOR)
    plot_histogram_1d(spectrum_plot, mc_predict_background_histogram, scale=mc_background_scale, style="mc", label=f"{mc_predict_background.label} Background × {mc_background_scale:.1f}", color=BACKGROUND_COLOR)
    plot_histogram_1d(spectrum_plot, iss_signal_histogram, style="iss", label=f"{iss_signal.label} Positive", color=SIGNAL_COLOR)
    plot_histogram_1d(spectrum_plot, iss_background_histogram, style="iss", label=f"{iss_background.label} Negative", color=BACKGROUND_COLOR)

    if efficiency_data is not None and target_efficiency is not None:
        signal_cut_value = efficiency_data["target_efficiency_data"][target_efficiency]["cut_value"]
        n_above_cut = np.sum(iss_background_values >= signal_cut_value).sum()
        spectrum_plot.axvline(signal_cut_value, color=CUT_COLOR, alpha=0.85, label=f"{100*target_efficiency:.0f}% efficient cut", linewidth=1, linestyle="dashed")

    spectrum_plot.set_xlabel(f"$\\Lambda_{{CC}}$")
    spectrum_plot.set_ylabel("Events")

    ymin, ymax = spectrum_plot.get_ylim()
    spectrum_plot.set_ylim(min(ymin, 0.1), ymax)
    spectrum_plot.legend()

    save_figure(spectrum_figure, plotdir, f"{prefix}_spectrum_nomon")

    for monitor in monitors_signal:
        mon_hist = _prediction_hist(monitor)
        scale = iss_signal_histogram.values.sum() / mon_hist.values.sum()
        plot_histogram_1d(spectrum_plot, _prediction_hist(monitor), scale=scale, style="mc", label=f"{monitor.label} Signal × {scale:.1f}", linewidth=1)
    for monitor in monitors_background:
        mon_hist = _prediction_hist(monitor)
        scale = iss_background_histogram.values.sum() / mon_hist.values.sum()
        plot_histogram_1d(spectrum_plot, _prediction_hist(monitor), scale=scale, style="mc", label=f"{monitor.label} Background × {scale:.1f}", linewidth=1)

    #for monitor in monitors_background:
    #    mon_hist = _prediction_hist(monitor)
    #    scale = iss_background_histogram.values.sum() / mon_hist.values.sum()
    #    plot_histogram_1d(spectrum_plot, _prediction_hist(monitor), scale=scale, style="mc", label=f"{monitor.label} Background")

    spectrum_plot.set_xlabel(f"$\\Lambda_{{CC}}$")
    ymin, ymax = spectrum_plot.get_ylim()
    spectrum_plot.set_ylim(min(ymin, 0.1), ymax)
    spectrum_plot.legend()
    save_figure(spectrum_figure, plotdir, f"{prefix}_spectrum")

    spectrum_train_predict_figure = plt.figure(figsize=(12, 6.15))
    spectrum_train_predict_plot = spectrum_train_predict_figure.subplots(1, 1)
    spectrum_train_predict_plot.set_xlabel(f"BDT {title}")
    spectrum_train_predict_plot.set_ylabel("MC Events")
    spectrum_train_predict_plot.set_title(f"{title} spectrum_train_predict{rig_title}")
    spectrum_train_predict_plot.set_yscale("log")
    plot_histogram_1d(spectrum_train_predict_plot, mc_train_signal_histogram, style="mc", label="Train Signal")
    plot_histogram_1d(spectrum_train_predict_plot, mc_train_background_histogram, style="mc", label="Train Background")
    plot_histogram_1d(spectrum_train_predict_plot, mc_predict_signal_histogram, style="mc", label="Predict Signal")
    plot_histogram_1d(spectrum_train_predict_plot, mc_predict_background_histogram, style="mc", label="Predict Background")
    ymin, ymax = spectrum_train_predict_plot.get_ylim()
    spectrum_train_predict_plot.set_ylim(min(ymin, 0.1), ymax)
    spectrum_train_predict_plot.legend()
    save_figure(spectrum_train_predict_figure, plotdir, f"{prefix}_spectrum_train_predict")

    spectrum_alt_figure = plt.figure(figsize=(8, 5.25))
    spectrum_alt_gridspec = GridSpec(2, 1, hspace=0.03)
    spectrum_alt_signal_plot = spectrum_alt_figure.add_subplot(spectrum_alt_gridspec[0,0])
    spectrum_alt_background_plot = spectrum_alt_figure.add_subplot(spectrum_alt_gridspec[1,0])#, sharex=spectrum_alt_signal_plot)
    if rig_range is not None:
        rig_min, rig_max = rig_range
        spectrum_alt_signal_plot.plot([], [], label=f"${rig_min:.0f}–{rig_max:.0f}$ GV", color="white")
    plot_histogram_1d(spectrum_alt_signal_plot, mc_predict_signal_histogram, scale=mc_signal_scale, style="mc", label="Signal (MC)", color="xkcd:sky blue", zorder=2.5)
    plot_histogram_1d(spectrum_alt_background_plot, mc_predict_background_histogram, scale=mc_background_scale, style="mc", label="Background (MC)", color="xkcd:orange", zorder=2.5)
    plot_histogram_1d(spectrum_alt_signal_plot, iss_signal_histogram, style="iss", label="AMS $R>0$", color="xkcd:bright blue", elinewidth=1, zorder=3)
    plot_histogram_1d(spectrum_alt_background_plot, iss_background_histogram, style="iss", label="AMS $R<0$", color="xkcd:red", elinewidth=1, zorder=3)

    spectrum_alt_signal_plot.set_xlabel("$\\Lambda_{CC}$")
    spectrum_alt_background_plot.set_xlabel("$\\Lambda_{CC}$")
    spectrum_alt_signal_plot.set_ylabel("Events")
    spectrum_alt_background_plot.set_ylabel("Events")
    spectrum_alt_signal_plot.set_ylim(bottom=0)
    spectrum_alt_background_plot.set_ylim(bottom=0)
    spectrum_alt_background_plot.set_xlim(-15, 25)
    spectrum_alt_signal_plot.set_xlim(-15, 25)
    background_plot_top = spectrum_alt_background_plot.get_ylim()[1]
    blinded_region_min = 10
    blinded_region_max = 20
    blinded_region_min = mva_binning.edges[mva_binning.edges >= blinded_region_min][0]
    blinded_region_max = mva_binning.edges[mva_binning.edges <= blinded_region_max][-1]
    spectrum_alt_background_plot.add_patch(Rectangle((blinded_region_min, 0), blinded_region_max - blinded_region_min, background_plot_top, facecolor="gray", zorder=4))
    spectrum_alt_background_plot.text((blinded_region_min + blinded_region_max) / 2, background_plot_top * 0.9, "Blinded region", zorder=4.5, color="white", ha="center", va="top")
    #spectrum_alt_signal_plot.get_xaxis().set_visible(False)
    spectrum_alt_signal_plot.set_xticks(spectrum_alt_signal_plot.get_xticks())
    spectrum_alt_signal_plot.set_xticklabels(["" for tick in spectrum_alt_signal_plot.get_xticks()])
    spectrum_alt_signal_plot.get_yaxis().get_major_formatter().set_useMathText(True)

    legend_handles_signal, legend_labels_signal = spectrum_alt_signal_plot.get_legend_handles_labels()
    legend_handles_background, legend_labels_background = spectrum_alt_background_plot.get_legend_handles_labels()
    legend_entries = list(zip(legend_handles_signal + legend_handles_background, legend_labels_signal + legend_labels_background))
    legend_entries = [legend_entries[0], legend_entries[2], legend_entries[4], legend_entries[1], legend_entries[3]]
    legend_handles, legend_labels = zip(*legend_entries)
    spectrum_alt_signal_plot.legend(legend_handles, legend_labels, loc="upper left")#, bbox_to_anchor=spectrum_alt_signal_plot.bbox)

    spectrum_alt_figure.subplots_adjust(left=0.075, top=0.96, bottom=0.1, right=0.975)
    save_figure(spectrum_alt_figure, plotdir, f"{prefix}_spectrum_alt", save_pdf=True)


    result_data["path"] = bdt_model_path
    result_data["variables"] = variables
    result_data["title"] = title
    result_data["min_score"] = min_score
    result_data["max_score"] = max_score
    result_data["rig_title"] = rig_title
    result_data["label"] = label
    if rig_range is not None:
        result_data["rigidity"] = rig_range

    total_gain = plot_feature_importance(bdt, variables, f"{title} Ranking{rig_title}", plotdir, prefix, labelling=labelling)
    result_data["total_gain"] = total_gain

    observables = sorted(set(variables) | set(compare_2d or []))

    check_mva_correlations(iss_signal, iss_background, iss_signal_values, iss_background_values, mc_predict_signal, mc_predict_background, mc_predict_signal_values, mc_predict_background_values, variables=observables, binnings=binnings, mva_binning=mva_binning, mva_label=label, title=title, rig_title=rig_title, plotdir=plotdir, prefix=f"{prefix}_correlation")

    print("Starting template fits", flush=True)

    template_cache = {}

    if kde_bandwidths is None:
        kde_bandwidths = [None]

    good_fit = False
    good_fit_index = -1
    fit_results = {}
    for kde_bandwidth in kde_bandwidths:
        kde_bandwidth_label = f"{kde_bandwidth * 1000:0>4.0f}"
        print(f"KDE Bandwidth {kde_bandwidth:.2f}")
        try:
            fit_result, template_cache, prediction_cache = perform_template_fit(negative_data=iss_background, positive_data=iss_signal, signal_template_data=iss_signal, background_template_data=mc_predict_background, mva_binning=mva_binning, bdt=bdt, variables=variables, plotdir=plotdir, prefix=f"{prefix}_llh_kde_{kde_bandwidth_label}", title=f"{title}{rig_title}", rig_title=rig_title, bdt_title=title, use_kde=True, method="ubllh", do_contour=True, template_cache=template_cache, attempt_2d_fit=attempt_2d_fit, efficiency_data=efficiency_data, target_efficiency=target_efficiency, signal_template_label="Data $R>0$", allow_deficit=allow_deficit, kde_bandwidth=kde_bandwidth)
            fit_results[kde_bandwidth] = dict(fit=fit_result)
            good_fit = True
            good_fit_index += 1
        except RuntimeError as error:
            print("Main template fit failed:", error)

        if good_fit:
            if good_fit_index == 0:
                if "cut_limit" in fit_result:
                    n_candidates = int(fit_result["cut_limit"][f"{target_efficiency},0.99"]["negative_events"])
                else:
                    n_candidates = max(10, int(fit_result["template_fit"]["fit_param_values"]["signal_counts"]) * 2, int(fit_result["confidence_intervals"]["event_count"]["sample_size"][2][1]))
                signal_like_events = iss_background.get_most_signal_like(iss_background_values, n_candidates)
                if any(np.isnan(event[variable]) for event in signal_like_events[0] for variable in variables):
                    print("Some variable is NaN:")
                    for event in signal_like_events[0]:
                        for variable in variables:
                            if np.isnan(event[variable]):
                                print(variable, event[variable])
                candidate_data = [(int(event.RunNumber), int(event.EventNumber), float(prediction), {variable: float_or_int(event[variable]) for variable in observables}) for event, prediction in sorted(zip(*signal_like_events), key=lambda t: -t[1])]
                result_data["candidates"] = candidate_data

            if isotope_fit_args is not None:
                print("Performing isotope template fits")
                isotope_signal_datasets, isotope_background_dataset, isotope_variables = isotope_fit_args
                perform_isotope_fits(iss_background.events, iss_background_values, isotope_signal_datasets, isotope_background_dataset, isotope_variables, binnings=binnings, plotdir=plotdir, prefix=f"{prefix}_isotope_fits_negative_{kde_bandwidth_label}", title=f"{title}{rig_title}", data_label=f"{iss_background.label} Negative")
                perform_isotope_fits(signal_like_events[0], signal_like_events[1], isotope_signal_datasets, isotope_background_dataset, isotope_variables, binnings=binnings, plotdir=plotdir, prefix=f"{prefix}_isotope_fits_candidate_{kde_bandwidth_label}", title=f"{title}{rig_title}", data_label=f"{n_candidates} most signal like negative events")
                #perform_isotope_fits(iss_signal.events, iss_signal_values, isotope_signal_datasets, isotope_background_dataset, isotope_variables, binnings=binnings, plotdir=plotdir, prefix=f"{prefix}_isotope_fits_positive", title=f"{title}{rig_title}", data_label=f"{iss_signal.label} Positive")

                _, mva_binning, fine_mva_binning, mva_coarse_background_template, mva_background_template, mva_coarse_signal_template, mva_signal_template = template_cache[(fit_result["kde_bandwidth"],)]
                perform_isotope_fits(iss_background.events, iss_background_values, isotope_signal_datasets, isotope_background_dataset, isotope_variables, binnings=binnings, plotdir=plotdir, prefix=f"{prefix}_isotope_fits_negative_2d", title=f"{title}{rig_title}", data_label=f"{iss_background.label} Negative", mva_templates=(mva_signal_template, mva_background_template, fine_mva_binning))

            if compare_2d is not None and good_fit_index == 0:
                print("Creating MVA vs variable plots")
                cut_value_mva = None
                if "cut_limit" in fit_result:
                    cut_value_mva = fit_result["cut_limit"][f"{target_efficiency},0.99"]["cut_value_signal"]
                for comparison_variable in compare_2d:
                    show_spectrum_2d(iss_signal, iss_background, iss_signal_values, iss_background_values, mc_predict_signal, mc_predict_background, mc_predict_signal_values, mc_predict_background_values, monitors_signal, comparison_variable, binnings, labelling=labelling, mva_binning=mva_binning, mva_label="$\\Lambda_{CC}$", title=title, rig_title=rig_title, plotdir=plotdir, prefix=prefix, n_signallike=n_candidates, cut_value_mva=cut_value_mva, target_efficiency=target_efficiency)

                if mass_selection is not None:
                    iss_signal_with_selection, iss_signal_values_with_selection = iss_signal.apply_selections(mass_selection, iss_signal_values)
                    iss_background_with_selection, iss_background_values_with_selection = iss_background.apply_selections(mass_selection, iss_background_values)
                    mc_predict_signal_with_selection, mc_predict_signal_values_with_selection = mc_predict_signal.apply_selections(mass_selection, mc_predict_signal_values)
                    mc_predict_background_with_selection, mc_predict_background_values_with_selection = mc_predict_background.apply_selections(mass_selection, mc_predict_background_values)
                    monitors_with_selection = [monitor.apply_selections(mass_selection)[0] for monitor in monitors_signal]
                    for comparison_variable in compare_2d:
                        mass_args = {}
                        if comparison_variable == mass_variable:
                            mass_args = dict(signal_masses=signal_masses, mass_target_efficiency=target_efficiency)
                        show_spectrum_2d(iss_signal_with_selection, iss_background_with_selection, iss_signal_values_with_selection, iss_background_values_with_selection, mc_predict_signal_with_selection, mc_predict_background_with_selection, mc_predict_signal_values_with_selection, mc_predict_background_values_with_selection, monitors_with_selection, comparison_variable, binnings, labelling=labelling, mva_binning=mva_binning, mva_label="$\\Lambda_{CC}$", title=f"{title} with RICH", rig_title=rig_title, plotdir=plotdir, prefix=f"{prefix}_with_selection", n_signallike=n_candidates, cut_value_mva=cut_value_mva, target_efficiency=target_efficiency, **mass_args)
                    

            if data_derived_candidate_fits is not None:
                fit_results[kde_bandwidth]["candidate_fits"] = data_derived_secondary_fit(positive_data=iss_signal, negative_data=iss_background, positive_predictions=iss_signal_values, negative_predictions=iss_background_values, variables=data_derived_candidate_fits, n_candidates=n_candidates, binnings=binnings, plotdir=plotdir, prefix=f"{prefix}_data_fits", title=f"{title}{rig_title}")

            if do_kde_bandwidth_study:
                fit_results[kde_bandwidth]["kde_bandwidth_study"] = {}
                for mod_factor in (0.5, 0.8, 0.9, 1.1, 1.2, 1.5, 2.0):
                    print(f"Modifying bandwidth by {mod_factor:.1f}")
                    factor_label = f"{int(mod_factor * 10):0>2}"
                    try:
                        fit_results[kde_bandwidth]["kde_bandwidth_study"][mod_factor], _, _ = perform_template_fit(negative_data=iss_background, positive_data=iss_signal, signal_template_data=iss_signal, background_template_data=mc_predict_background, mva_binning=mva_binning, bdt=bdt, variables=variables, plotdir=plotdir, prefix=f"{prefix}_llh_kde_{kde_bandwidth_label}_{factor_label}", title=f"{title}{rig_title} KDE x{mod_factor:.1f}", rig_title=rig_title, bdt_title=title, use_kde=True, method="ubllh", kde_bandwidth=mod_factor * fit_result["kde_bandwidth"], do_contour=True, template_cache=template_cache, signal_template_label="Data $R>0$", allow_deficit=allow_deficit)
                    except RuntimeError:
                        pass

            if do_alternative_signal_template_study:
                for mon_dataset in monitors_signal:
                    fit_results[kde_bandwidth][f"signal_template_{mon_dataset.name}"], _, _ = perform_template_fit(negative_data=iss_background, positive_data=iss_signal, signal_template_data=mon_dataset, background_template_data=mc_predict_background, mva_binning=mva_binning, bdt=bdt, variables=variables, plotdir=plotdir, prefix=f"{prefix}_llh_kde_{kde_bandwidth_label}_signal_template_{mon_dataset.name}", rig_title=rig_title, title=f"{title}{rig_title} KDE {signal_template.label}", bdt_title=title, use_kde=True, method="ubllh", kde_bandwidth=fit_result["kde_bandwidth"], do_contour=True, template_cache=None, signal_template_label="Data $R>0$", allow_deficit=allow_deficit)
                    pass

            if do_injection_study:
                print("Starting injection tests", flush=True)
                injection_results = {}

                injection_plotdir = os.path.join(plotdir, "injection")
                os.makedirs(injection_plotdir, exist_ok=True)

                if injection_use_mc:
                    injection_results["mc"] = perform_injection_study(signal_template_data=iss_signal, background_template_data=mc_predict_background, mva_binning=mva_binning, bdt=bdt, background_dataset=mc_predict_background, background_prediction_hist=mc_predict_background_histogram, injection_dataset=mc_predict_signal, injection_events_min=injection_events_min, injection_events_max=injection_events_max, background_events=len(iss_background.events), injection_steps=injection_steps, draw_toy=True, plotdir=injection_plotdir, prefix=f"{prefix}_llh_kde_{kde_bandwidth_label}_injection_mc", title=title, rig_title=rig_title, variables=variables, prediction_cache=prediction_cache, template_cache=template_cache, kde_bandwidth=fit_result["kde_bandwidth"], mva_function=_predict, efficiency_data=efficiency_data, target_efficiency=target_efficiency, allow_deficit=allow_deficit)
                if injection_use_data:
                    injection_results["iss"] = perform_injection_study(signal_template_data=iss_signal, background_template_data=mc_predict_background, mva_binning=mva_binning, bdt=bdt, background_dataset=iss_background, background_prediction_hist=iss_background_histogram, injection_dataset=mc_predict_signal, injection_events_min=injection_events_min, injection_events_max=injection_events_max, background_events=len(iss_background.events), injection_steps=injection_steps, draw_toy=True, plotdir=injection_plotdir, prefix=f"{prefix}_llh_kde_{kde_bandwidth_label}_injection_iss", title=title, rig_title=rig_title, variables=variables, prediction_cache=prediction_cache, template_cache=template_cache, kde_bandwidth=fit_result["kde_bandwidth"], mva_function=_predict, efficiency_data=efficiency_data, target_efficiency=target_efficiency, allow_deficit=allow_deficit)
                if injection_use_monitors:
                    for monitor_dataset in monitors_signal:
                        injection_results[monitor_dataset.name] = perform_injection_study(signal_template_data=iss_signal, background_template_data=mc_predict_background, mva_binning=mva_binning, bdt=bdt, background_dataset=mc_predict_background, background_prediction_hist=mc_predict_background_histogram, injection_dataset=monitor_dataset, injection_events_min=injection_events_min, injection_events_max=injection_events_max, background_events=len(iss_background.events), injection_steps=injection_steps, draw_toy=True, plotdir=injection_plotdir, prefix=f"{prefix}_llh_kde_{kde_bandwidth_label}_injection_mon", title=title, rig_title=rig_title, variables=variables, prediction_cache=prediction_cache, template_cache=template_cache, kde_bandwidth=fit_result["kde_bandwidth"], mva_function=_predict, efficiency_data=efficiency_data, target_efficiency=target_efficiency, allow_deficit=allow_deficit)
                if injection_use_template:
                    injection_results["mc"] = perform_injection_study(signal_template_data=iss_signal, background_template_data=mc_predict_background, mva_binning=mva_binning, bdt=bdt, background_dataset=mc_predict_background, background_prediction_hist=mc_predict_background_histogram, injection_dataset=mc_predict_signal, injection_events_min=injection_events_min, injection_events_max=injection_events_max, background_events=len(iss_background.events), injection_steps=injection_steps, draw_toy=True, plotdir=injection_plotdir, prefix=f"{prefix}_llh_kde_{kde_bandwidth_label}_injection_template", title=title, rig_title=rig_title, variables=variables, prediction_cache=prediction_cache, template_cache=template_cache, kde_bandwidth=fit_result["kde_bandwidth"], mva_function=_predict, efficiency_data=efficiency_data, target_efficiency=target_efficiency, allow_deficit=allow_deficit, draw_from_template=True)

                fit_results[kde_bandwidth]["injection"] = injection_results
            if do_sensitivity_study:
                print("Starting sensitivity test", flush=True)
                sensitivity_plotdir = os.path.join(plotdir, "sensitivity")
                os.makedirs(sensitivity_plotdir, exist_ok=True)
                sensitivity_results = {}
                sensitivity_results["fits"] = perform_injection_study(signal_template_data=iss_signal, background_template_data=mc_predict_background, mva_binning=mva_binning, bdt=bdt, background_dataset=mc_predict_background, background_prediction_hist=mc_predict_background_histogram, injection_dataset=mc_predict_signal, injection_events_min=0, injection_events_max=0, background_events=len(iss_background.events), injection_steps=sensitivity_steps, draw_toy=True, plotdir=sensitivity_plotdir, prefix=f"{prefix}_llh_kde_{kde_bandwidth_label}_sensitivity", title=title, rig_title=rig_title, variables=variables, prediction_cache=prediction_cache, template_cache=template_cache, kde_bandwidth=fit_result["kde_bandwidth"], mva_function=_predict, efficiency_data=efficiency_data, target_efficiency=target_efficiency, allow_deficit=allow_deficit)
                sigmas = range(1, 4)
                event_count_limits = {sigma: [] for sigma in sigmas}
                ratio_limits = {sigma: [] for sigma in sigmas}
                for amount, index, sensitivity_fit_result in sensitivity_results["fits"]["fits"]:
                    for sigma in sigmas:
                        event_count_limits[sigma].append(sensitivity_fit_result["confidence_intervals"]["event_count"]["constant"][sigma][1])
                        ratio_limits[sigma].append(sensitivity_fit_result["confidence_intervals"]["ratio"]["constant"][sigma][1])
                sensitivity_results["sensitivity"] = {}
                for sigma in sigmas:
                    limit_figure = plt.figure(figsize=(12, 6.15))
                    limit_plots = limit_figure.subplots(1, 2)
                    limit_figure.suptitle(f"{title}{rig_title}, ${sigma}\\sigma$ sensitivity")
                    event_count_binning = make_log_binning(0.1, 10, 50)
                    ratio_binning = make_log_binning(1e-9, 1e-7, 50)
                    event_count_plot, ratio_plot = limit_plots
                    event_count_plot.set_title(f"${sigma}\\sigma$")
                    ratio_plot.set_title(f"${sigma}\\sigma$")
                    event_limits = np.array(event_count_limits[sigma])
                    event_limit_hist = Histogram.fill_direct((event_count_binning,), event_limits, labels=("$N_{Events}$",))
                    event_limit_mean = np.mean(event_limits)
                    event_limit_std = np.std(event_limits, ddof=1)
                    event_limit_median = np.median(event_limits)
                    event_limit_quantile = (np.quantile(event_limits, norm.cdf(1)) - np.quantile(event_limits, norm.cdf(-1))) / 2
                    event_limit_quantile_upper = np.quantile(event_limits, norm.cdf(1)) - np.quantile(event_limits, norm.cdf(0))
                    event_limit_quantile_lower = np.quantile(event_limits, norm.cdf(0)) - np.quantile(event_limits, norm.cdf(-1))
                    print(f"sensitivity {sigma}σ events", f"{uncertainties.ufloat(event_limit_mean, event_limit_std):P} or {uncertainties.ufloat(event_limit_median, event_limit_quantile):P}", "cl", fit_result["confidence_intervals"]["event_count"]["sample_size"][sigma][1])
                    plot_histogram_1d(event_count_plot, event_limit_hist, style="mc")
                    event_count_plot.errorbar(event_limit_mean, 1, 0, event_limit_std, fmt=".", color="tab:orange")
                    event_count_plot.errorbar(event_limit_median, 2, 0, event_limit_quantile, fmt=".", color="tab:green")
                    event_count_plot.axvline(fit_result["confidence_intervals"]["event_count"]["sample_size"][sigma][1], color="tab:red")
                    event_count_plot.set_ylim(bottom=0)
                    event_count_plot.set_ylabel("Samples")

                    limits = np.array(ratio_limits[sigma])
                    ratio_limit_hist = Histogram.fill_direct((ratio_binning,), limits, labels=("Ratio",))
                    ratio_limit_mean = np.mean(limits)
                    ratio_limit_std = np.std(limits, ddof=1)
                    ratio_limit_median = np.median(limits)
                    ratio_limit_quantile = (np.quantile(limits, norm.cdf(1)) - np.quantile(limits, norm.cdf(-1))) / 2
                    ratio_limit_quantile_upper = np.quantile(limits, norm.cdf(1)) - np.quantile(limits, norm.cdf(0))
                    ratio_limit_quantile_lower = np.quantile(limits, norm.cdf(0)) - np.quantile(limits, norm.cdf(-1))
                    print(f"sensitivity {sigma}σ ratio", f"{uncertainties.ufloat(ratio_limit_mean, ratio_limit_std):P} or {uncertainties.ufloat(ratio_limit_median, ratio_limit_quantile):P}", "cl", fit_result["confidence_intervals"]["ratio"]["sample_size"][sigma][1])
                    plot_histogram_1d(ratio_plot, ratio_limit_hist, style="mc")
                    ratio_plot.errorbar(ratio_limit_mean, 1, 0, ratio_limit_std, fmt=".", color="tab:orange")
                    ratio_plot.errorbar(ratio_limit_median, 2, 0, ratio_limit_quantile, fmt=".", color="tab:green")
                    ratio_plot.axvline(fit_result["confidence_intervals"]["ratio"]["sample_size"][sigma][1], color="tab:red")
                    ratio_plot.set_ylim(bottom=0)
                    ratio_plot.set_ylabel("Samples")

                    save_figure(limit_figure, plotdir, f"{prefix}_llh_kde_{kde_bandwidth_label}_sensitivity_{sigma}_sigma")

                    sensitivity_results["sensitivity"][sigma] = {"events": dict(mean=event_limit_mean, std=event_limit_std, median=event_limit_median, quantile=event_limit_quantile, quantile_lower=event_limit_quantile_lower, quantile_upper=event_limit_quantile_upper), "ratio": dict(mean=ratio_limit_mean, std=ratio_limit_std, median=ratio_limit_median, quantile=ratio_limit_quantile, quantile_lower=ratio_limit_quantile_lower, quantile_upper=ratio_limit_quantile_upper)}
                fit_results[kde_bandwidth]["sensitivity"] = sensitivity_results

    result_data["fits"] = fit_results



    #for rebin_factor in (2, 4, 5, 10):
    #    coarse_mva_binning = reduce_bins(mva_binning, rebin_factor)
    #    #perform_template_fit(iss_background_histogram.rebin(coarse_mva_binning), iss_background_values, iss_signal_histogram.rebin(coarse_mva_binning), mc_predict_background_histogram.rebin(coarse_mva_binning), plotdir=plotdir, prefix=f"{prefix}_coarse_{rebin_factor}_llh", title=f"{title}{rig_title}, Rebinned x{rebin_factor} Binned Likelihood", smooth=False, method="llh")
    #    #perform_template_fit(iss_background_histogram.rebin(coarse_mva_binning), iss_background_values, iss_signal_histogram.rebin(coarse_mva_binning), mc_predict_background_histogram.rebin(coarse_mva_binning), plotdir=plotdir, prefix=f"{prefix}_coarse_{rebin_factor}_lsq", title=f"{title}{rig_title}, Rebinned x{rebin_factor} LeastSquares", smooth=False, method="lsq")
    #    perform_template_fit(iss_background_histogram.rebin(coarse_mva_binning), iss_background_values, iss_signal_histogram.rebin(coarse_mva_binning), iss_signal_values, mc_predict_background_histogram.rebin(coarse_mva_binning), mc_predict_background_values, plotdir=plotdir, prefix=f"{prefix}_coarse_{rebin_factor}_ubllh", title=f"{title}{rig_title}, Rebinned x{rebin_factor} Unbinned Likelihood", smooth=False, method="ubllh")

    if binnings is not None:
        signal_cut_value = calculate_cut_value_for_efficiency(mc_predict_signal_histogram, 0.9)
        print(f"Signal like: BDT > {signal_cut_value:.2f}")
        signal_like_signal = mc_predict_signal_events[mc_predict_signal_predictions >= signal_cut_value]
        signal_like_background  = mc_predict_background_events[mc_predict_background_predictions >= signal_cut_value]
        background_like_signal = mc_predict_signal_events[mc_predict_signal_predictions < signal_cut_value]
        background_like_background  = mc_predict_background_events[mc_predict_background_predictions < signal_cut_value]
        print(f"Signal: s={len(signal_like_signal)}, b={len(background_like_signal)}, t={len(mc_predict_signal_predictions)}")
        print(f"Background: s={len(signal_like_background)}, b={len(background_like_background)}, t={len(mc_predict_background_predictions)}")
        for variable in variables:
            var_cut_figure = plt.figure(figsize=(12, 6.15))
            signal_plot, background_plot = var_cut_figure.subplots(1, 2)
            signal_plot.set_title("Signal")
            background_plot.set_title("Background")
            signal_plot.set_xlabel(variable)
            background_plot.set_xlabel(variable)
            var_binning = binnings.variable_binnings[variable]
            signal_like_signal_histogram = Histogram.fill_direct((var_binning,), signal_like_signal[variable], labels=(variable,))
            background_like_signal_histogram = Histogram.fill_direct((var_binning,), background_like_signal[variable], labels=(variable,))
            signal_like_background_histogram = Histogram.fill_direct((var_binning,), signal_like_background[variable], labels=(variable,))
            background_like_background_histogram = Histogram.fill_direct((var_binning,), background_like_background[variable], labels=(variable,))
            if var_binning.log:
                signal_plot.set_xscale("log")
                background_plot.set_xscale("log")

            shaded_steps(signal_plot, var_binning.edges, np.zeros_like(signal_like_signal_histogram.values), signal_like_signal_histogram.values, color="green", label="Passed")
            shaded_steps(signal_plot, var_binning.edges, signal_like_signal_histogram.values, signal_like_signal_histogram.values + background_like_signal_histogram.values, color="red", label="Failed")
            shaded_steps(background_plot, var_binning.edges, np.zeros_like(signal_like_background_histogram.values), signal_like_background_histogram.values, color="green", label="Passed")
            shaded_steps(background_plot, var_binning.edges, signal_like_background_histogram.values, signal_like_background_histogram.values + background_like_background_histogram.values, color="red", label="Failed")

            signal_plot.legend()
            background_plot.legend()

            save_figure(var_cut_figure, plotdir, f"{prefix}_bdtcut_{variable}")


    print(result_data)
    #_print_dict(result_data)
    with open(os.path.join(f"{prefix}.pck"), "wb") as pickle_file:
        import pickle
        pickle.dump(result_data, pickle_file)
    with open(os.path.join(resultdir, f"{prefix}.json"), "w") as model_file:
        json.dump(result_data, model_file)

    return bdt, result_data


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-sample", required=True, help="Name of signal sample.")
    parser.add_argument("--background-sample", required=True, help="Name of background sample.")
    parser.add_argument("--iss-signal", required=True, nargs=6, help="Label and path to ISS signal histograms, events and selections.")
    parser.add_argument("--iss-background", required=True, nargs=6, help="Label and path to ISS background histograms, events and selections.")
    parser.add_argument("--mc-train-signal", dest="mc_train_signal_datasets", action="append", required=True, nargs=9, help="Label and path to MC signal histograms, events and selections for training.")
    parser.add_argument("--mc-train-background", dest="mc_train_background_datasets", action="append", required=True, nargs=9, help="Label and path to MC background histograms, events and selections for training.")
    parser.add_argument("--mc-predict-signal", dest="mc_predict_signal_datasets", action="append", required=True, nargs=9, help="Label and path to MC signal histograms, events and selections for prediction.")
    parser.add_argument("--mc-predict-background", dest="mc_predict_background_datasets", action="append", required=True, nargs=9, help="Label and path to MC background histograms, events and selections for prediction.")
    parser.add_argument("--monitor-signal", nargs=9, dest="signal_monitors", action="append", help="Path to signal dataset used for monitoring, not training")
    parser.add_argument("--monitor-background", nargs=9, dest="background_monitors", action="append", help="Path to background dataset used for monitoring, not training")
    parser.add_argument("--mc-weighting-file", required=True, help="Path to weight file (same as used for histograms, for weighting triggers)")
    parser.add_argument("--variables", required=True, nargs="+", help="Variables to use in MVA.")
    parser.add_argument("--compare-2d", nargs="+", help="Variables to plot MVA output against.")
    parser.add_argument("--resultdir", default="results", help="Directory to store result files in.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--config", nargs=2, required=True, help="Path to config file and workdir")
    parser.add_argument("--outputprefix", default="mva", help="Outputprefix for results and plots.")
    parser.add_argument("--title", default="MVA", help="MVA title for plots")
    parser.add_argument("--bin", type=int, help="Rigidity bin of this data.")
    parser.add_argument("--binning", default="rigidity_search", help="Rigidity binning by which training is split.")
    parser.add_argument("--train-without-weights", dest="train_with_weights", action="store_false", help="Don't use event weights in training.")
    parser.add_argument("--test-without", default="no", choices=("no", "patterns", "variables", "all"), help="Also train estimator without individual variables and groups of variables.")
    parser.add_argument("--do-input-comparison", action="store_true", help="Create Signal/Background and ISS/MC comparison plots of the input variables.")
    parser.add_argument("--do-injection-study", dest="do_injection_study", action="store_true", help="Perform template fits with injection signal events (from MC)")
    parser.add_argument("--injection-events-min", type=int, default=0, help="Minimum number of events to inject in injection study.")
    parser.add_argument("--injection-events-max", type=int, default=15, help="Maximum number of events to inject in injection study.")
    parser.add_argument("--injection-steps", type=int, default=8, help="How often to retry the injection for each number of injected events.")
    parser.add_argument("--injection-use-mc", action="store_true", help="Try injecting signal events from signal MC.")
    parser.add_argument("--injection-use-data", action="store_true", help="Try injecting signal events from signal data.")
    parser.add_argument("--injection-use-monitors", action="store_true", help="Also try injecting signal events from monitor datasets.")
    parser.add_argument("--injection-use-template", action="store_true", help="Inject signal events into background drawn from template.")
    parser.add_argument("--do-sensitivity-study", dest="do_sensitivity_study", action="store_true", help="Perform template fits on pure MC background to calculate sensitivity.")
    parser.add_argument("--sensitivity-steps", type=int, default=100, help="How often to retry the template fit in the sensitivity study.")
    parser.add_argument("--do-kde-bandwidth-study", dest="do_kde_bandwidth_study", action="store_true", help="Perform template fits with modified KDE bandwidth.")
    #parser.add_argument("--kde-bandwidth-factors", type=float, nargs="+", help="Factors by which to modify the KDE bandwidth.")
    parser.add_argument("--kde-bandwidths", type=float, nargs="+", help="Values to use as the KDE bandwidth.")
    parser.add_argument("--mva-parameters", nargs=4, action="append", dest="mva_parameters", help="MVA parameter set (name, depth, number of trees, eta)")
    parser.add_argument("--data-derived-candidate-fits", nargs="+", help="Variables to perform fits on using data derived templates, after MVA template fit.")
    parser.add_argument("--attempt-2d-fit", help="Attemp a 2D template fit in MVA vs this variable.")
    parser.add_argument("--calculate-limit-from-cut", default=0.9, type=float, help="Target efficiency of cut to use.")
    parser.add_argument("--isotope-templates", nargs="+", help="Datasets to use as templates in isotope fits.")
    parser.add_argument("--isotope-variables", nargs="+", help="Variables to perform isotope fits in.")
    parser.add_argument("--mass-selection", nargs="+", help="Additional selections to optionally check candidate events against mass.")
    parser.add_argument("--mass-variable", help="Variable to use for mass candidate checks.")
    parser.add_argument("--allow-deficit", action="store_true", help="Allow deficit (negative number) of signal or background events in template fit.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel")
    parser.add_argument("--reload-mva", action="store_true", help="Reload MVA from file instead of training it from data.")

    args = parser.parse_args()

    config_file, workdir = args.config
    config = get_config(config_file)
    binnings = Binnings((config, workdir))
    labelling = VariableLabels(config=config, workdir=workdir, energy_estimator=config["samples"][args.signal_sample]["energy_estimator"])

    variables = args.variables
    branches = variables + ["TotalWeight", "TotalFlatWeight", "RunNumber", "EventNumber"] + (args.compare_2d or []) + (args.data_derived_candidate_fits or []) + (args.isotope_variables or []) + [f"PassesSelection{selection}" for selection in (args.mass_selection or [])]
    if args.mass_variable is not None:
        branches.append(args.mass_variable)
    if args.attempt_2d_fit:
        branches.append(args.attempt_2d_fit)
    branches = sorted(set(branches))

    title = args.title
    rig_title = ""
    rigidity_search_binning = binnings.special_binnings[args.binning]
    rig_min = rigidity_search_binning.edges[1]
    rig_max = rigidity_search_binning.edges[-2]
    if args.bin is not None:
        rig_min, rig_max = rigidity_search_binning.edges[args.bin], rigidity_search_binning.edges[args.bin + 1]
        rig_title = f", {rig_min:.2f} < |R| / GV <= {rig_max:.2f}"
        print(f"Rigidity range {rig_min:.2f} to {rig_max:.2f}")

    mc_weighting = load_mc_weighting(args.mc_weighting_file)
    rigidity_binning = binnings.special_binnings["rigidity_binning"]

    iss_signal = Dataset.load(args.iss_signal, variables, branches)
    iss_background = Dataset.load(args.iss_background, variables, branches)
    mc_train_signal_datasets = [Dataset.load(arg, variables, branches, rigidity_binning, mc_weighting) for arg in args.mc_train_signal_datasets]
    mc_train_signal = sum(mc_train_signal_datasets[1:], start=mc_train_signal_datasets[0])
    mc_train_background_datasets = [Dataset.load(arg, variables, branches, rigidity_binning, mc_weighting) for arg in args.mc_train_background_datasets]
    mc_train_background = sum(mc_train_background_datasets[1:], start=mc_train_background_datasets[0])
    mc_predict_signal_datasets = [Dataset.load(arg, variables, branches, rigidity_binning, mc_weighting) for arg in args.mc_predict_signal_datasets]
    mc_predict_signal = sum(mc_predict_signal_datasets[1:], start=mc_predict_signal_datasets[0])
    mc_predict_background_datasets = [Dataset.load(arg, variables, branches, rigidity_binning, mc_weighting) for arg in args.mc_predict_background_datasets]
    mc_predict_background = sum(mc_predict_background_datasets[1:], start=mc_predict_background_datasets[0])
    mc_signal = mc_train_signal + mc_predict_signal
    mc_background = mc_train_background + mc_predict_background
    signal_monitors = [Dataset.load(monitor, variables, branches, rigidity_binning, mc_weighting) for monitor in (args.signal_monitors or [])]
    background_monitors = [Dataset.load(monitor, variables, branches, rigidity_binning, mc_weighting) for monitor in (args.background_monitors or [])]

    print("Events", len(iss_signal.events), len(iss_background.events), len(mc_train_signal.events), len(mc_train_background.events), len(mc_predict_signal.events), len(mc_predict_background.events))

    signal_datasets_by_name = {
        iss_signal.name: iss_signal,
        mc_train_signal.name: mc_train_signal,
        mc_predict_signal.name: mc_predict_signal,
    }
    background_datasets_by_name = {
        iss_background.name: iss_background,
        mc_train_background.name: mc_train_background,
        mc_predict_background.name: mc_predict_background,
    }
    for monitor in signal_monitors:
        signal_datasets_by_name[monitor.name] = monitor
    for monitor in background_monitors:
        background_datasets_by_name[monitor.name] = monitor

    isotope_fit_args = None
    if args.isotope_templates is not None and args.isotope_variables is not None:
        isotope_signal_datasets = {name: signal_datasets_by_name[name] for name in args.isotope_templates}
        isotope_background_dataset = mc_predict_background
        isotope_fit_args = (isotope_signal_datasets, isotope_background_dataset, args.isotope_variables)

    signal_masses = None
    if args.mass_selection is not None and args.mass_variable is not None:
        signal_masses = [MC_PARTICLE_MASSES[MC_PARTICLE_IDS[particle]] for particle in config["analysis"]["signal_ids"]]

    plotdir = args.plotdir
    os.makedirs(plotdir, exist_ok=True)

    print("\n".join(["variables:"] + [f"{index:>3}: {variable}" for index, variable in enumerate(variables)]))

    agreement_plotdir = os.path.join(plotdir, "agreement")
    separation_plotdir = os.path.join(plotdir, "separation")
    likelihood_plotdir = os.path.join(plotdir, "likelihood")
    efficiency_plotdir = os.path.join(plotdir, "efficiency")
    os.makedirs(agreement_plotdir, exist_ok=True)
    os.makedirs(separation_plotdir, exist_ok=True)
    os.makedirs(likelihood_plotdir, exist_ok=True)
    os.makedirs(efficiency_plotdir, exist_ok=True)

    if args.do_input_comparison:
        input_comparison_results = {}
        input_comparison_results["signal_efficiency"] = compare_selection_efficiency([mc_signal] + signal_monitors, title=title, rig_title=rig_title, plotdir=efficiency_plotdir, prefix=f"{args.outputprefix}_signal_efficiency", rig_min=rig_min, rig_max=rig_max)
        input_comparison_results["background_efficiency"] = compare_selection_efficiency([mc_background], title=title, rig_title=rig_title, plotdir=efficiency_plotdir, prefix=f"{args.outputprefix}_background_efficiency", rig_min=rig_min, rig_max=rig_max)
        print(input_comparison_results)
        for variable in variables:
            agreement_figure = plt.figure(figsize=(16, 8.2))
            gridspec = GridSpec(6, 2, hspace=0)

            agreement_figure.suptitle(f"{title} {variable}{rig_title}")
            for column, (label, iss_dataset, mc_dataset, monitors, is_signal), in enumerate((("Signal", iss_signal, mc_signal, signal_monitors, True), ("Background", iss_background, mc_background, (), False))):
                lin_plot = agreement_figure.add_subplot(gridspec[:2,column])
                log_plot = agreement_figure.add_subplot(gridspec[2:4,column], sharex=lin_plot)
                res_plot = agreement_figure.add_subplot(gridspec[4:5,column], sharex=lin_plot)
                ratio_plot = agreement_figure.add_subplot(gridspec[5:6,column], sharex=lin_plot)

                lin_plot.set_title(label)
                ratio_plot.set_xlabel(variable)
                #lin_plot.set_ylabel(f"{label} Events")
                #log_plot.set_ylabel(f"{label} Events")
                res_plot.set_ylabel("(ISS-MC)/$\\sigma$")
                ratio_plot.set_ylabel("ISS/MC")

                iss_hist = iss_dataset.hists[variable]
                mc_hist = mc_dataset.hists[variable]

                mc_scale = iss_hist.values.sum() / mc_hist.values.sum()
                plot_histogram_1d(lin_plot, iss_hist, style="iss", label=iss_dataset.label, show_overflow=True)
                plot_histogram_1d(lin_plot, mc_hist, scale=mc_scale, style="mc", label=mc_dataset.label, show_overflow=True)
                plot_histogram_1d(log_plot, iss_hist, style="iss", label=iss_dataset.label, log=True, show_overflow=True)
                plot_histogram_1d(log_plot, mc_hist, scale=mc_scale, style="mc", label=mc_dataset.label, log=True, show_overflow=True)
                for monitor in monitors:
                    mon_hist = monitor.hists[variable]
                    mon_scale = iss_hist.values.sum() / mon_hist.values.sum()
                    plot_histogram_1d(lin_plot, mon_hist, scale=mon_scale, style="mc", label=monitor.label, show_overflow=True)
                    plot_histogram_1d(log_plot, mon_hist, scale=mon_scale, style="mc", label=monitor.label, log=True, show_overflow=True)

                iss_values = iss_hist.values
                iss_errors = iss_hist.get_errors()
                mc_values = mc_hist.values * mc_scale
                mc_errors = mc_hist.get_errors() * mc_scale
                mc_iss_difference = iss_values - mc_values
                mc_iss_error = np.sqrt(iss_errors**2 + mc_errors**2)
                mc_iss_residuals = mc_iss_difference / mc_iss_error
                mc_iss_ratio = iss_values / mc_values
                mc_iss_ratio_error = mc_iss_ratio * np.sqrt((iss_errors / iss_values)**2 + (mc_errors / mc_values)**2)
                res_plot.errorbar(iss_hist.binnings[0].bin_centers, mc_iss_residuals, mc_iss_error / mc_iss_error, fmt=".")
                res_plot.axhline(0, color="darkgray")
                #res_plot.set_ylim(-10, 10)
                ratio_plot.errorbar(iss_hist.binnings[0].bin_centers, mc_iss_ratio, mc_iss_ratio_error, fmt=".")
                ratio_plot.axhline(1, color="darkgray")
                ratio_plot.set_ylim(0, 3)

                lin_plot.legend()

            save_figure(agreement_figure, agreement_plotdir, f"{args.outputprefix}_{variable}")

            mc_signal_hist = mc_signal.hists[variable]
            mc_background_hist = mc_background.hists[variable]
            iss_signal_hist = iss_signal.hists[variable]
            iss_background_hist = iss_background.hists[variable]

            separation_figure = plt.figure(figsize=(12, 6.15))
            separation_plot = separation_figure.subplots(1, 1)
            separation_plot.set_title(f"{variable}{rig_title}")
            separation_plot.set_xlabel(variable)
            separation_plot.set_ylabel("Events (MC)")
            mc_bkg_scale = mc_signal_hist.values.sum() / mc_background_hist.values.sum()
            plot_histogram_1d(separation_plot, mc_signal_hist, style="mc", label="MC Signal")
            plot_histogram_1d(separation_plot, mc_background_hist, scale=mc_bkg_scale, style="mc", label="MC Background")
            separation_plot.legend()
            save_figure(separation_figure, separation_plotdir, f"{args.outputprefix}_{variable}")

            combined_figure = plt.figure(figsize=(12, 6.15))
            combined_plot = combined_figure.subplots(1, 1)

            combined_rebin_factor = get_rebin_factor(mc_background_hist.binnings[0])
            background_binning = reduce_bins(mc_background_hist.binnings[0], combined_rebin_factor)
            combined_mc_background_hist = mc_background_hist.rebin(background_binning)
            combined_iss_background_hist = iss_background_hist.rebin(background_binning)

            plot_histogram_1d(combined_plot, iss_signal_hist, scale=1 / iss_signal_hist.values.sum(), style="iss", label="Data $R>0$", color=SIGNAL_COLOR, show_overflow=True)
            plot_histogram_1d(combined_plot, mc_signal_hist, scale=1 / mc_signal_hist.values.sum(), style="mc", label=f"MC $R>0$", color=SIGNAL_COLOR, show_overflow=True)
            plot_histogram_1d(combined_plot, combined_iss_background_hist, scale=1 / combined_iss_background_hist.values.sum() / combined_rebin_factor, style="iss", label=f"Data $R<0$", color=BACKGROUND_COLOR, show_overflow=True)
            plot_histogram_1d(combined_plot, combined_mc_background_hist, scale=1 / combined_mc_background_hist.values.sum() / combined_rebin_factor, style="mc", label=f"MC $R<0$", color=BACKGROUND_COLOR, show_overflow=True)
            combined_plot.set_title(f"{title}{rig_title}")
            combined_plot.set_xlabel(labelling.get_label(variable))
            combined_plot.set_ylabel("Normalized Events")
            combined_plot.set_ylim(bottom=0)
            combined_plot.legend()
            combined_figure.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9)
            save_figure(combined_figure, agreement_plotdir, f"{args.outputprefix}_{variable}_combined")

            calculate_rejection(mc_signal_hist, mc_background_hist, separation_plotdir, f"{args.outputprefix}_{variable}", title=variable, label=variable, rig_title=rig_title)
            calculate_efficiency(mc_signal_hist, mc_background_hist, separation_plotdir, f"{args.outputprefix}_{variable}_log", title=variable, label=variable, rig_title=rig_title, log=True)
            calculate_efficiency(mc_signal_hist, mc_background_hist, separation_plotdir, f"{args.outputprefix}_{variable}_lin", title=variable, label=variable, rig_title=rig_title, log=False)

        # evaluate separation
        for variable in variables:
            signal_distribution = mc_signal.hists[variable]
            mc_signal_likelihood = calculate_likelihood(signal_distribution, mc_signal.events[variable])
            mc_background_likelihood = calculate_likelihood(signal_distribution, mc_background.events[variable])
            iss_signal_likelihood = calculate_likelihood(signal_distribution, iss_signal.events[variable])
            iss_background_likelihood = calculate_likelihood(signal_distribution, iss_background.events[variable])

            max_likelihood = max(np.max(mc_signal_likelihood), np.max(mc_background_likelihood), np.max(iss_signal_likelihood), np.max(iss_background_likelihood))

            llh_binning = Binning(np.linspace(0, min(1, 1.1 * max_likelihood), 50))
            mc_signal_llh_hist = WeightedHistogram(llh_binning, labels=(f"{variable} likelihood",))
            mc_background_llh_hist = WeightedHistogram(llh_binning, labels=(f"{variable} likelihood",))
            iss_signal_llh_hist = WeightedHistogram(llh_binning, labels=(f"{variable} likelihood",))
            iss_background_llh_hist = WeightedHistogram(llh_binning, labels=(f"{variable} likelihood",))
            mc_signal_llh_hist.fill(mc_signal_likelihood, weights=mc_signal.events["TotalWeight"])
            mc_background_llh_hist.fill(mc_background_likelihood, weights=mc_background.events["TotalWeight"])
            iss_signal_llh_hist.fill(iss_signal_likelihood, weights=iss_signal.events["TotalWeight"])
            iss_background_llh_hist.fill(iss_background_likelihood, weights=iss_background.events["TotalWeight"])

            likelihood_figure = plt.figure(figsize=(12, 6.15))
            likelihood_plot = likelihood_figure.subplots(1, 1)
            likelihood_plot.set_title(f"{variable}{rig_title}")
            plot_histogram_1d(likelihood_plot, mc_signal_llh_hist, label=f"{mc_signal.label} Signal", style="mc")
            plot_histogram_1d(likelihood_plot, mc_background_llh_hist, label=f"{mc_background.label} Background", style="mc", scale=mc_signal_llh_hist.values.sum() / mc_background_llh_hist.values.sum())
            plot_histogram_1d(likelihood_plot, iss_signal_llh_hist, label=f"{iss_signal.label} Signal", style="iss", scale=mc_signal_llh_hist.values.sum() / iss_signal_llh_hist.values.sum())
            likelihood_plot.legend()
            save_figure(likelihood_figure, likelihood_plotdir, f"{args.outputprefix}_{variable}")

        with open(os.path.join(args.resultdir, f"{args.outputprefix}_input_comparison.json"), "w") as input_comparison_result_file:
            json.dump(input_comparison_results, input_comparison_result_file)

    if args.mva_parameters is not None:
        train_parameter_sets = {name: dict(ntrees=int(ntrees), max_depth=int(depth), eta=float(eta)) for name, depth, ntrees, eta in args.mva_parameters}

        for train_param_name, train_params in train_parameter_sets.items():
            print("BDT", train_param_name, flush=True)
            bdt_resultdir = os.path.join(args.resultdir, train_param_name)
            bdt_plotdir = os.path.join(plotdir, train_param_name)
            mc_all_bdt, mc_all_results = train_estimator(variables, mc_train_signal, mc_train_background, mc_predict_signal, mc_predict_background, iss_signal, iss_background, signal_monitors, background_monitors, prefix=f"{args.outputprefix}_all", title=title, label=f"{train_param_name}", rig_title=rig_title, rig_range=(rig_min, rig_max), resultdir=bdt_resultdir, plotdir=bdt_plotdir, binnings=binnings, labelling=labelling, nprocesses=args.nprocesses, do_injection_study=args.do_injection_study, injection_events_min=args.injection_events_min, injection_events_max=args.injection_events_max, injection_steps=args.injection_steps, injection_use_mc=args.injection_use_mc, injection_use_data=args.injection_use_data, injection_use_monitors=args.injection_use_monitors, injection_use_template=args.injection_use_template, do_sensitivity_study=args.do_sensitivity_study, sensitivity_steps=args.sensitivity_steps, do_kde_bandwidth_study=args.do_kde_bandwidth_study, compare_2d=args.compare_2d, data_derived_candidate_fits=args.data_derived_candidate_fits, attempt_2d_fit=args.attempt_2d_fit, isotope_fit_args=isotope_fit_args, target_efficiency=args.calculate_limit_from_cut, mass_selection=args.mass_selection, mass_variable=args.mass_variable, signal_masses=signal_masses, use_train_weights=args.train_with_weights, reload_bdt=args.reload_mva, allow_deficit=args.allow_deficit, kde_bandwidths=args.kde_bandwidths, **train_params)

            if args.test_without != "no":
                if args.test_without in ("variables", "all"):
                    for index, variable in enumerate(variables):
                        print(f"Without {variable}", flush=True)
                        variables_without = variables[:index] + variables[index+1:]
                        train_estimator(variables_without, mc_train_signal, mc_train_background, mc_predict_signal, mc_predict_background, iss_signal, iss_background, signal_monitors, background_monitors, prefix=f"{args.outputprefix}_without_{variable}", title=f"{title} without {variable}", label=f"{train_param_name} without {variable}", rig_title=rig_title, rig_range=(rig_min, rig_max), resultdir=bdt_resultdir, plotdir=bdt_plotdir, comparison=(mc_all_results, "MC all"), nprocesses=args.nprocesses, binnings=binnings, labelling=labelling, compare_2d=args.compare_2d, data_derived_candidate_fits=args.data_derived_candidate_fits, attempt_2d_fit=args.attempt_2d_fit, target_efficiency=args.calculate_limit_from_cut, mass_selection=args.mass_selection, mass_variable=args.mass_variable, signal_masses=signal_masses, use_train_weights=args.train_with_weights, reload_bdt=args.reload_mva, allow_deficit=args.allow_deficit, kde_bandwidths=args.kde_bandwidths, **train_params)

                if args.test_without in ("patterns", "all"):
                    for pattern in ("TrkChi2X*", "TrkChi2Y*", "TrkChi2*", "*Likelihood"):
                        print(f"Without {pattern}", flush=True)
                        matching = fnmatch(variables, pattern)
                        if matching:
                            remaining = sorted(set(variables) - set(matching))
                            joined = ", ".join(matching)
                            train_estimator(remaining, mc_train_signal, mc_train_background, mc_predict_signal, mc_predict_background, iss_signal, iss_background, signal_monitors, background_monitors, prefix=f"{args.outputprefix}_without_{pattern}", title=f"{title} without {pattern}", label=f"{train_param_name} without {joined}", rig_title=rig_title, rig_range=(rig_min, rig_max), resultdir=bdt_resultdir, plotdir=bdt_plotdir, comparison=(mc_all_results, "MC all"), nprocesses=args.nprocesses, binnings=binnings, labelling=labelling, compare_2d=args.compare_2d, data_derived_candidate_fits=args.data_derived_candidate_fits, attempt_2d_fit=args.attempt_2d_fit, target_efficiency=args.calculate_limit_from_cut, mass_selection=args.mass_selection, mass_variable=args.mass_variable, signal_masses=signal_masses, use_train_weights=args.train_with_weights, reload_bdt=args.reload_mva, allow_deficit=allow_deficit, kde_bandwidths=args.kde_bandwidths, **train_params)


if __name__ == "__main__":
    main()
