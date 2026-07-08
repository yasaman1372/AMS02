
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from uncertainties import ufloat

from .binnings import Binning
from .histograms import WeightedHistogram, plot_histogram_1d, plot_histogram_2d
from .statistics import calculate_efficiency, calculate_efficiency_error
from .utilities import format_order_of_magnitude, round_up, set_energy_ticks, make_colormap_passed, make_colormap_failed, save_figure


def constant(c, dtype=np.float32):
    def _constant(energy):
        return c * np.ones_like(energy, dtype=dtype)
    return _constant

def linear_log_energy(a, b):
    def _linear_log_energy(energy):
        return a * np.log10(energy) + b
    return _linear_log_energy

def broken_log_energy(a, b, c):
    def _broken_log_energy(energy):
        return np.maximum(c, a * np.log10(energy) + b)
    return _broken_log_energy

CUT_VALUE_FUNCTIONS = {
    "broken_log_energy": (broken_log_energy, "max({c}, {a}*log10(E/GeV)+{b})"),
    "linear_log_energy": (linear_log_energy, "{a}*log10(E/GeV)+{b}"),
}


def parse_cut_value(cut_value):
    if isinstance(cut_value, dict):
        function_maker, label_pattern = CUT_VALUE_FUNCTIONS[cut_value["function"]]
        parameters = cut_value["parameters"]
        return function_maker(**parameters), label_pattern.format(**parameters)
    elif isinstance(cut_value, float):
        return constant(cut_value, np.float32), f"{cut_value:.2g}"
    elif isinstance(cut_value, int):
        return constant(cut_value, np.int32), int(cut_value)
    elif isinstance(cut_value, bool):
        return constant(cut_value, bool), "True" if cut_value else "False"
    raise ValueError(f"Unable to parse cut value {cut_value!r} of type {type(cut_value)}.")


def cut_twosided(min_value_function, max_value_function):
    def _cut_twosided(values, energies):
        min_value = min_value_function(energies)
        max_value = max_value_function(energies)
        return np.all((values >= min_value, values <= max_value), axis=0)
    return _cut_twosided

def cut_greater(min_value_function):
    def _cut_greater(values, energies):
        min_value = min_value_function(energies)
        return values >= min_value
    return _cut_greater

def cut_lesser(max_value_function):
    def _cut_lesser(values, energies):
        max_value = max_value_function(energies)
        return values <= max_value
    return _cut_lesser

def cut_bool(target_function):
    def _cut_bool(values, energies):
        target = target_function(energies)
        return values == target
    return _cut_bool

def cut_pattern(mask_function, value_function):
    def _cut_pattern(values, energies):
        mask = mask_function(energies)
        value = value_function(energies)
        return (values & mask) == value
    return _cut_pattern

def combine_cuts_any(cuts):
    def _cut_any(values, energies):
        return np.any([cut(values, energies) for cut in cuts], axis=0)
    return _cut_any

def combine_cuts_all(cuts):
    def _cut_all(values, energies):
        return np.all([cut(values, energies) for cut in cuts], axis=0)
    return _cut_all


class Cut:
    def __init__(self, cut_function, passed_histogram, failed_histogram, passed_histogram_per_energy, failed_histogram_per_energy, passed_histogram_per_mc_momentum, failed_histogram_per_mc_momentum, label, cut_value_lines=None, tagger=None, fill_hists=True):
        self.cut_function = cut_function
        self.passed_histogram = passed_histogram
        self.failed_histogram = failed_histogram
        self.passed_histogram_per_energy = passed_histogram_per_energy
        self.failed_histogram_per_energy = failed_histogram_per_energy
        self.passed_histogram_per_mc_momentum = passed_histogram_per_mc_momentum
        self.failed_histogram_per_mc_momentum = failed_histogram_per_mc_momentum
        self.label = label
        self.cut_value_lines = cut_value_lines
        self.tagger = tagger
        self.fill_hists = fill_hists

    @staticmethod
    def create(cut_function, binning, energy_binning, label, variable, energy_estimator, cut_value_lines=None, tagger=None, fill_hists=True):
        return Cut(
            cut_function=cut_function,
            passed_histogram=WeightedHistogram(binning, labels=(variable,)),
            failed_histogram=WeightedHistogram(binning, labels=(variable,)),
            passed_histogram_per_energy=WeightedHistogram(energy_binning, binning, labels=(f"{energy_estimator} / GeV", variable)),
            failed_histogram_per_energy=WeightedHistogram(energy_binning, binning, labels=(f"{energy_estimator} / GeV", variable)),
            passed_histogram_per_mc_momentum=WeightedHistogram(energy_binning, binning, labels=(f"MC Momentum / GeV", variable)),
            failed_histogram_per_mc_momentum=WeightedHistogram(energy_binning, binning, labels=(f"MC Momentum / GeV", variable)),
            label=label,
            cut_value_lines=cut_value_lines,
            tagger=tagger,
            fill_hists=fill_hists)

    @staticmethod
    def load(cut_config, variable, binning, energy_binning, energy_estimator, config, workdir, fill_hists=True):
        if "min" in cut_config and "max" in cut_config:
            min_value_function, min_value_label = parse_cut_value(cut_config["min"])
            max_value_function, max_value_label = parse_cut_value(cut_config["max"])
            cut_func = cut_twosided(min_value_function=min_value_function, max_value_function=max_value_function)
            cut_value_lines = [min_value_function, max_value_function]
            label = f"{min_value_label} ≤ {variable} ≤ {max_value_label}"
        elif "min" in cut_config:
            min_value_function, min_value_label = parse_cut_value(cut_config["min"])
            cut_func = cut_greater(min_value_function=min_value_function)
            cut_value_lines = [min_value_function]
            label = f"{variable} ≥ {min_value_label}"
        elif "max" in cut_config:
            max_value_function, max_value_label = parse_cut_value(cut_config["max"])
            cut_func = cut_lesser(max_value_function=max_value_function)
            cut_value_lines = [max_value_function]
            label = f"{variable} ≤ {max_value_label}"
        elif "bool" in cut_config:
            value_function, value_label = parse_cut_value(cut_config["bool"])
            cut_func = cut_bool(target_function=value_function)
            cut_value_lines = []
            label = f"{variable} is {value_label}"
        elif "mask" in cut_config and "value" in cut_config:
            mask_function, mask_label = parse_cut_value(cut_config["mask"])
            value_function, value_label = parse_cut_value(cut_config["value"])
            cut_func = cut_pattern(mask_function=mask_function, value_function=value_function)
            cut_value_lines = []
            label = f"{variable} & {mask_label} = {value_label}"
        elif "any" in cut_config:
            cuts = [
                Cut.load(subcut_config, variable, binning, energy_binning, energy_estimator, config, workdir, fill_hists=fill_hists)
                for subcut_config in cut_config["any"]
            ]
            cut_func = combine_cuts_any([cut.cut_function for cut in cuts])
            cut_value_lines = [line for c in cuts for line in cuts.cut_value_lines]
            label = " or ".join([f"({cut.label})" for cut in cuts])
        elif "all" in cut_config:
            cuts = [
                Cut.load(subcut_config, variable, binning, energy_binning, energy_estimator, config, workdir, fill_hists=fill_hists)
                for subcut_config in cut_config["all"]
            ]
            cut_func = combine_cuts_all([cut.cut_function for cut in cuts])
            cut_value_lines = [line for c in cuts for line in cuts.cut_value_lines]
            label = " and ".join([f"({cut.label})" for cut in cuts])
        else:
            raise ValueError(f"Cannot parse cut {cut_config!r}.")
        tagger = cut_config.get("tag_cuts", None)
        return Cut.create(cut_func, binning, energy_binning, label, variable, energy_estimator, cut_value_lines=cut_value_lines, tagger=tagger, fill_hists=fill_hists)

    def select(self, values, energies, weights, mc_momentum=None):
        passed = self.cut_function(values, energies)
        if not self.fill_hists:
            return passed
        failed = np.invert(passed)
        self.passed_histogram.fill(values[passed], weights=weights[passed])
        self.failed_histogram.fill(values[failed], weights=weights[failed])
        self.passed_histogram_per_energy.fill(energies[passed], values[passed], weights=weights[passed])
        self.failed_histogram_per_energy.fill(energies[failed], values[failed], weights=weights[failed])
        if mc_momentum is not None:
            self.passed_histogram_per_mc_momentum.fill(mc_momentum[passed], values[passed], weights=weights[passed])
            self.failed_histogram_per_mc_momentum.fill(mc_momentum[failed], values[failed], weights=weights[failed])
        return passed

    def add_to_file(self, file_dict, name):
        self.passed_histogram.add_to_file(file_dict, f"{name}_passed")
        self.failed_histogram.add_to_file(file_dict, f"{name}_failed")
        self.passed_histogram_per_energy.add_to_file(file_dict, f"{name}_passed_per_energy")
        self.failed_histogram_per_energy.add_to_file(file_dict, f"{name}_failed_per_energy")
        self.passed_histogram_per_mc_momentum.add_to_file(file_dict, f"{name}_passed_per_mc_momentum")
        self.failed_histogram_per_mc_momentum.add_to_file(file_dict, f"{name}_failed_per_mc_momentum")
        file_dict[f"{name}_label"] = self.label

    @staticmethod
    def from_file(file_dict, name):
        return Cut(
            cut_function=None,
            passed_histogram=WeightedHistogram.from_file(file_dict, f"{name}_passed"),
            failed_histogram=WeightedHistogram.from_file(file_dict, f"{name}_failed"),
            passed_histogram_per_energy=WeightedHistogram.from_file(file_dict, f"{name}_passed_per_energy"),
            failed_histogram_per_energy=WeightedHistogram.from_file(file_dict, f"{name}_failed_per_energy"),
            passed_histogram_per_mc_momentum=WeightedHistogram.from_file(file_dict, f"{name}_passed_per_mc_momentum"),
            failed_histogram_per_mc_momentum=WeightedHistogram.from_file(file_dict, f"{name}_failed_per_mc_momentum"),
            label=file_dict[f"{name}_label"].item())

    def add(self, other):
        assert self.label == other.label
        self.passed_histogram.add(other.passed_histogram)
        self.failed_histogram.add(other.failed_histogram)
        self.passed_histogram_per_energy.add(other.passed_histogram_per_energy)
        self.failed_histogram_per_energy.add(other.failed_histogram_per_energy)
        self.passed_histogram_per_mc_momentum.add(other.passed_histogram_per_mc_momentum)
        self.failed_histogram_per_mc_momentum.add(other.failed_histogram_per_mc_momentum)

    def __iadd__(self, other):
        self.add(other)
        return self

    def __add__(self, other):
        assert self.label == other.label
        return Cut(
            cut_function=None,
            passed_histogram=self.passed_histogram + other.passed_histogram,
            failed_histogram=self.failed_histogram + other.failed_histogram,
            passed_histogram_per_energy=self.passed_histogram_per_energy + other.passed_histogram_per_energy,
            failed_histogram_per_energy=self.failed_histogram_per_energy + other.failed_histogram_per_energy,
            passed_histogram_per_mc_momentum=self.passed_histogram_per_mc_momentum + other.passed_histogram_per_mc_momentum,
            failed_histogram_per_mc_momentum=self.failed_histogram_per_mc_momentum + other.failed_histogram_per_mc_momentum,
            label=self.label,
            tagger=self.tagger)

    def has_mc_data(self):
        return self.passed_histogram_per_mc_momentum.values.sum() + self.failed_histogram_per_mc_momentum.values.sum() > 0

    def get_efficiency(self):
        passed = self.passed_histogram.values.sum()
        failed = self.failed_histogram.values.sum()
        all = passed + failed
        return calculate_efficiency(passed, all), calculate_efficiency_error(passed, all)

    def get_efficiency_per_energy(self):
        passed = self.passed_histogram_per_energy.values.sum(axis=1)
        failed = self.failed_histogram_per_energy.values.sum(axis=1)
        all = passed + failed
        return calculate_efficiency(passed, all), calculate_efficiency_error(passed, all)

    def get_efficiency_per_mc_momentum(self):
        passed = self.passed_histogram_per_mc_momentum.values.sum(axis=1)
        failed = self.failed_histogram_per_mc_momentum.values.sum(axis=1)
        all = passed + failed
        return calculate_efficiency(passed, all), calculate_efficiency_error(passed, all)

    def plot(self, plot, style="mc", **kwargs):
        plot_histogram_1d(plot, self.passed_histogram, style=style, color="green", label=f"passed ({self.passed_histogram.values.sum():.3e})", show_overflow=True, **kwargs)
        plot_histogram_1d(plot, self.failed_histogram, style=style, color="red", label=f"failed ({self.failed_histogram.values.sum():.3e})", show_overflow=True, **kwargs)
        efficiency = ufloat(*self.get_efficiency())
        plot.plot([np.nan], [np.nan], color="white", label=f"$\\epsilon={efficiency:L}$")
        plot.plot([np.nan], [np.nan], color="white", label=f"Total Events: {self.passed_histogram.values.sum()+self.failed_histogram.values.sum():.3e}")

    def plot_efficiency_per_energy(self, plot, style="mc", **kwargs):
        efficiency, efficiency_error = self.get_efficiency_per_energy()
        energy_bin_centers = self.passed_histogram_per_energy.binnings[0].bin_centers
        facecolor = "none" if style == "mc" else None
        plot.errorbar(energy_bin_centers, efficiency, efficiency_error, marker="o", mfc=facecolor, **kwargs)

    def plot_efficiency_per_mc_momentum(self, plot, style="mc", **kwargs):
        efficiency, efficiency_error = self.get_efficiency_per_mc_momentum()
        energy_bin_centers = self.passed_histogram_per_energy.binnings[0].bin_centers
        facecolor = "none" if style == "mc" else None
        plot.errorbar(energy_bin_centers, efficiency, efficiency_error, marker="o", mfc=facecolor, **kwargs)

    def plot_per_energy(self, plot, **kwargs):
        scale = 1 / (self.passed_histogram_per_energy.values.sum(axis=1) + self.failed_histogram_per_energy.values.sum(axis=1))
        positive_passed_values = self.passed_histogram_per_energy.values[1:-1,:] * np.expand_dims(scale[1:-1], 1)
        positive_failed_values = self.failed_histogram_per_energy.values[1:-1,:] * np.expand_dims(scale[1:-1], 1)
        positive_passed_values = positive_passed_values[positive_passed_values > 0]
        positive_failed_values = positive_failed_values[positive_failed_values > 0]
        max_value = 0
        min_value = 1
        if len(positive_passed_values) > 0:
            max_value = max(positive_passed_values.max(), max_value)
            min_value = min(positive_passed_values.min(), min_value)
        if len(positive_failed_values) > 0:
            max_value = max(positive_failed_values.max(), max_value)
            min_value = min(positive_failed_values.min(), min_value)
        plot_histogram_2d(plot, self.passed_histogram_per_energy, show_overflow_x=False, min_value=min_value, max_value=max_value, scale=scale, cmap=make_colormap_passed(), show_overflow=True, log=True)
        plot_histogram_2d(plot, self.failed_histogram_per_energy, show_overflow_x=False, min_value=min_value, max_value=max_value, scale=scale, cmap=make_colormap_failed(), show_overflow=True, log=True)
        if self.cut_value_lines is not None:
            energy_values = self.passed_histogram_per_energy.binnings[0].bin_centers
            for cut_value_function in self.cut_value_lines:
                plot.plot(energy_values, cut_value_function(energy_values), "k-")
        set_energy_ticks(plot)

    def plot_per_mc_momentum(self, plot, **kwargs):
        scale = 1 / (self.passed_histogram_per_mc_momentum.values.sum(axis=1) + self.failed_histogram_per_mc_momentum.values.sum(axis=1))
        positive_passed_values = self.passed_histogram_per_mc_momentum.values[1:-1,:] * np.expand_dims(scale[1:-1], 1)
        positive_failed_values = self.failed_histogram_per_mc_momentum.values[1:-1,:] * np.expand_dims(scale[1:-1], 1)
        positive_passed_values = positive_passed_values[positive_passed_values > 0]
        positive_failed_values = positive_failed_values[positive_failed_values > 0]
        max_value = 0
        min_value = 1
        if len(positive_passed_values) > 0:
            max_value = max(positive_passed_values.max(), max_value)
            min_value = min(positive_passed_values.min(), min_value)
        if len(positive_failed_values) > 0:
            max_value = max(positive_failed_values.max(), max_value)
            min_value = min(positive_failed_values.min(), min_value)
        plot_histogram_2d(plot, self.passed_histogram_per_mc_momentum, show_overflow_x=False, min_value=min_value, max_value=max_value, scale=scale, cmap=make_colormap_passed(), show_overflow=True, log=True)
        plot_histogram_2d(plot, self.failed_histogram_per_mc_momentum, show_overflow_x=False, min_value=min_value, max_value=max_value, scale=scale, cmap=make_colormap_failed(), show_overflow=True, log=True)
        set_energy_ticks(plot)


class Selection:
    def __init__(self, cuts, energy_estimator, energy_binning, total_passed_histogram=None, total_failed_histogram=None, total_passed_mc_histogram=None, total_failed_mc_histogram=None, after_all_other_passed_histograms=None, after_all_other_failed_histograms=None, fill_hists=True):
        self.cuts = cuts
        self.energy_estimator = energy_estimator
        self.energy_binning = energy_binning
        if total_passed_histogram is None:
            total_passed_histogram = WeightedHistogram(energy_binning)
        self.total_passed_histogram = total_passed_histogram
        if total_failed_histogram is None:
            total_failed_histogram = WeightedHistogram(energy_binning)
        self.total_failed_histogram = total_failed_histogram
        if total_passed_mc_histogram is None:
            total_passed_mc_histogram = WeightedHistogram(energy_binning)
        self.total_passed_mc_histogram = total_passed_mc_histogram
        if total_failed_mc_histogram is None:
            total_failed_mc_histogram = WeightedHistogram(energy_binning)
        self.total_failed_mc_histogram = total_failed_mc_histogram
        if after_all_other_passed_histograms is None:
            after_all_other_passed_histograms = {
                cut_name: WeightedHistogram(cut.passed_histogram.binnings[0])
                for cut_name, cut in cuts.items()
            }
        self.after_all_other_passed_histograms = after_all_other_passed_histograms
        if after_all_other_failed_histograms is None:
            after_all_other_failed_histograms = {
                cut_name: WeightedHistogram(cut.failed_histogram.binnings[0])
                for cut_name, cut in cuts.items()
            }
        self.after_all_other_failed_histograms = after_all_other_failed_histograms
        self.fill_hists = fill_hists


    @staticmethod
    def load(selection_config, energy_estimator, binnings, config, workdir, fill_hists=True):
        cuts = {}
        energy_binning = binnings["energy"]
        for variable, cut_config in selection_config["cuts"].items():
            cuts[variable] = Cut.load(cut_config, variable, binnings[variable], energy_binning, energy_estimator=energy_estimator, config=config, workdir=workdir, fill_hists=fill_hists)
        return Selection(cuts, energy_estimator, energy_binning, fill_hists=fill_hists)

    def add_to_file(self, file_dict, name):
        file_dict[f"{name}_cuts"] = list(self.cuts.keys())
        for cut_name, cut in self.cuts.items():
            cut.add_to_file(file_dict, f"{name}_cut_{cut_name}")
            self.after_all_other_passed_histograms[cut_name].add_to_file(file_dict, f"{name}_after_all_other_{cut_name}_passed")
            self.after_all_other_failed_histograms[cut_name].add_to_file(file_dict, f"{name}_after_all_other_{cut_name}_failed")
        self.total_passed_histogram.add_to_file(file_dict, f"{name}_selection_total_passed")
        self.total_failed_histogram.add_to_file(file_dict, f"{name}_selection_total_failed")
        self.total_passed_mc_histogram.add_to_file(file_dict, f"{name}_selection_total_passed_mc")
        self.total_failed_mc_histogram.add_to_file(file_dict, f"{name}_selection_total_failed_mc")
        self.energy_binning.add_to_file(file_dict, f"{name}_energy_binning")
        file_dict[f"{name}_energy_estimator"] = self.energy_estimator

    @staticmethod
    def from_file(file_dict, name):
        cut_names = list(file_dict[f"{name}_cuts"])
        energy_binning = Binning.from_file(file_dict, f"{name}_energy_binning")
        energy_estimator = file_dict[f"{name}_energy_estimator"].item()
        total_passed = WeightedHistogram.from_file(file_dict, f"{name}_selection_total_passed")
        total_failed = WeightedHistogram.from_file(file_dict, f"{name}_selection_total_failed")
        total_passed_mc = WeightedHistogram.from_file(file_dict, f"{name}_selection_total_passed_mc")
        total_failed_mc = WeightedHistogram.from_file(file_dict, f"{name}_selection_total_failed_mc")
        cuts = {cut_name: Cut.from_file(file_dict, f"{name}_cut_{cut_name}") for cut_name in cut_names}
        after_all_other_passed = {cut_name: WeightedHistogram.from_file(file_dict, f"{name}_after_all_other_{cut_name}_passed") for cut_name in cut_names}
        after_all_other_failed = {cut_name: WeightedHistogram.from_file(file_dict, f"{name}_after_all_other_{cut_name}_failed") for cut_name in cut_names}
        return Selection(cuts, energy_estimator, energy_binning,
                         total_passed_histogram=total_passed, total_failed_histogram=total_failed,
                         total_passed_mc_histogram=total_passed_mc, total_failed_mc_histogram=total_failed_mc,
                         after_all_other_passed_histograms=after_all_other_passed, after_all_other_failed_histograms=after_all_other_failed)

    def select(self, chunk, debug=False, return_pattern=False, is_mc=None):
        return chunk[self.apply(chunk, debug=debug, return_pattern=return_pattern, is_mc=is_mc)]

    def apply(self, chunk, debug=False, return_pattern=False, is_mc=None):
        selections = []
        rig = np.abs(chunk[self.energy_estimator])
        mc_momentum = None
        if is_mc:
            mc_momentum = np.abs(chunk.McMomentum)
        #weights = chunk.McWeight
        #weights = np.ones(len(chunk))
        weights = chunk.TotalWeight
        for variable, cut in self.cuts.items():
            selections.append(cut.select(chunk[variable], rig, weights, mc_momentum=mc_momentum))
        if debug:
            cut_labels = np.array([cut.label for cut in self.cuts.values()])
            for index, event in enumerate(chunk):
                if not np.all([s[index] for s in selections]):
                    label = ", ".join(f"{cut_label} ({variable}={event[variable]})" for passed_cut, cut_label, variable in zip(selections, cut_labels, self.cuts.keys()) if not passed_cut[index])
                    event = chunk[index]
                    print(f"Event {event.RunNumber} {event.EventNumber} failed selection: {label}", flush=True)
                else:
                    print(f"Event {event.RunNumber} {event.EventNumber} passed.", flush=True)
        if len(self.cuts) > 1 and self.fill_hists:
            for index, (variable, selection) in enumerate(zip(self.cuts, selections)):
                selection_without = np.all(selections[:index] + selections[index+1:], axis=0)
                passed_after = np.all((selection_without, selection), axis=0)
                failed_after = np.all((selection_without, np.invert(selection)), axis=0)
                self.after_all_other_passed_histograms[variable].fill(chunk[variable][passed_after], weights=weights[passed_after])
                self.after_all_other_failed_histograms[variable].fill(chunk[variable][failed_after], weights=weights[failed_after])

        passed = np.all(selections, axis=0)
        selection_pattern = np.sum(selections * 2**np.arange(len(selections))[:,None], axis=0)
        if not self.fill_hists:
            if return_pattern:
                return passed, selection_pattern
            return passed
        failed = np.invert(passed)
        self.total_passed_histogram.fill(rig[passed], weights=weights[passed])
        self.total_failed_histogram.fill(rig[failed], weights=weights[failed])
        if mc_momentum is not None:
            self.total_passed_mc_histogram.fill(mc_momentum[passed], weights=weights[passed])
            self.total_failed_mc_histogram.fill(mc_momentum[failed], weights=weights[failed])
        if return_pattern:
            return passed, selection_pattern
        return passed

    def add(self, other):
        assert set(self.cuts) == set(other.cuts)
        for cut_name in self.cuts:
            self.cuts[cut_name].add(other.cuts[cut_name])
            self.after_all_other_passed_histograms[cut_name].add(other.after_all_other_passed_histograms[cut_name])
            self.after_all_other_failed_histograms[cut_name].add(other.after_all_other_failed_histograms[cut_name])
        self.total_passed_histogram.add(other.total_passed_histogram)
        self.total_failed_histogram.add(other.total_failed_histogram)
        self.total_passed_mc_histogram.add(other.total_passed_mc_histogram)
        self.total_failed_mc_histogram.add(other.total_failed_mc_histogram)

    def __iadd__(self, other):
        self.add(other)
        return self

    def __add__(self, other):
        cuts = {variable: self.cuts[variable] + other.cuts[variable] for variable in self.cuts}
        assert self.energy_estimator == other.energy_estimator
        assert self.energy_binning == other.energy_binning
        after_all_other_passed = {variable: self.after_all_other_passed_histograms[variable] + other.after_all_other_passed_histograms[variable] for variable in self.cuts}
        after_all_other_failed = {variable: self.after_all_other_failed_histograms[variable] + other.after_all_other_failed_histograms[variable] for variable in self.cuts}
        return Selection(
            cuts=cuts,
            energy_estimator=self.energy_estimator,
            energy_binning=self.energy_binning,
            total_passed_histogram=self.total_passed_histogram + other.total_passed_histogram,
            total_failed_histogram=self.total_failed_histogram + other.total_failed_histogram,
            total_passed_mc_histogram=self.total_passed_mc_histogram + other.total_passed_mc_histogram,
            total_failed_mc_histogram=self.total_failed_mc_histogram + other.total_failed_mc_histogram,
            after_all_other_passed_histograms=after_all_other_passed,
            after_all_other_failed_histograms=after_all_other_failed)


    def has_mc_data(self):
        return self.total_passed_mc_histogram.values.sum() + self.total_failed_mc_histogram.values.sum() > 0

    def plot_cut(self, cut_name, plot, style="mc", **kwargs):
        cut = self.cuts[cut_name]
        plot.set_xlabel(cut_name)
        plot.set_ylabel("Events")
        plot.set_title(cut.label)
        cut.plot(plot, style=style, **kwargs)

    def cut_plot_efficiency(self, cut_name, plot, style="mc", **kwargs):
        cut = self.cuts[cut_name]
        plot.set_xlabel("E / GeV")
        plot.set_ylabel("$\\epsilon$")
        plot.set_xscale("log")
        plot.set_ylim(0, 1)
        cut.plot_efficiency_per_energy(plot, label=cut.label, style=style, **kwargs)

    def cut_plot_mc_efficiency(self, cut_name, plot, style="mc", **kwargs):
        cut = self.cuts[cut_name]
        plot.set_xlabel("MC E / GeV")
        plot.set_ylabel("$\\epsilon$")
        plot.set_xscale("log")
        plot.set_ylim(0, 1)
        cut.plot_efficiency_per_mc_momentum(plot, label=cut.label, style=style, **kwargs)

    def plot(self, resultdir="plots", prefix="selection", title=None, style="mc"):
        for cut_name in self.cuts:
            cut_figure = plt.figure(figsize=(12, 6.15))
            cut_plot = cut_figure.subplots(1, 1)
            self.plot_cut(cut_name, cut_plot, style=style)
            cut_plot.legend()
            save_figure(cut_figure, resultdir, f"{prefix}_cut_{cut_name}", close_figure=False)
            cut_plot.set_yscale("log")
            ymin, ymax = cut_plot.get_ylim()

            cut_plot.set_ylim(None, ymax)
            save_figure(cut_figure, resultdir, f"{prefix}_cut_{cut_name}_log")

            if len(self.cuts) > 1:
                after_all_other_figure = plt.figure(figsize=(12, 6.15))
                after_all_other_plot = after_all_other_figure.subplots(1, 1)
                after_all_other_plot.set_xlabel(cut_name)
                after_all_other_plot.set_ylabel("Events")
                after_all_other_plot.set_title(f"{self.cuts[cut_name].label}, after all other cuts")
                plot_histogram_1d(after_all_other_plot, self.after_all_other_passed_histograms[cut_name],
                                  style=style, color="green", label=f"passed({self.after_all_other_passed_histograms[cut_name].values.sum():.3e})")
                plot_histogram_1d(after_all_other_plot, self.after_all_other_failed_histograms[cut_name],
                                  style=style, color="red", label=f"failed ({self.after_all_other_failed_histograms[cut_name].values.sum():.3e})")
                after_all_other_plot.plot([np.nan], [np.nan], color="white", label=f"Total Events: {self.after_all_other_passed_histograms[cut_name].values.sum()+self.after_all_other_failed_histograms[cut_name].values.sum():.3e}")
                after_all_other_plot.legend()
                save_figure(after_all_other_figure, resultdir, f"{prefix}_cut_{cut_name}_after_all_other", close_figure=False)
                after_all_other_plot.set_yscale("log")
                after_all_other_plot.set_ylim(0.9, ymax)
                save_figure(after_all_other_figure, resultdir, f"{prefix}_cut_{cut_name}_after_all_other_log")

            per_energy_figure = plt.figure(figsize=(12, 6.15))
            per_energy_plot = per_energy_figure.subplots(1, 1)
            per_energy_plot.set_title(f"{self.cuts[cut_name].label} per energy")
            self.cuts[cut_name].plot_per_energy(per_energy_plot)
            save_figure(per_energy_figure, resultdir, f"{prefix}_cut_{cut_name}_per_energy")

            if self.cuts[cut_name].has_mc_data():
                per_mc_momentum_figure = plt.figure(figsize=(12, 6.15))
                per_mc_momentum_plot = per_mc_momentum_figure.subplots(1, 1)
                per_mc_momentum_plot.set_title(f"{self.cuts[cut_name].label} per MC momentum")
                self.cuts[cut_name].plot_per_mc_momentum(per_mc_momentum_plot)
                save_figure(per_mc_momentum_figure, resultdir, f"{prefix}_cut_{cut_name}_per_mc_momentum")

        efficiency_figure = plt.figure(figsize=(16, 8.2))
        efficiency_plot = efficiency_figure.subplots(1, 1)
        efficiency_plot.set_title(title)
        for cut_name in self.cuts:
            self.cut_plot_efficiency(cut_name, efficiency_plot, style=style)
        total_passed = self.total_passed_histogram.values
        total_failed = self.total_failed_histogram.values
        total_all = total_passed + total_failed
        total_efficiency = calculate_efficiency(total_passed, total_all)
        total_efficiency_error = calculate_efficiency_error(total_passed, total_all)
        energy_bin_centers = self.energy_binning.bin_centers
        facecolor = "none" if style == "mc" else None
        efficiency_plot.errorbar(energy_bin_centers, total_efficiency, total_efficiency_error, label="Total", marker="o", mfc=facecolor)
        efficiency_plot.set_ylim(0, 1)
        set_energy_ticks(efficiency_plot)
        efficiency_plot.legend()
        save_figure(efficiency_figure, resultdir, f"{prefix}_efficiency")

        efficiency_mc_figure = plt.figure(figsize=(16, 8.2))
        efficiency_mc_plot = efficiency_mc_figure.subplots(1, 1)
        efficiency_mc_plot.set_title(title)
        for cut_name in self.cuts:
            self.cut_plot_mc_efficiency(cut_name, efficiency_mc_plot, style=style)
        total_mc_passed = self.total_passed_mc_histogram.values
        total_mc_failed = self.total_failed_mc_histogram.values
        total_mc_all = total_mc_passed + total_mc_failed
        total_mc_efficiency = calculate_efficiency(total_mc_passed, total_mc_all)
        total_mc_efficiency_mc_error = calculate_efficiency_error(total_mc_passed, total_mc_all)
        energy_bin_centers = self.energy_binning.bin_centers
        facecolor = "none" if style == "mc" else None
        efficiency_mc_plot.errorbar(energy_bin_centers, total_mc_efficiency, total_mc_efficiency_mc_error, label="Total", marker="o", mfc=facecolor)
        efficiency_mc_plot.legend()
        efficiency_mc_plot.set_ylim(0, 1)
        set_energy_ticks(efficiency_mc_plot)
        save_figure(efficiency_mc_figure, resultdir, f"{prefix}_efficiency_mc")
