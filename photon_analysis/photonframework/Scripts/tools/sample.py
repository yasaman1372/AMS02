
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from scipy.interpolate import interp1d, UnivariateSpline

from .binnings import Binning, Binnings
from .constants import ACCEPTANCE_CATEGORIES, MC_PARTICLE_IDS
from .corrections import Corrections
from .histograms import plot_histogram_1d
from .roottree import read_tree, count_total_events
from .selection import Selection, cut_pattern
from .statistics import fermi_function, inverse_fermi_function, scaled_fermi_function
from .variables import DerivedVariables
from .utilities import fit_flux, load_flux, load_mc_trigger_density, power_law, resolve_derived_branches, save_figure, set_energy_ticks, merge_run_and_event_number


class Sample:
    def __init__(self, sample_name, selections, estimators, mc_estimators, corrections=None, derived_variables=None, energy_range=None, time_range=None, mc_weighting=None, label=None, tag_cuts_name=None, binnings=None):
        self.sample_name = sample_name
        self.selections = selections
        self.estimators = estimators
        self.mc_estimators = mc_estimators
        self.corrections = corrections
        self.derived_variables = derived_variables
        self.energy_range = energy_range
        self.time_range = time_range
        self.mc_weighting = mc_weighting
        self.label = label or sample_name
        self.tag_cuts_name = tag_cuts_name
        self.binnings = binnings

    @staticmethod
    def load(config, sample_name, workdir, fill_selection_hists=True):
        binnings = Binnings.from_config(config)
        sample_config = config["samples"][sample_name]
        estimators = config["analysis"].get("estimators", {})
        for name, value in sample_config.get("estimators", {}).items():
            estimators[name] = value
        mc_estimators = config["analysis"].get("mc_estimators", {})
        for name, value in sample_config.get("mc_estimators", {}).items():
            mc_estimators[name] = value
        derived_variables = DerivedVariables(config=config, workdir=workdir, energy_estimator=estimators["Energy"], binnings=binnings)
        selections = {
            selection_name: Selection.load(config["selections"][selection_name], estimators["Energy"], binnings, config=config, workdir=workdir, fill_hists=fill_selection_hists)
            for selection_name in sample_config["selections"]
        }
        label = sample_config.get("label", None)
        tag_cuts = sample_config.get("tag_cuts", None)
        return Sample(sample_name, selections, estimators, mc_estimators, derived_variables=derived_variables, binnings=binnings, label=label, tag_cuts_name=tag_cuts)

    def set_energy_range(self, energy_min, energy_max):
        self.energy_range = (energy_min, energy_max)

    def set_time_range(self, time_min, time_max):
        self.time_range = (time_min, time_max)

    def set_corrections(self, dataset_correction, config, workdir):
        if dataset_correction in config["datasets"]:
            dataset_correction = config["datasets"][dataset_correction].get("corrections", None)
        if dataset_correction is None:
            return
        sample_config = config["samples"][self.sample_name]
        if "corrections" not in sample_config:
            return
        sample_correction = sample_config["corrections"]
        corrections_path = os.path.join(workdir, "corrections", "corrections", "results", f"corrections_{dataset_correction}_{sample_correction}.json")
        self.corrections = Corrections.load(corrections_path)


    def set_mc_weighting(self, filename, scale_factor=None):
        self.mc_weighting = load_mc_weighting(filename, scale_factor=scale_factor)

    def get_branches(self, branches, apply_selections=True):
        read_all_branches = branches is None
        branches = (list(branches) or []) + ["McParticleID", "McMomentum", "McWeight", "TotalWeight"]
        if self.time_range is not None:
            branches.append("Time")
        branches.append(self.estimators["Energy"])
        if apply_selections:
            selection_branches = [var for sel in self.selections.values() for var in sel.cuts]
        else:
            selection_branches = []
        all_branches = list(set(branches) | set(selection_branches))
        primary_branches, derived_branches = self.derived_variables.resolve_branches(all_branches)
        derived_weight_branches = set()
        branches_to_read = (primary_branches - derived_weight_branches) if not read_all_branches else None
        return branches_to_read, derived_branches, derived_weight_branches

    def read_tree(self, filename, treename, branches, rank, nranks, chunk_size=1000000, verbose=True, prefix="selections", resultdir="results", debug=False, apply_selections=True, pass_all=False, apply_eventlist=None, return_selection_pattern=False):
        branches_to_read, derived_branches, _ = self.get_branches(branches, apply_selections=apply_selections)
        if apply_eventlist is not None:
            runevent_ref = merge_run_and_event_number(apply_eventlist[:,0], apply_eventlist[:,1])

        for chunk in read_tree(filename, treename, branches=branches_to_read, rank=rank, nranks=nranks, chunk_size=chunk_size, verbose=verbose):
            if return_selection_pattern:
                run_number = ak.to_numpy(chunk.RunNumber)
                event_number = ak.to_numpy(chunk.EventNumber)
            if apply_eventlist is not None:
                runevent = merge_run_and_event_number(ak.to_numpy(chunk.RunNumber), ak.to_numpy(chunk.EventNumber))
                eventlist_selection = np.isin(runevent, runevent_ref, assume_unique=True)
                chunk = chunk[eventlist_selection]
            chunk = self.apply(chunk, derived_branches=derived_branches, apply_selections=apply_selections, pass_all=pass_all, debug=debug, return_pattern=return_selection_pattern)
            if return_selection_pattern:
                chunk, selection_pattern = chunk
                yield run_number, event_number, selection_pattern
            else:
                if len(chunk) > 0:
                    yield chunk

        if apply_selections:
            if any(selection.fill_hists for selection in self.selections.values()):
                self.save_selections(resultdir, f"{prefix}_{rank}")

    def get_max_cuts_per_selection(self):
        return max(len(selection.cuts) for selection in self.selections.values())

    def get_selection_pattern_dtype(self):
        max_cuts_per_selection = self.get_max_cuts_per_selection()
        if max_cuts_per_selection < 8:
            return np.uint8
            default_pattern = 0xff
        elif max_cuts_per_selection < 16:
            return np.uint16
            default_pattern = 0xffff
        elif max_cuts_per_selection < 32:
            return np.uint32
            default_pattern = 0xffffffff
        elif max_cuts_per_selection < 64:
            return np.uint64
            default_pattern = 0xffffffffffffffff
        else:
            raise ValueError(f"Cannot save cut pattern with {max_cuts_per_selection} in one int.")

    def get_selection_pattern_default_value(self):
        max_cuts_per_selection = self.get_max_cuts_per_selection()
        if max_cuts_per_selection < 8:
            return 0xff
        elif max_cuts_per_selection < 16:
            return 0xffff
        elif max_cuts_per_selection < 32:
            return 0xffffffff
        elif max_cuts_per_selection < 64:
            return 0xffffffffffffffff
        else:
            raise ValueError(f"Cannot save cut pattern with {max_cuts_per_selection} in one int.")

    def apply(self, chunk, derived_branches, apply_selections=True, pass_all=False, debug=False, return_pattern=False, is_mc=None):
        if self.energy_range is not None:
            energy = np.abs(chunk[self.estimators["energy"]])
            min_energy, max_energy = self.energy_range
            chunk = chunk[(energy >= min_energy) & (energy < max_energy)]
        if is_mc is None:
            is_mc = ak.any(chunk.McParticleID != 0)
        if not is_mc:
            if self.time_range is not None:
                time_min, time_max = self.time_range
                time = chunk.Time
                chunk = chunk[(time >= time_min) & (time <= time_max)]
        if len(chunk) == 0:
            if return_pattern:
                return chunk, np.zeros(0, dtype=bool)
            return chunk
        #default_weight = np.ones(len(chunk))
        #chunk = chunk.with_field("TotalWeight", default_weight)
        #if self.corrections is not None:
        #    chunk = self.corrections.apply(chunk)
        for branch in derived_branches:
            chunk.add_field(branch, self.derived_variables.functions[branch])
        #if is_mc and self.mc_weighting is not None:
        #    chunk.with_field("McToIssWeight", self.mc_weighting.get_weights(chunk.McParticleID, np.abs(chunk.McMomentum)))
        #else:
        #    chunk.with_field("McToIssWeight", np.ones(len(chunk)))
        #chunk = ak.with_field(chunk, default_weight * chunk.McToIssWeight, "TotalWeight")

        if return_pattern:
            pattern_dtype = self.get_selection_pattern_dtype()
            default_pattern = self.get_selection_pattern_default_value()

            selection_patterns = np.full((len(chunk), len(self.selections)), default_pattern, dtype=pattern_dtype)
            passed_pattern = np.ones(len(chunk), dtype=bool)

        if apply_selections:
            for selection_index, (selection_name, selection) in enumerate(self.selections.items()):
                passed, selection_pattern = selection.apply(chunk, debug=debug, return_pattern=True, is_mc=is_mc)
                chunk_after_selection = chunk[passed]
                if return_pattern:
                    selection_patterns[passed_pattern, selection_index] = selection_pattern
                    passed_pattern[passed_pattern] = passed
                if not pass_all:
                    chunk = chunk_after_selection
                if len(chunk) == 0:
                    break
        if return_pattern:
            return chunk, selection_patterns
        return chunk

    def save_selections(self, resultdir, prefix):
        result_dict = {"selections": list(self.selections.keys())}
        for selection_name, selection in self.selections.items():
            if selection.fill_hists:
                selection.add_to_file(result_dict, f"selection_{selection_name}")
        os.makedirs(resultdir, exist_ok=True)
        np.savez(os.path.join(resultdir, f"{prefix}_selections.npz"), **result_dict)

    def merge_selections(self, nranks, first_rank=0, prefix="selections", resultdir="results"):
        for rank in range(first_rank, first_rank + nranks):
            temp_filename = os.path.join(resultdir, f"{prefix}_{rank}_selections.npz")
            with np.load(temp_filename) as temp_file:
                for selection_name, selection in self.selections.items():
                    selection.add(Selection.from_file(temp_file, f"selection_{selection_name}"))
            os.remove(temp_filename)

    def load_selections(self, filenames):
        for filename in filenames:
            with np.load(filename) as np_file:
                for selection_name, selection in self.selections.items():
                    selection.add(Selection.from_file(np_file, f"selection_{selection_name}"))

    def plot_selections(self, resultdir="plots", outputprefix="sample", style="mc", energy_estimator="E / GeV"):
        for selection_name, selection in self.selections.items():
            selection.plot(resultdir=resultdir, prefix=f"{outputprefix}_selection_{selection_name}", title=f"Selection {selection_name!r}")

        stack_figure = plt.figure(figsize=(12, 6.15))
        stack_plot = stack_figure.subplots(1, 1)
        stack_plot.set_title(self.label)
        stack_plot.set_xlabel(f"{energy_estimator} / GeV")
        stack_plot.set_ylabel("Events")
        first_selection = self.selections[list(self.selections.keys())[0]]
        total_events = first_selection.total_passed_histogram + first_selection.total_failed_histogram
        plot_histogram_1d(stack_plot, total_events, style=style, label="Before Selection", log=True)
        for selection_name, selection in self.selections.items():
            plot_histogram_1d(stack_plot, selection.total_passed_histogram, style=style, label=selection_name, log=True)
        stack_plot.legend()
        set_energy_ticks(stack_plot)
        save_figure(stack_figure, resultdir, f"{outputprefix}_stacked")

        if first_selection.has_mc_data():
            stack_mc_figure = plt.figure(figsize=(12, 6.15))
            stack_mc_plot = stack_mc_figure.subplots(1, 1)
            stack_mc_plot.set_title(self.label)
            stack_mc_plot.set_xlabel(f"MC Momentum / GeV")
            stack_mc_plot.set_ylabel("Events")
            total_events = first_selection.total_passed_mc_histogram + first_selection.total_failed_mc_histogram
            plot_histogram_1d(stack_mc_plot, total_events, style=style, label="Before Selection", log=True)
            for selection_name, selection in self.selections.items():
                plot_histogram_1d(stack_mc_plot, selection.total_passed_mc_histogram, style=style, label=selection_name, log=True)
            stack_mc_plot.legend()
            set_energy_ticks(stack_mc_plot)
            save_figure(stack_mc_figure, resultdir, f"{outputprefix}_stacked_mc")
