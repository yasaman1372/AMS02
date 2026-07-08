#!/usr/bin/env python3

import tracemalloc

import glob
import multiprocessing as mp
import os
import time

import numpy as np
#np.warnings.filterwarnings("error", category=np.VisibleDeprecationWarning)
np.seterr(divide="ignore")
import matplotlib.pyplot as plt
import awkward as ak
import uproot

from tools.binnings import Binnings
from tools.config import get_config
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d
from tools.roottree import read_tree, clear_cache
from tools.sample import Sample
from tools.utilities import save_figure


class EventCache:
    def __init__(self, resultdir, prefix, branches, events_per_file=100000):
        self.resultdir = resultdir
        self.prefix = prefix
        self.branches = branches
        self.write_counter = 0
        self.events_per_file = events_per_file
        self.reset_cache()

    def reset_cache(self):
        self.events = {branch: [] for branch in self.branches}
        self.cached_events = 0
        self.cache_entries = 0

    def append(self, events):
        if len(events) > 0:
            for branch in self.branches:
                self.events[branch].append(ak.to_numpy(events[branch]))
            self.cached_events += len(events)
            self.cache_entries += 1
            if self.cached_events >= self.events_per_file or self.cache_entries >= self.events_per_file / 100:
                self.write()

    def write(self):
        if self.write_counter == 0:
            # make sure there are no files from previous runs left
            old_files = [filename for filename in os.listdir(self.resultdir) if filename.startswith(self.prefix)]
            for filename in old_files:
                os.remove(os.path.join(self.resultdir, filename))
        if self.cached_events > 0:
            data = {branch: np.concatenate(self.events[branch]) for branch in self.branches}
            np.savez(os.path.join(self.resultdir, f"{self.prefix}_{self.write_counter}.npz"), **data)
            self.write_counter += 1
            self.reset_cache()


def _make_var_histogram(variable, binnings):
    if ":" in variable:
        variable, binning_name = variable.split(":", 1)
        return WeightedHistogram(binnings[binning_name], labels=(variable,))
    return WeightedHistogram(binnings[variable], labels=(variable,))

def _make_var_per_energy_histogram(variable, binnings, energy_estimator):
    if ":" in variable:
        variable, binning_name = variable.split(":", 1)
        return WeightedHistogram(binnings["energy"], binnings[binning_name], labels=(energy_estimator, variable))
    return WeightedHistogram(binnings["energy"], binnings[variable], labels=(energy_estimator, variable))

def _make_pair_histogram(pair, binnings):
    var_binnings = []
    variables = []
    for variable in pair:
        if ":" in variable:
            var, binning_name = variable.split(":", 1)
            variables.append(var)
            var_binnings.append(binnings[binning_name])
        else:
            variables.append(variable)
            var_binnings.append(binnings[variable])
    return WeightedHistogram(*var_binnings, labels=tuple(variables))


class ResultHists:
    def __init__(self, variables, pairs, hists, hists_per_energy, pair_hists, energy_estimator, resultdir, prefix, label):
        self.variables = variables
        self.pairs = pairs
        self.energy_estimator = energy_estimator
        self.resultdir = resultdir
        self.prefix = prefix
        self.label = label
        self.hists = hists
        self.hists_per_energy = hists_per_energy
        self.pair_hists = pair_hists


    @staticmethod
    def create(variables, pairs, binnings, energy_estimator, resultdir, prefix, label):
        hists = {
            variable.split(":")[0]: _make_var_histogram(variable, binnings)
            for variable in variables
        }
        hists_per_energy = {
            variable.split(":")[0]: _make_var_per_energy_histogram(variable, binnings, energy_estimator)
            for variable in variables
        }
        pair_hists = {
            tuple((s.split(":")[0] for s in pair)): _make_pair_histogram(pair, binnings)
            for pair in pairs
        }
        variables = [variable.split(":")[0] for variable in variables]
        pairs = [(pair[0].split(":")[0], pair[1].split(":")[0]) for pair in pairs]
        return ResultHists(variables, pairs, hists, hists_per_energy, pair_hists, energy_estimator, resultdir, prefix, label)

    def fill(self, events):
        weights = events.TotalWeight
        rigidity = np.abs(events[self.energy_estimator])
        for variable in self.variables:
            self.hists[variable].fill(events[variable], weights=weights)
            self.hists_per_energy[variable].fill(rigidity, events[variable], weights=weights)
        for pair in self.pairs:
            self.pair_hists[pair].fill(events[pair[0]], events[pair[1]], weights=weights)


    def save(self, save_as_root=False):
        results = {"variables": self.variables, "pairs": self.pairs, "label": self.label, "prefix": self.prefix, "energy_estimator": self.energy_estimator}
        for variable in self.variables:
            self.hists[variable].add_to_file(results, f"hist_{variable}")
            self.hists_per_energy[variable].add_to_file(results, f"hist_per_energy_{variable}")
        for pair in self.pairs:
            self.pair_hists[pair].add_to_file(results, f"pair_hist_{pair[0]}_{pair[1]}")
        np.savez(os.path.join(self.resultdir, f"{self.prefix}.npz"), **results)
        if save_as_root:
            with uproot.recreate(os.path.join(self.resultdir, f"{self.prefix}.root")) as root_file:
                for variable in self.variables:
                    root_file[f"hist_{variable}"] = self.hists[variable].to_uhi_proxy()
                    root_file[f"hist_{variable}_per_energy"] = self.hists_per_energy[variable].to_uhi_proxy()
                for pair in self.pairs:
                    root_file[f"pair_hist_{pair[0]}_{pair[1]}"] = self.pair_hists[pair].to_uhi_proxy()

    @staticmethod
    def load_file(filename, resultdir, prefix):
        with np.load(filename) as file:
            variables = file["variables"]
            pairs = [(pair[0], pair[1]) for pair in file["pairs"]]
            label = file["label"]
            energy_estimator = file["energy_estimator"]
            hists = {variable: WeightedHistogram.from_file(file, f"hist_{variable}") for variable in variables}
            hists_per_energy = {variable: WeightedHistogram.from_file(file, f"hist_per_energy_{variable}") for variable in variables}
            pair_hists = {(pair[0], pair[1]): WeightedHistogram.from_file(file, f"pair_hist_{pair[0]}_{pair[1]}") for pair in pairs}
            return ResultHists(variables, pairs, hists, hists_per_energy, pair_hists, energy_estimator, resultdir, prefix, label)

    @staticmethod
    def load(resultdir, prefix):
        filename = os.path.join(resultdir, f"{prefix}.npz")
        if os.path.exists(filename):
            return load_file(filename)
        else:
            filenames = list(glob.glob(os.path.join(resultdir, f"{prefix}.npz")))
            if not filenames:
                filenames = glob.glob(os.path.join(resultdir, f"{prefix}_*.npz"))
            return ResultHists.sum((ResultHists.load_file(filename, resultdir, prefix) for filename in filenames))

    @staticmethod
    def load_files(filenames):
        if len(filenames) == 1:
            filename = filenames[0]
            assert filename.endswith("_0_Hists.npz")
            prefix = filename[:-len("_0_Hists.npz")]
        else:
            prefix = os.path.commonprefix(filenames)
        if prefix.endswith("_"):
            prefix = prefix[:-1]
        resultdir = os.path.dirname(prefix)
        filename_prefix = os.path.basename(prefix)
        return ResultHists.sum((ResultHists.load_file(filename, resultdir, filename_prefix) for filename in filenames))

    def __add__(self, other):
        assert np.all(self.variables == other.variables)
        assert np.all(self.pairs == other.pairs)
        assert self.energy_estimator == other.energy_estimator
        hists = {
            variable: self.hists[variable] + other.hists[variable]
            for variable in self.variables
        }
        pair_hists = {
            pair: self.pair_hists[pair] + other.pair_hists[pair]
            for pair in self.pairs
        }
        hists_per_energy = {
            variable: self.hists_per_energy[variable] + other.hists_per_energy[variable]
            for variable in self.variables
        }
        return ResultHists(variables=self.variables, pairs=self.pairs, hists=hists, hists_per_energy=hists_per_energy, pair_hists=pair_hists, energy_estimator=self.energy_estimator, resultdir=self.resultdir, prefix=self.prefix, label=self.label)

    @staticmethod
    def sum(result_hists):
        hists = list(result_hists)
        first = hists[0]
        for hist in hists[1:]:
            first = first + hist
        return first


    def merge_files(self, prefix):
        temp_filename = os.path.join(self.resultdir, f"{prefix}.npz")
        with np.load(temp_filename) as file:
            for variable in self.variables:
                self.hists[variable].add(WeightedHistogram.from_file(file, f"hist_{variable}"))
                self.hists_per_energy[variable].add(WeightedHistogram.from_file(file, f"hist_per_energy_{variable}"))
            for pair in self.pairs:
                self.pair_hists[pair].add(WeightedHistogram.from_file(file, f"pair_hist_{pair[0]}_{pair[1]}"))
        os.remove(temp_filename)


    def plot(self, plotdir):
        for variable, hist in self.hists.items():
            figure = plt.figure(figsize=(12, 6.15))
            figure.suptitle(f"{self.label}")
            plot = figure.subplots(1, 1)
            plot_histogram_1d(plot, hist, show_overflow=True)
            save_figure(figure, plotdir, f"{self.prefix}_{variable}_lin")
        for variable, hist in self.hists.items():
            figure = plt.figure(figsize=(12, 6.15))
            figure.suptitle(f"{self.label}")
            plot = figure.subplots(1, 1)
            plot_histogram_1d(plot, hist, log=True, show_overflow=True)
            save_figure(figure, plotdir, f"{self.prefix}_{variable}_log")
        for variable, hist in self.hists_per_energy.items():
            figure = plt.figure(figsize=(12, 6.15))
            figure.suptitle(f"{self.label}")
            plot = figure.subplots(1, 1)
            plot_histogram_2d(plot, hist)
            save_figure(figure, plotdir, f"{self.prefix}_{variable}_per_energy")
        for pair, hist in self.pair_hists.items():
            figure = plt.figure(figsize=(12, 6.15))
            figure.suptitle(f"{self.label}")
            plot = figure.subplots(1, 1)
            plot_histogram_2d(plot, hist)
            save_figure(figure, plotdir, f"{self.prefix}_{'_vs_'.join(pair)}")


class Results:
    def __init__(self, variables, pairs, binnings, energy_estimator, resultdir, prefix, label, write_events=False, events_per_file=100000):
        self.branches = sorted(set([variable.split(":")[0] for variable in variables] + [energy_estimator, "TotalWeight"]))
        self.hists = ResultHists.create(variables, pairs, binnings, energy_estimator, resultdir, f"{prefix}_Hists", label)
        self.write_events = write_events
        if write_events:
            self.event_cache = EventCache(resultdir, f"{prefix}_Events", self.branches, events_per_file=events_per_file)

    def fill(self, events):
        self.hists.fill(events)
        if self.write_events:
            self.event_cache.append(events)

    def save(self, save_as_root=False):
        filename = self.hists.save(save_as_root=save_as_root)
        if self.write_events:
            self.event_cache.write()

    def merge_files(self, prefix):
        self.hists.merge_files(f"{prefix}_Hists")

    def plot(self, plotdir):
        self.hists.plot(plotdir)


class PerBinResults:
    def __init__(self, variables, pairs, binnings, energy_estimator, resultdir, prefix, label, rigidity_binning, write_events=False, events_per_file=100000):
        self.rigidity_binning = rigidity_binning
        self.energy_estimator = energy_estimator
        self.nbins = len(rigidity_binning.edges) - 1
        self.results = {bin: Results(variables, pairs, binnings, energy_estimator, resultdir, f"{prefix}_r{bin}", label, write_events=write_events, events_per_file=events_per_file) for bin in range(1, self.nbins - 1)}
        self.results_all = Results(variables, pairs, binnings, energy_estimator, resultdir, f"{prefix}_all", label)

    def fill(self, events):
        rigidity = np.abs(events[self.energy_estimator])
        rig_bins = self.rigidity_binning.get_indices(rigidity)
        for bin, results in self.results.items():
            results.fill(events[rig_bins == bin])
        self.results_all.fill(events)

    def save(self, save_as_root=False):
        for results in self.results.values():
            results.save(save_as_root=save_as_root)
        self.results_all.save(save_as_root=save_as_root)

    def merge_files(self, prefix):
        for bin, results in self.results.items():
            results.merge_files(f"{prefix}_r{bin}")
        self.results_all.merge_files(f"{prefix}_all")

    def plot(self, plotdir):
        for results in self.results.values():
            results.plot(plotdir)
        self.results_all.plot(plotdir)


def handle_file(arg):
    filename, treename, chunk_size, rank, nranks, kwargs = arg

    tracemalloc.start()

    variables = kwargs["variables"]
    pairs = kwargs["pairs"]
    config = kwargs["config"]
    sample_name = kwargs["sample"]
    corrections = kwargs["corrections"]
    workdir = kwargs["workdir"]
    resultdir = kwargs["resultdir"]
    outputprefix = kwargs["outputprefix"]
    label = kwargs["label"]
    do_per_energy = kwargs["do_per_energy"]
    verbose = kwargs["verbose"]
    energy_range = kwargs["energy_range"]
    time_range = kwargs["time_range"]
    cutoff = kwargs["cutoff"]
    mc_weighting_file = kwargs["mc_weighting_file"]
    write_events = kwargs["write_events"]
    events_per_file = kwargs["events_per_file"]
    events_binning_name = kwargs["events_binning"]
    debug = kwargs["debug"]
    debug_selection = kwargs["debug_selection"]
    with_selection_hists = kwargs["with_selection_hists"]


    sample = Sample.load(config, sample_name, workdir, fill_selection_hists=with_selection_hists)
    binnings = sample.binnings
    events_binning = binnings[events_binning_name] if events_binning_name is not None else None

    if corrections is not None:
        sample.set_corrections(corrections, config, workdir)
    energy_estimator = sample.estimators["Energy"]
    if energy_range is not None:
        sample.set_energy_range(*energy_range)
    if time_range is not None:
        sample.set_time_range(*time_range)
    if cutoff is not None:
        cutoff_type, cutoff_angle, cutoff_factor_str = cutoff
        cutoff_factor = float(cutoff_factor_str)
        sample.set_cutoff(cutoff_type, cutoff_angle, cutoff_factor)
    if mc_weighting_file is not None:
        sample.set_mc_weighting(mc_weighting_file)

    branches = list(set([variable.split(":")[0] for variable in variables] + [variable.split(":")[0] for pair in pairs for variable in pair] + [energy_estimator, "TotalWeight"]))
    if debug:
        branches.extend(["RunNumber", "EventNumber"])
    if events_binning is None:
        results = Results(variables, pairs, binnings, energy_estimator, resultdir, f"{outputprefix}_{rank}", label, write_events=write_events, events_per_file=events_per_file)
    else:
        results = PerBinResults(variables, pairs, binnings, energy_estimator, resultdir, f"{outputprefix}_{rank}", label, events_binning, write_events=write_events, events_per_file=events_per_file)

    for events in sample.read_tree(filename, treename, branches=branches, rank=rank, nranks=nranks, chunk_size=chunk_size, verbose=verbose, resultdir=resultdir, prefix=outputprefix, debug=debug_selection):
        if debug:
            for variable in variables:
                print(variable, events[variable])
        results.fill(events)

        snapshot = tracemalloc.take_snapshot()
        stats = snapshot.statistics("lineno", cumulative=True)
        total_mem = sum([s.size for s in stats])
        #for stat in stats[:5]:
        #    print(f"{stat.size/2**20:>5.1f} MiB: {stat.traceback}")
        print(f"Rank {rank},{total_mem:.0f},{time.time()}", flush=True)

    result_filename = results.save()
    tracemalloc.stop()
    return rank


def make_args(filename, treename, chunk_size, nranks, parallel, **kwargs):
    parallel_index, parallel_total = parallel
    assert parallel_index >= 0 and parallel_index < parallel_total
    for rank in range(parallel_index * nranks, (parallel_index + 1) * nranks):
        yield (filename, treename, chunk_size, rank, (nranks * parallel_total), kwargs)


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--tree", nargs="+", required=True, help="Path to tree file(s).")
    parser.add_argument("--treename", help="Tree name in file.")
    parser.add_argument("--config", required=True, nargs=2, help="Path to analysis config file and workdir.")
    parser.add_argument("--sample", required=True, help="Sample selection from config to apply.")
    parser.add_argument("--corrections", help="MC corrections to apply.")
    parser.add_argument("--outputprefix", default="Histograms", help="Prefix for the output file.")
    parser.add_argument("--label", help="Dataset label for plots.")
    parser.add_argument("--resultdir", default="results", help="Directory to store temporary files and results in.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of parallel processes.")
    parser.add_argument("--parallel", type=int, nargs=2, default=(0, 1), help="Index of this job and total number of parallel processes if multiple jobs are running in parallel.")
    parser.add_argument("--chunk-size", type=int, default=200000, help="Number of events per chunk.")
    parser.add_argument("--variables", nargs="+", help="Variables to make histograms of.")
    parser.add_argument("--pairs", nargs="+", help="Variable pairs to make 2D-histograms of.")
    parser.add_argument("--observed-selections", nargs="+", help="Selections to check but not apply.")
    parser.add_argument("--energy-range", nargs=2, type=float, help="Rigidity range [start, end).")
    parser.add_argument("--time-range", nargs=2, type=int, help="ISS Time range [start, end].")
    parser.add_argument("--cutoff", nargs=3, help="ISS Geomagnetic cutoff (type, angle, safety factor).")
    parser.add_argument("--mc-weighting-file", help="Flux file to calculate MC weights from.")
    parser.add_argument("--per-energy", action="store_true", help="Also create variable-per-energy 2d histograms.")
    parser.add_argument("--per-energy-binning", help="Name of rigidity binning.")
    parser.add_argument("--write-events", action="store_true", help="Write events to file.")
    parser.add_argument("--events-per-file", type=int, default=100000, help="Number of events to cache before writing to a file.")
    parser.add_argument("--plot-style", default="mc", choices=["mc", "iss"], help="Plot style (\"mc\" or \"iss\").")
    parser.add_argument("--save-as-root", action="store_true", help="Save results in ROOT files in addition to NPZ files.")
    parser.add_argument("--quiet", "-q", dest="verbose", action="store_false", help="Print progress.")
    parser.add_argument("--skip-plots", action="store_false", dest="do_plots", help="Skip creating plots of histograms and selections.")
    parser.add_argument("--debug", action="store_true", help="Print single event information.")
    parser.add_argument("--debug-selection", action="store_true", help="Print single event selection information.")
    parser.add_argument("--with-selection-hists", action="store_true", help="Create data about selection efficiency.")

    args = parser.parse_args()

    config_file, workdir = args.config
    config = get_config(config_file)
    sample = Sample.load(config, args.sample, workdir, fill_selection_hists=args.with_selection_hists)
    if args.corrections is not None:
        sample.set_corrections(args.corrections, config, workdir)
    binnings = sample.binnings

    treename = args.treename
    if not treename:
        treename = config["analysis"].get("treename", "PhotonTree")

    pairs = []
    if args.pairs is not None:
        assert len(args.pairs) % 2 == 0
        pairs = list(zip(args.pairs[0::2], args.pairs[1::2]))

    parallel = args.parallel
    parallel_index, parallel_total = parallel
    prefix = f"{args.outputprefix}_{parallel_index}"

    dataset_label = args.label or prefix
    label = f"{dataset_label} {sample.label}"

    variables = sorted(set(args.variables or ()))

    if args.per_energy_binning is None:
        results = Results(variables, pairs, binnings, sample.estimators["Energy"], args.resultdir, prefix, label)
    else:
        binning = binnings[args.per_energy_binning]
        results = PerBinResults(variables, pairs, binnings, sample.estimators["Energy"], args.resultdir, prefix, label, binning)

    os.makedirs(args.resultdir, exist_ok=True)
    os.makedirs(args.plotdir, exist_ok=True)

    clear_cache(treename)

    with mp.get_context("fork" if args.verbose else "spawn").Pool(args.nprocesses) as pool:
        pool_args = make_args(args.tree, treename, args.chunk_size, args.nprocesses, parallel,
                              variables=variables, pairs=pairs,
                              config=config, sample=args.sample, corrections=args.corrections,
                              resultdir=args.resultdir, outputprefix=prefix, label=label,
                              workdir=workdir, energy_range=args.energy_range,
                              time_range=args.time_range, cutoff=args.cutoff, mc_weighting_file=args.mc_weighting_file,
                              events_binning=args.per_energy_binning,
                              do_per_energy=args.per_energy, write_events=args.write_events,
                              events_per_file=args.events_per_file, verbose=args.verbose,
                              debug=args.debug, debug_selection=args.debug_selection,
                              with_selection_hists=args.with_selection_hists)
        for rank in pool.imap_unordered(handle_file, pool_args):
            results.merge_files(f"{prefix}_{rank}")

    clear_cache(treename)

    results.save(save_as_root=args.save_as_root)
    if args.do_plots:
        results.plot(plotdir=args.plotdir)

    if args.with_selection_hists:
        sample.merge_selections(nranks=args.nprocesses, first_rank=parallel_index * args.nprocesses, prefix=prefix)
        if args.do_plots:
            sample.plot_selections(resultdir=args.plotdir, outputprefix=prefix, style=args.plot_style)
        sample.save_selections(args.resultdir, prefix)


if __name__ == "__main__":
    main()
