#!/usr/bin/env python3

import json
import os
import multiprocessing as mp
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

from tools.binnings import Binnings
from tools.config import get_config
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d
from tools.roottree import read_tree
from tools.sample import Sample
from tools.selection import Selection


def _tuple_or_str(value):
    if isinstance(value, str):
        return value
    return tuple(value)

class ResultHists:
    def __init__(self, triggered_and_passed_hists, triggered_and_failed_hists, tagged_and_passed_hists, tagged_and_failed_hists, triggered_and_passed_hists_2d, triggered_and_failed_hists_2d, tagged_and_passed_hists_2d, tagged_and_failed_hists_2d):
        self.triggered_and_passed_hists = triggered_and_passed_hists
        self.triggered_and_failed_hists = triggered_and_failed_hists
        self.tagged_and_passed_hists = tagged_and_passed_hists
        self.tagged_and_failed_hists = tagged_and_failed_hists
        self.triggered_and_passed_hists_2d = triggered_and_passed_hists_2d
        self.triggered_and_failed_hists_2d = triggered_and_failed_hists_2d
        self.tagged_and_passed_hists_2d = tagged_and_passed_hists_2d
        self.tagged_and_failed_hists_2d = tagged_and_failed_hists_2d

    @staticmethod
    def create(energy_estimators, energy_binning, cut_keys, taggers, binnings):
        hists = dict(triggered=dict(passed={}, failed={}), tagged=dict(passed={}, failed={}))
        hists_2d = dict(triggered=dict(passed={}, failed={}), tagged=dict(passed={}, failed={}))
        for presel in ("triggered", "tagged"):
            for success in ("passed", "failed"):
                for energy_estimator in energy_estimators:
                    hists[presel][success][energy_estimator] = {}
                    hists_2d[presel][success][energy_estimator] = {}
                    for cut_key in cut_keys:
                        selection_name, cut_variable = cut_key
                        hists[presel][success][energy_estimator][cut_key] = WeightedHistogram(energy_binning, labels=(energy_estimator,))
                        hists_2d[presel][success][energy_estimator][cut_key] = WeightedHistogram(energy_binning, binnings[cut_variable], labels=(energy_estimator, cut_variable))
                    for tagger in taggers:
                        hists[presel][success][energy_estimator][tagger] = WeightedHistogram(energy_binning, labels=(energy_estimator,))
        return ResultHists(
            hists["triggered"]["passed"], hists["triggered"]["failed"],
            hists["tagged"]["passed"], hists["tagged"]["failed"],
            hists_2d["triggered"]["passed"], hists_2d["triggered"]["failed"],
            hists_2d["tagged"]["passed"], hists_2d["tagged"]["failed"])

    def fill_triggered(self, key, events, passed):
        selection_name, cut_variable = key
        passed_events = events[passed]
        failed_events = events[np.invert(passed)]
        for energy_estimator in self.triggered_and_passed_hists:
            self.triggered_and_passed_hists[energy_estimator][key].fill(np.abs(passed_events[energy_estimator]), weights=passed_events.TotalWeight)
            self.triggered_and_failed_hists[energy_estimator][key].fill(np.abs(failed_events[energy_estimator]), weights=failed_events.TotalWeight)
            self.triggered_and_passed_hists_2d[energy_estimator][key].fill(np.abs(passed_events[energy_estimator]), passed_events[cut_variable], weights=passed_events.TotalWeight)
            self.triggered_and_failed_hists_2d[energy_estimator][key].fill(np.abs(failed_events[energy_estimator]), failed_events[cut_variable], weights=failed_events.TotalWeight)

    def fill_tagged(self, key, events, passed, fill_2d=True):
        if fill_2d:
            selection_name, cut_variable = key
        passed_events = events[passed]
        failed_events = events[np.invert(passed)]
        for energy_estimator in self.tagged_and_passed_hists:
            self.tagged_and_passed_hists[energy_estimator][key].fill(np.abs(passed_events[energy_estimator]), weights=passed_events.TotalWeight)
            self.tagged_and_failed_hists[energy_estimator][key].fill(np.abs(failed_events[energy_estimator]), weights=failed_events.TotalWeight)
            if fill_2d:
                self.tagged_and_passed_hists_2d[energy_estimator][key].fill(np.abs(passed_events[energy_estimator]), passed_events[cut_variable], weights=passed_events.TotalWeight)
                self.tagged_and_failed_hists_2d[energy_estimator][key].fill(np.abs(failed_events[energy_estimator]), failed_events[cut_variable], weights=failed_events.TotalWeight)

    def save(self, resultdir, prefix):
        energy_estimators = list(self.triggered_and_passed_hists)
        keys_1d = list(self.triggered_and_passed_hists[energy_estimators[0]])
        keys_2d = list(self.triggered_and_passed_hists_2d[energy_estimators[0]])
        results = {"keys_1d": json.dumps(keys_1d), "keys_2d": json.dumps(keys_2d), "energy_estimators": json.dumps(energy_estimators)}
        for key in keys_1d:
            key_str = key if isinstance(key, str) else ":".join(key)
            for energy_estimator in energy_estimators:
                self.triggered_and_passed_hists[energy_estimator][key].add_to_file(results, f"hist_{energy_estimator}_triggered_and_passed_{key_str}")
                self.triggered_and_failed_hists[energy_estimator][key].add_to_file(results, f"hist_{energy_estimator}_triggered_and_failed_{key_str}")
                self.tagged_and_passed_hists[energy_estimator][key].add_to_file(results, f"hist_{energy_estimator}_tagged_and_passed_{key_str}")
                self.tagged_and_failed_hists[energy_estimator][key].add_to_file(results, f"hist_{energy_estimator}_tagged_and_failed_{key_str}")
        for key in keys_2d:
            key_str = ":".join(key)
            for energy_estimator in energy_estimators:
                self.triggered_and_passed_hists_2d[energy_estimator][key].add_to_file(results, f"hist_{energy_estimator}_triggered_and_passed_2d_{key_str}")
                self.triggered_and_failed_hists_2d[energy_estimator][key].add_to_file(results, f"hist_{energy_estimator}_triggered_and_failed_2d_{key_str}")
                self.tagged_and_passed_hists_2d[energy_estimator][key].add_to_file(results, f"hist_{energy_estimator}_tagged_and_passed_2d_{key_str}")
                self.tagged_and_failed_hists_2d[energy_estimator][key].add_to_file(results, f"hist_{energy_estimator}_tagged_and_failed_2d_{key_str}")
        np.savez(os.path.join(resultdir, f"{prefix}.npz"), **results)

    @staticmethod
    def load(resultdir, prefix):
        result = None
        print(os.path.join(resultdir, f"{prefix}_*.npz"), glob(os.path.join(resultdir, f"{prefix}_*.npz")))
        for filename in glob(os.path.join(resultdir, f"{prefix}_*.npz")):
            if result is None:
                result = ResultHists.load_file(filename)
            else:
                result.merge(ResultHists.load_file(filename))
        return result

    @staticmethod
    def load_file(filename):
        with np.load(filename) as file:
            keys_1d = [_tuple_or_str(key) for key in json.loads(file["keys_1d"].item())]
            keys_2d = [_tuple_or_str(key) for key in json.loads(file["keys_2d"].item())]
            energy_estimators = json.loads(file["energy_estimators"].item())
            hists = dict(triggered=dict(passed={}, failed={}), tagged=dict(passed={}, failed={}))
            hists_2d = dict(triggered=dict(passed={}, failed={}), tagged=dict(passed={}, failed={}))
            for presel in ("triggered", "tagged"):
                for success in ("passed", "failed"):
                    for energy_estimator in energy_estimators:
                        hists[presel][success][energy_estimator] = {}
                        hists_2d[presel][success][energy_estimator] = {}
                        for key in keys_1d:
                            key_str = key if isinstance(key, str) else ":".join(key)
                            hists[presel][success][energy_estimator][key] = WeightedHistogram.from_file(file, f"hist_{energy_estimator}_{presel}_and_{success}_{key_str}")
                        for key in keys_2d:
                            key_str = ":".join(key)
                            hists_2d[presel][success][energy_estimator][key] = WeightedHistogram.from_file(file, f"hist_{energy_estimator}_{presel}_and_{success}_2d_{key_str}")
            return ResultHists(
                hists["triggered"]["passed"], hists["triggered"]["failed"],
                hists["tagged"]["passed"], hists["tagged"]["failed"],
                hists_2d["triggered"]["passed"], hists_2d["triggered"]["failed"],
                hists_2d["tagged"]["passed"], hists_2d["tagged"]["failed"])

    def merge(self, other):
        for energy_estimator in self.triggered_and_passed_hists:
            for key in self.triggered_and_passed_hists[energy_estimator]:
                self.triggered_and_passed_hists[energy_estimator][key].add(other.triggered_and_passed_hists[energy_estimator][key])
                self.triggered_and_failed_hists[energy_estimator][key].add(other.triggered_and_failed_hists[energy_estimator][key])
                self.tagged_and_passed_hists[energy_estimator][key].add(other.tagged_and_passed_hists[energy_estimator][key])
                self.tagged_and_failed_hists[energy_estimator][key].add(other.tagged_and_failed_hists[energy_estimator][key])
            for key in self.triggered_and_passed_hists_2d[energy_estimator]:
                self.triggered_and_passed_hists_2d[energy_estimator][key].add(other.triggered_and_passed_hists_2d[energy_estimator][key])
                self.triggered_and_failed_hists_2d[energy_estimator][key].add(other.triggered_and_failed_hists_2d[energy_estimator][key])
                self.tagged_and_passed_hists_2d[energy_estimator][key].add(other.tagged_and_passed_hists_2d[energy_estimator][key])
                self.tagged_and_failed_hists_2d[energy_estimator][key].add(other.tagged_and_failed_hists_2d[energy_estimator][key])


def handle_file(arg):
    filename, treename, chunk_size, rank, nranks, kwargs = arg

    config = kwargs["config"]
    workdir = kwargs["workdir"]
    sample_name = kwargs["sample"]
    preselection_names = kwargs["preselections"]
    energy_estimators = kwargs["energy_estimators"]
    resultdir = kwargs["resultdir"]
    outputprefix = f"{kwargs['outputprefix']}_{rank}_{nranks}"
    verbose = kwargs["verbose"]


    sample = Sample.load(config, sample_name, workdir)
    binnings = sample.binnings
    energy_binning = binnings["energy"]


    if not "tag_cuts" in config:
        raise ValueError("No tag cuts found in configuration.")
    if sample.tag_cuts_name is None:
        raise ValueError(f"No tag sample configured for sample {sample_name!r}.")

    branches = set(energy_estimators)

    tag_cuts_config = config["tag_cuts"][sample.tag_cuts_name]
    tag_selections = {
        tagger: {
            selection_name: Selection.load(config["selections"][selection_name], sample.estimators["Energy"], binnings, config=config, workdir=workdir, fill_hists=False)
            for selection_name in tag_cuts
        }
        for tagger, tag_cuts in tag_cuts_config.items()
    }

    preselections = [
        Selection.load(config["selections"][preselection_name], sample.estimators["Energy"], binnings, config=config, workdir=workdir, fill_hists=False)
        for preselection_name in preselection_names
    ]

    for preselection in preselections:
        branches |= set(preselection.cuts)

    for selections in tag_selections.values():
        for selection in selections.values():
            branches |= set(selection.cuts)

    cuts = {}
    cuts_per_tagger = {tagger: [] for tagger in tag_selections}
    default_tagger = "default"

    for selection_name, selection in sample.selections.items():
        for cut_variable, cut in selection.cuts.items():
            key = (selection_name, cut_variable)
            cuts[key] = cut
            tagger = cut.tagger or default_tagger
            cuts_per_tagger[tagger].append(key)
            branches.add(cut_variable)

    hists = ResultHists.create(energy_estimators, energy_binning, cuts.keys(), cuts_per_tagger, binnings)

    branches = list(branches)

    for events in sample.read_tree(filename, treename, branches=branches, chunk_size=chunk_size, rank=rank, nranks=nranks, verbose=verbose, resultdir=resultdir, prefix=outputprefix, apply_selections=False):
        triggered_events = events
        for preselection in preselections:
            triggered_events = preselection.select(triggered_events)
        for cut_key, cut in cuts.items():
            _, cut_variable = cut_key
            triggered_and_passed = cut.select(triggered_events[cut_variable], triggered_events[sample.estimators["Energy"]], triggered_events.TotalWeight)
            hists.fill_triggered(cut_key, triggered_events, triggered_and_passed)
        for tagger, selections in tag_selections.items():
            tagged_events = triggered_events
            for tag_selection_name, tag_selection in selections.items():
                tagged_events = tag_selection.select(tagged_events)
            tagged_and_passed_all = np.ones(len(tagged_events), dtype=bool)
            for cut_key in cuts_per_tagger[tagger]:
                cut = cuts[cut_key]
                _, cut_variable = cut_key
                tagged_and_passed = cut.select(tagged_events[cut_variable], tagged_events[sample.estimators["Energy"]], tagged_events.TotalWeight)
                tagged_and_passed_all = tagged_and_passed_all & tagged_and_passed
                hists.fill_tagged(cut_key, tagged_events, tagged_and_passed)
            hists.fill_tagged(tagger, tagged_events, tagged_and_passed_all, fill_2d=False)


    hists.save(resultdir, outputprefix)

    return outputprefix


def make_args(filename, treename, chunk_size, nranks, parallel, **kwargs):
    parallel_index, parallel_total = parallel
    assert parallel_index >= 0 and parallel_index < parallel_total
    for rank in range(parallel_index * nranks, (parallel_index + 1) * nranks):
        yield (filename, treename, chunk_size, rank, (nranks * parallel_total), kwargs)


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--tree", required=True, help="Path to tree file.")
    parser.add_argument("--treename", default="PhotonTree", help="Tree name in file.")
    parser.add_argument("--config", required=True, nargs=2, help="Path to analysis config file and workdir.")
    parser.add_argument("--sample", required=True, help="Sample selection from config to apply.")
    parser.add_argument("--preselections", nargs="+", default=[], help="Preselection to apply before anything else.")
    parser.add_argument("--energy-estimators", nargs="+", help="Energy estimators to create the efficiency as a function of.")
    parser.add_argument("--outputprefix", default="Histograms", help="Prefix for the output file.")
    parser.add_argument("--label", help="Dataset label for plots.")
    parser.add_argument("--resultdir", default="results", help="Directory to store temporary files and results in.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of parallel processes.")
    parser.add_argument("--parallel", type=int, nargs=2, default=(0, 1), help="INdex of this job and total number of parallel processes if multiple jobs are running in parallel.")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Number of events per chunk.")
    parser.add_argument("--plot-style", default="mc", choices=["mc", "iss"], help="Plot style (\"mc\" or \"iss\").")
    parser.add_argument("--verbose", action="store_true", help="Print progress.")

    args = parser.parse_args()

    config_file, workdir = args.config
    config = get_config(config_file)
    sample = Sample.load(config, args.sample, workdir)
    binnings = sample.binnings

    energy_estimators = args.energy_estimators
    if energy_estimators is None:
        energy_estimators = (sample.estimators["Energy"],)

    os.makedirs(args.resultdir, exist_ok=True)

    results = None

    parallel = args.parallel
    parallel_index, parallel_total = parallel
    prefix = f"{args.outputprefix}_{parallel_index}"

    with mp.get_context("fork" if args.verbose else "spawn").Pool(args.nprocesses) as pool:
        pool_args = make_args(args.tree, args.treename, args.chunk_size, args.nprocesses, parallel,
                              sample=args.sample, preselections=args.preselections,
                              resultdir=args.resultdir, outputprefix=prefix,
                              workdir=workdir,
                              energy_estimators=energy_estimators,
                              config=config, verbose=args.verbose)
        for rank_prefix in pool.imap_unordered(handle_file, pool_args):
            rank_result = ResultHists.load_file(os.path.join(args.resultdir, f"{rank_prefix}.npz"))
            if results is None:
                results = rank_result
            else:
                results.merge(rank_result)
            os.remove(os.path.join(args.resultdir, f"{rank_prefix}.npz"))

    results.save(args.resultdir, prefix)


if __name__ == "__main__":
    main()
