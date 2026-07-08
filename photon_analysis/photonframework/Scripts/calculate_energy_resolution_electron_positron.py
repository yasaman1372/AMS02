#!/usr/bin/env python3

import multiprocessing as mp
import os

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

from tools.binnings import make_lin_binning, make_energy_binning_from_config, make_mc_dataset_edge_binning, combine_binnings, reduce_bins
from tools.config import get_config
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d
from tools.roottree import read_tree
from tools.statistics import hist_mean_and_std
from tools.utilities import save_figure, set_energy_ticks

BRANCHES = ["TrkTrackBestPairElectronRigidity", "TrkTrackBestPairPositronRigidity", "McSecondaryPairProductionElectronMomentum", "McSecondaryPairProductionPositronMomentum"]

def handle_file(arg):
    filenames, treename, chunk_size, rank, nranks, kwargs = arg

    energy_binning = kwargs["energy_binning"]
    relative_difference_binning = kwargs["relative_difference_binning"]

    migration_electron_histogram = Histogram(energy_binning, energy_binning)
    relative_difference_electron_histogram = Histogram(energy_binning, relative_difference_binning)
    migration_positron_histogram = Histogram(energy_binning, energy_binning)
    relative_difference_positron_histogram = Histogram(energy_binning, relative_difference_binning)
    
    for events in read_tree(filenames, treename, rank=rank, nranks=nranks, chunk_size=chunk_size, branches=BRANCHES, cache_file=False):
        migration_electron_histogram.fill(events.McSecondaryPairProductionElectronMomentum, np.abs(events.TrkTrackBestPairElectronRigidity))
        print(events.TrkTrackBestPairElectronRigidity, flush = True)
        migration_positron_histogram.fill(events.McSecondaryPairProductionPositronMomentum, np.abs(events.TrkTrackBestPairPositronRigidity))
        relative_difference_positron_histogram.fill(events.McSecondaryPairProductionPositronMomentum, (np.abs(events.TrkTrackBestPairPositronRigidity) - events.McSecondaryPairProductionPositronMomentum) / events.McSecondaryPairProductionPositronMomentum)

    return migration_electron_histogram, relative_difference_electron_histogram, migration_positron_histogram, relative_difference_positron_histogram


def make_args(filename, treename, chunk_size, nranks, **kwargs):
    for rank in range(nranks):
        yield (filename, treename, chunk_size, rank, nranks, kwargs)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs=2, required=True, help="Path to config file and working directory.")
    parser.add_argument("--tree", nargs="+", required=True, help="MC Gamma Tree to read.")
    parser.add_argument("--treename", default="GammaTree", help="Name of the tree in the ROOT file to read.")
    parser.add_argument("--dataset", required=True, help="Name of the MC dataset to calculate the energy resolution for.")
    parser.add_argument("--outputprefix", default="EnergyResolutionElectronPositron", help="Prefix for plots and result files.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Number of events to read each step.")
    parser.add_argument("--title", default="Energy Resolution", help="Title for plots.")

    args = parser.parse_args()
    config_filename, workdir = args.config
    config = get_config(config_filename)

    os.makedirs(args.resultdir, exist_ok=True)
    os.makedirs(args.plotdir, exist_ok=True)

    energy_binning = make_energy_binning_from_config(config)
    energy_binning_with_dataset_edges = combine_binnings((energy_binning, make_mc_dataset_edge_binning(config, args.dataset)))
    relative_difference_binning = make_lin_binning(-1, 1, 100)

    migration_electron_histogram = Histogram(energy_binning_with_dataset_edges, energy_binning_with_dataset_edges, labels=("$R_{MC}$ / GeV", "$R_{rec}$ / GeV"))
    relative_difference_electron_histogram = Histogram(energy_binning_with_dataset_edges, relative_difference_binning, labels=("$R_{MC}$ / GeV", "$(R_{rec}-R_{MC})/R_{MC}$"))

    migration_positron_histogram = Histogram(energy_binning_with_dataset_edges, energy_binning_with_dataset_edges, labels=("$R_{MC}$ / GeV", "$R_{rec}$ / GeV"))
    relative_difference_positron_histogram = Histogram(energy_binning_with_dataset_edges, relative_difference_binning, labels=("$R_{MC}$ / GeV", "$(R_{rec}-R_{MC})/R_{MC}$"))


    with mp.Pool(args.nprocesses) as pool:
        pool_args = make_args(args.tree, args.treename, chunk_size=args.chunk_size, nranks=args.nprocesses, energy_binning=energy_binning_with_dataset_edges, relative_difference_binning=relative_difference_binning)
        for mig_e_hist, rel_diff_e_hist, mig_p_hist, rel_diff_p_hist in pool.imap_unordered(handle_file, pool_args):
            migration_electron_histogram.add(mig_e_hist)
            relative_difference_electron_histogram.add(rel_diff_e_hist)
            migration_positron_histogram.add(mig_p_hist)
            relative_difference_positron_histogram.add(rel_diff_p_hist)

    results = {}

    migration_electron_histogram = migration_electron_histogram.rebin(energy_binning, energy_binning)
    relative_difference_electron_histogram = relative_difference_electron_histogram.rebin(energy_binning, relative_difference_binning)
    migration_electron_histogram.add_to_file(results, "migration_electron_hist")
    relative_difference_electron_histogram.add_to_file(results, "relative_difference_electron_hist")
    
    migration_positron_histogram = migration_positron_histogram.rebin(energy_binning, energy_binning)
    relative_difference_positron_histogram = relative_difference_positron_histogram.rebin(energy_binning, relative_difference_binning)
    migration_positron_histogram.add_to_file(results, "migration_positron_hist")
    relative_difference_positron_histogram.add_to_file(results, "relative_difference_positron_hist")


    migration_e_figure = plt.figure(figsize=(8, 4.2))
    migration_e_plot = migration_e_figure.subplots(1, 1)
    migration_e_plot.set_title(f"{args.title}, Electrons")
    plot_histogram_2d(migration_e_plot, migration_electron_histogram, scale=1 / migration_electron_histogram.values.sum(axis=1), log=True, show_overflow=False, cmap = "jet")
    migration_e_plot.plot(energy_binning.edges[1:-1], energy_binning.edges[1:-1], "-", color="gray", alpha=0.5, linewidth=1)
    migration_e_plot.set_xlim(energy_binning.edges[1], energy_binning.edges[-2])
    migration_e_plot.set_ylim(energy_binning.edges[1], energy_binning.edges[-2])
    set_energy_ticks(migration_e_plot)
    set_energy_ticks(migration_e_plot, axis="y")
    migration_e_figure.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
    save_figure(migration_e_figure, args.plotdir, f"{args.outputprefix}_migration_electron")

    migration_p_figure = plt.figure(figsize=(8, 4.2))
    migration_p_plot = migration_p_figure.subplots(1, 1)
    migration_p_plot.set_title(f"{args.title}, Positrons")
    plot_histogram_2d(migration_p_plot, migration_positron_histogram, scale=1 / migration_positron_histogram.values.sum(axis=1), log=True, show_overflow=False, cmap = "jet")
    migration_p_plot.plot(energy_binning.edges[1:-1], energy_binning.edges[1:-1], "-", color="gray", alpha=0.5, linewidth=1)
    migration_p_plot.set_xlim(energy_binning.edges[1], energy_binning.edges[-2])
    migration_p_plot.set_ylim(energy_binning.edges[1], energy_binning.edges[-2])
    set_energy_ticks(migration_p_plot)
    set_energy_ticks(migration_p_plot, axis="y")
    migration_p_figure.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
    save_figure(migration_p_figure, args.plotdir, f"{args.outputprefix}_migration_positron")



    reldiff_e_mean, reldiff_e_std, reldiff_mean_e_error = hist_mean_and_std(relative_difference_electron_histogram)
    results["energy_values_electron"] = energy_binning.bin_centers[1:-1]
    results["relative_bias_electron"] = reldiff_e_mean
    results["relative_resolution_electron"] = reldiff_e_std

    reldiff_p_mean, reldiff_p_std, reldiff_mean_p_error = hist_mean_and_std(relative_difference_positron_histogram)
    results["energy_values_positron"] = energy_binning.bin_centers[1:-1]
    results["relative_bias_positron"] = reldiff_p_mean
    results["relative_resolution_positron"] = reldiff_p_std

    relative_difference_e_figure = plt.figure(figsize=(8, 4.2))
    relative_difference_e_plot = relative_difference_e_figure.subplots(1, 1)
    relative_difference_e_plot.set_title(f"{args.title}, Electrons")
    plot_histogram_2d(relative_difference_e_plot, relative_difference_electron_histogram, scale=1 / relative_difference_electron_histogram.values.sum(axis=1), log=True, show_overflow=False, cmap = "jet")
    relative_difference_e_plot.plot(energy_binning.edges[1:-1], np.zeros_like(energy_binning.edges[1:-1]), "-", color="gray", alpha=0.5, linewidth=1)
    relative_difference_e_plot.plot(energy_binning.bin_centers[1:-1], reldiff_e_mean, ".", markersize=1, color="black")
    relative_difference_e_plot.plot(energy_binning.bin_centers[1:-1], reldiff_e_mean + reldiff_e_std, "^", markersize=1, color="black")
    relative_difference_e_plot.plot(energy_binning.bin_centers[1:-1], reldiff_e_mean - reldiff_e_std, "v", markersize=1, color="black")
    relative_difference_e_plot.set_xlim(energy_binning.edges[1], energy_binning.edges[-2])
    relative_difference_e_plot.set_ylim(relative_difference_binning.edges[1], relative_difference_binning.edges[-2])
    set_energy_ticks(relative_difference_e_plot)
    relative_difference_e_figure.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
    save_figure(relative_difference_e_figure, args.plotdir, f"{args.outputprefix}_relative_difference_electron")

    relative_difference_p_figure = plt.figure(figsize=(8, 4.2))
    relative_difference_p_plot = relative_difference_p_figure.subplots(1, 1)
    relative_difference_p_plot.set_title(f"{args.title}, Positrons")
    plot_histogram_2d(relative_difference_p_plot, relative_difference_positron_histogram, scale=1 / relative_difference_positron_histogram.values.sum(axis=1), log=True, show_overflow=False, cmap = "jet")
    relative_difference_p_plot.plot(energy_binning.edges[1:-1], np.zeros_like(energy_binning.edges[1:-1]), "-", color="gray", alpha=0.5, linewidth=1)
    relative_difference_p_plot.plot(energy_binning.bin_centers[1:-1], reldiff_p_mean, ".", markersize=1, color="black")
    relative_difference_p_plot.plot(energy_binning.bin_centers[1:-1], reldiff_p_mean + reldiff_p_std, "^", markersize=1, color="black")
    relative_difference_p_plot.plot(energy_binning.bin_centers[1:-1], reldiff_p_mean - reldiff_p_std, "v", markersize=1, color="black")
    relative_difference_p_plot.set_xlim(energy_binning.edges[1], energy_binning.edges[-2])
    relative_difference_p_plot.set_ylim(relative_difference_binning.edges[1], relative_difference_binning.edges[-2])
    set_energy_ticks(relative_difference_p_plot)
    relative_difference_p_figure.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
    save_figure(relative_difference_p_figure, args.plotdir, f"{args.outputprefix}_relative_difference_positron")


    np.savez_compressed(os.path.join(args.resultdir, f"{args.outputprefix}.npz"), **results)

if __name__ == "__main__":
    main()
