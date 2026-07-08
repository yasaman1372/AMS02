#!/usr/bin/env python3

import multiprocessing as mp
import os

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline, BSpline

from tools.binnings import make_lin_binning, make_energy_binning_from_config, make_mc_dataset_edge_binning, combine_binnings, reduce_bins, make_flux_energy_binning
from tools.config import get_config
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d
from tools.roottree import read_tree
from tools.statistics import hist_mean_and_std, calculate_chisq
from tools.utilities import save_figure, set_energy_ticks
from tools.constants import BastianPlots

def power_law(x, a, b ,c):
    return a * x**b + c

def log_fit(x, a, b):
    return a * np.log(x) + b

def exponential_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

def broken_power_law(x, b, c, d, e):
    y = np.where(x < 30, b * x**c, d * x**e)
    return y

def handle_file(arg):
    filenames, treename, chunk_size, rank, nranks, kwargs = arg

    energy_branch = kwargs["energy_branch"]
    mc_energy_branch = kwargs["mc_energy_branch"]

    branches = [energy_branch, mc_energy_branch, "TotalWeight"]

    energy_binning = kwargs["energy_binning"]
    relative_difference_binning = kwargs["relative_difference_binning"]

    migration_histogram = Histogram(energy_binning, energy_binning)
    relative_difference_histogram = Histogram(energy_binning, relative_difference_binning)
    relative_difference_histogram_rec = WeightedHistogram(energy_binning, relative_difference_binning)

    for events in read_tree(filenames, treename, rank=rank, nranks=nranks, chunk_size=chunk_size, branches=branches, cache_file=False):
        migration_histogram.fill(events[mc_energy_branch], events[energy_branch])
        relative_difference_histogram.fill(events[mc_energy_branch], (events[energy_branch] - events[mc_energy_branch]) / events[mc_energy_branch])
        relative_difference_histogram_rec.fill(events[energy_branch], (events[energy_branch] - events[mc_energy_branch]) / events[mc_energy_branch], weights=events.TotalWeight)

    return migration_histogram, relative_difference_histogram, relative_difference_histogram_rec


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
    parser.add_argument("--outputprefix", default="EnergyResolution", help="Prefix for plots and result files.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Number of events to read each step.")
    parser.add_argument("--title", default="Energy Resolution", help="Title for plots.")
    parser.add_argument("--recon-energy-branch", default="Energy", help="The branch that contains the reconstructed Energy")
    parser.add_argument("--mc-energy-branch", default="McEnergy", help="The branch containing the MC Truth energy to use")
    parser.add_argument("--comparison", choices=list(BastianPlots.keys()), help="Version of bastians data to compare to.")
    parser.add_argument("--no-title", action='store_true', help='Use this to prevent titles from being generated.')
    parser.add_argument("--recon-energy-label", default='rec', help='How the reconstructed energy should be names E_???')
    parser.add_argument("--transparent", action="store_true", help='set this to make the background of the png transparent')
    parser.add_argument("--energy-binning", default='default', choices=['default', 'flux', 'config'], help='what energy binning should be used')

    args = parser.parse_args()
    config_filename, workdir = args.config
    config = get_config(config_filename)

    energy_branch = args.recon_energy_branch
    mc_energy_branch = args.mc_energy_branch

    os.makedirs(args.resultdir, exist_ok=True)
    os.makedirs(args.plotdir, exist_ok=True)

    if args.energy_binning == 'default':
        energy_binning = make_energy_binning_from_config(config)
    elif args.energy_binning == 'config':
        energy_binning = make_energy_binning_from_config(config)
    elif args.energy_binning == 'flux':
        energy_binning = make_flux_energy_binning()
    energy_binning_with_dataset_edges = combine_binnings((energy_binning, make_mc_dataset_edge_binning(config, args.dataset)))
    relative_difference_binning = make_lin_binning(-1, 1, 100)

    comparison = None
    if args.comparison is not None:
        comparison = BastianPlots[args.comparison]["energy resolution"]

    migration_histogram = Histogram(energy_binning_with_dataset_edges, energy_binning_with_dataset_edges, labels=("$E_{MC}$ / GeV", f"$E_{{{args.recon_energy_label}}}$ / GeV"))
    relative_difference_histogram = Histogram(energy_binning_with_dataset_edges, relative_difference_binning, labels=("$E_{MC}$ / GeV", f"$(E_{{{args.recon_energy_label}}}-E_{{MC}})/E_{{MC}}$"))
    relative_difference_histogram_rec = WeightedHistogram(energy_binning_with_dataset_edges, relative_difference_binning, labels=(f"$E_{{{args.recon_energy_label}}}$ / GeV", f"$(E_{{{args.recon_energy_label}}}-E_{{MC}})/E_{{MC}}$"))


    with mp.Pool(args.nprocesses) as pool:
        pool_args = make_args(args.tree, args.treename, chunk_size=args.chunk_size, nranks=args.nprocesses, energy_branch=energy_branch, mc_energy_branch=mc_energy_branch, energy_binning=energy_binning_with_dataset_edges, relative_difference_binning=relative_difference_binning)
        for mig_hist, rel_diff_hist, rel_diff_hist_rec in pool.imap_unordered(handle_file, pool_args):
            migration_histogram.add(mig_hist)
            relative_difference_histogram.add(rel_diff_hist)
            relative_difference_histogram_rec.add(rel_diff_hist_rec)

    results = {}

    migration_histogram = migration_histogram.rebin(energy_binning, energy_binning)
    relative_difference_histogram = relative_difference_histogram.rebin(energy_binning, relative_difference_binning)
    relative_difference_histogram_rec = relative_difference_histogram_rec.rebin(energy_binning, relative_difference_binning)
    migration_histogram.add_to_file(results, "migration_hist")
    relative_difference_histogram.add_to_file(results, "relative_difference_hist")
    relative_difference_histogram_rec.add_to_file(results, "relative_difference_hist_rec")

    migration_figure = plt.figure(figsize=(6, 5))
    migration_plot = migration_figure.subplots(1, 1)
    plot_histogram_2d(migration_plot, migration_histogram, scale=1 / migration_histogram.values.sum(axis=1), log=True, show_overflow=False, cmap="jet")
    migration_plot.plot(energy_binning.edges[1:-1], energy_binning.edges[1:-1], "-", color="gray", alpha=0.5, linewidth=1)
    migration_plot.set_xlim(energy_binning.edges[1], energy_binning.edges[-2])
    migration_plot.set_ylim(energy_binning.edges[1], energy_binning.edges[-2])
    set_energy_ticks(migration_plot)
    set_energy_ticks(migration_plot, axis="y")
    migration_figure.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
    if args.no_title:
        save_figure(migration_figure, args.plotdir, f"{args.outputprefix}_migration_NoTitle", transparent=args.transparent)
    migration_plot.set_title(args.title)
    save_figure(migration_figure, args.plotdir, f"{args.outputprefix}_migration", transparent=args.transparent)

    reldiff_mean, reldiff_std, reldiff_mean_error = hist_mean_and_std(relative_difference_histogram)
    results["energy_values"] = energy_binning.bin_centers[1:-1]
    results["relative_bias"] = reldiff_mean
    results["relative_resolution"] = reldiff_std

    reldiff_mean_rec, reldiff_std_rec, reldiff_mean_error_rec = hist_mean_and_std(relative_difference_histogram_rec)
    results["energy_values"] = energy_binning.bin_centers[1:-1]
    results["relative_bias_rec"] = reldiff_mean_rec
    results["relative_resolution_rec"] = reldiff_std_rec


    relative_difference_figure = plt.figure(figsize=(6, 5))
    relative_difference_plot = relative_difference_figure.subplots(1, 1)
    plot_histogram_2d(relative_difference_plot, relative_difference_histogram, scale=1 / relative_difference_histogram.values.sum(axis=1), log=True, show_overflow=False, cmap="jet")
    relative_difference_plot.plot(energy_binning.edges[1:-1], np.zeros_like(energy_binning.edges[1:-1]), "-", color="gray", alpha=0.5, linewidth=1)
    relative_difference_plot.plot(energy_binning.bin_centers[1:-1], reldiff_mean, ".", markersize=1, color="black")
    relative_difference_plot.plot(energy_binning.bin_centers[1:-1], reldiff_mean + reldiff_std, "^", markersize=1, color="black")
    relative_difference_plot.plot(energy_binning.bin_centers[1:-1], reldiff_mean - reldiff_std, "v", markersize=1, color="black")
    relative_difference_plot.set_xlim(energy_binning.edges[1], energy_binning.edges[-2])
    relative_difference_plot.set_ylim(relative_difference_binning.edges[1], relative_difference_binning.edges[-2])
    set_energy_ticks(relative_difference_plot)
    relative_difference_figure.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
    if args.no_title:
        save_figure(relative_difference_figure, args.plotdir, f"{args.outputprefix}_relative_difference_NoTitle", transparent=args.transparent)
    relative_difference_plot.set_title(args.title)
    save_figure(relative_difference_figure, args.plotdir, f"{args.outputprefix}_relative_difference", transparent=args.transparent)

    if comparison is not None:
        relative_difference_figure = plt.figure(figsize=(8, 4.2))
        relative_difference_plot = relative_difference_figure.subplots(1, 1)
        plot_histogram_2d(relative_difference_plot, relative_difference_histogram, scale=1 / relative_difference_histogram.values.sum(axis=1), log=True, show_overflow=False, cmap="jet")
        relative_difference_plot.plot(energy_binning.edges[1:-1], np.zeros_like(energy_binning.edges[1:-1]), "-", color="gray", alpha=0.5, linewidth=1)
        relative_difference_plot.plot(energy_binning.bin_centers[1:-1], reldiff_mean, ".", markersize=1, color="black")
        relative_difference_plot.plot(energy_binning.bin_centers[1:-1], reldiff_mean + reldiff_std, "^", markersize=1, color="black")
        relative_difference_plot.plot(energy_binning.bin_centers[1:-1], reldiff_mean - reldiff_std, "v", markersize=1, color="black")
        relative_difference_plot.plot(comparison["mean"]["x"], comparison["mean"]["y"], color = 'green', label = 'Bastian')
        relative_difference_plot.plot(comparison["rms"]["x"], comparison["rms"]["y"], color = 'blue', label = 'Bastian')
        relative_difference_plot.set_xlim(energy_binning.edges[1], energy_binning.edges[-2])
        relative_difference_plot.set_ylim(relative_difference_binning.edges[1], relative_difference_binning.edges[-2])
        set_energy_ticks(relative_difference_plot)
        relative_difference_figure.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
        if args.no_title:
            save_figure(relative_difference_figure, args.plotdir, f"{args.outputprefix}_relative_difference_comparison_NoTitle", transparent=args.transparent)
        relative_difference_plot.set_title(args.title)
        save_figure(relative_difference_figure, args.plotdir, f"{args.outputprefix}_relative_difference_comparison", transparent=args.transparent)

        valid = ~(np.isnan(reldiff_mean) | np.isnan(reldiff_mean_error))
        popt_log, pcov = curve_fit(log_fit, energy_binning.bin_centers[1:-1][valid], reldiff_mean[valid], sigma = reldiff_mean_error[valid], absolute_sigma = True, maxfev=10000)
        _,_,chi2_log = calculate_chisq(reldiff_mean[valid], log_fit(energy_binning.bin_centers[1:-1][valid], *popt_log), reldiff_mean_error[valid], len(popt_log))
        print(popt_log, chi2_log, flush = True)
        label_log = f'${{{popt_log[0]:.3f}}}*\log(E)+{{{popt_log[1]:.3f}}}$\n'+\
                    f'$\chi^2: {{{chi2_log:.3f}}}$'
        bspline = UnivariateSpline(energy_binning.bin_centers[1:-1][valid], reldiff_mean[valid], w = 1/reldiff_mean_error[valid], s = 1000)
        
        relative_bias_figure = plt.figure(figsize=(8, 4.2))
        relative_bias_plot = relative_bias_figure.subplots(1, 1)
        relative_bias_plot.errorbar(energy_binning.bin_centers[1:-1], reldiff_mean, yerr=reldiff_mean_error, ls = '',marker = "o", markersize=1, color="black", label = 'This Analysis')
        relative_bias_plot.plot(comparison["mean"]["x"], comparison["mean"]["y"], color = 'red', label = 'Bastian')
        relative_bias_plot.plot(energy_binning.bin_centers[1:-1][valid], log_fit(energy_binning.bin_centers[1:-1][valid], *popt_log), color='green', label = label_log)
        relative_bias_plot.plot(energy_binning.bin_centers[1:-1][valid], bspline(energy_binning.bin_centers[1:-1][valid]), color='magenta', label = 'spline fit', alpha = 0.75)
        relative_bias_plot.set_xlabel("$E_{MC} / GeV$")
        relative_bias_plot.set_ylabel("Relative Bias")
        relative_bias_plot.semilogx()
        relative_bias_plot.set_xlim(1.5*1e-1, energy_binning.edges[-2])
        relative_bias_plot.set_ylim(-0.5,0.5)
        relative_bias_plot.axhline(0, color="grey")
        set_energy_ticks(relative_bias_plot)
        relative_bias_plot.legend()
        if args.no_title:
            save_figure(relative_bias_figure, args.plotdir, f"{args.outputprefix}_relative_bias_NoTitle", transparent=args.transparent)
        relative_bias_plot.set_title(f"{args.title} Relative Energy Bias")
        save_figure(relative_bias_figure, args.plotdir, f"{args.outputprefix}_relative_bias", transparent=args.transparent)

        valid = ~(np.isnan(reldiff_mean_rec) | np.isnan(reldiff_mean_error_rec))
        valid = valid & (energy_binning.bin_centers[1:-1] <= 30) & (energy_binning.bin_centers[1:-1] >= 0.5)
        popt_log, pcov = curve_fit(log_fit, energy_binning.bin_centers[1:-1][valid], reldiff_mean_rec[valid], sigma = reldiff_mean_error_rec[valid], absolute_sigma = True, maxfev=10000)
        _,_,chi2_log = calculate_chisq(reldiff_mean_rec[valid], log_fit(energy_binning.bin_centers[1:-1][valid], *popt_log), reldiff_mean_error_rec[valid], len(popt_log))
        print(popt_log, chi2_log, flush = True)
        label_log = f'${{{popt_log[0]:.3f}}}*\log(E)+{{{popt_log[1]:.3f}}}$\n'+\
                    f'$\chi^2: {{{chi2_log:.3f}}}$'
        bspline = UnivariateSpline(energy_binning.bin_centers[1:-1][valid], reldiff_mean[valid], w = 1/reldiff_mean_error[valid], s = 1000)
        
        relative_bias_figure = plt.figure(figsize=(8, 4.2))
        relative_bias_plot = relative_bias_figure.subplots(1, 1)
        relative_bias_plot.errorbar(energy_binning.bin_centers[1:-1], reldiff_mean_rec, yerr=reldiff_mean_error_rec, ls = '',marker = "o", markersize=1, color="black", label = 'This Analysis')
        #relative_bias_plot.plot(comparison["mean"]["x"], comparison["mean"]["y"], color = 'red', label = 'Bastian')
        relative_bias_plot.plot(energy_binning.bin_centers[1:-1][valid], log_fit(energy_binning.bin_centers[1:-1][valid], *popt_log), color='green', label = label_log)
        #relative_bias_plot.plot(energy_binning.bin_centers[1:-1][valid], bspline(energy_binning.bin_centers[1:-1][valid]), color='magenta', label = 'spline fit', alpha = 0.75)
        relative_bias_plot.set_xlabel(f"$E_{{{args.recon_energy_label}}} / GeV$")
        relative_bias_plot.set_ylabel("Relative Bias")
        relative_bias_plot.semilogx()
        relative_bias_plot.set_xlim(1.5*1e-1, energy_binning.edges[-2])
        relative_bias_plot.set_ylim(-0.5,0.5)
        relative_bias_plot.axhline(0, color="grey")
        set_energy_ticks(relative_bias_plot)
        relative_bias_plot.legend()
        if args.no_title:
            save_figure(relative_bias_figure, args.plotdir, f"{args.outputprefix}_relative_bias_{args.recon_energy_label}_NoTitle", transparent=args.transparent)
        relative_bias_plot.set_title(f"{args.title} Relative Energy Bias")
        save_figure(relative_bias_figure, args.plotdir, f"{args.outputprefix}_relative_bias_{args.recon_energy_label}", transparent=args.transparent)


        relative_resolution_figure = plt.figure(figsize=(8, 4.2))
        relative_resolution_plot = relative_resolution_figure.subplots(1, 1)
        relative_resolution_plot.plot(energy_binning.bin_centers[1:-1], reldiff_std, "o", markersize=1, color="black", label = 'This Analysis')
        relative_resolution_plot.plot(comparison["rms"]["x"], comparison["rms"]["y"], color = 'red', label = 'Bastian')
        relative_resolution_plot.set_xlabel("$E_{MC} / GeV$")
        relative_resolution_plot.set_ylabel("Relative Resolution")
        relative_resolution_plot.set_xlim(1.5*1e-1, energy_binning.edges[-2])
        relative_resolution_plot.set_ylim(0,0.6)
        relative_resolution_plot.semilogx()
        set_energy_ticks(relative_resolution_plot)
        relative_resolution_plot.legend()
        if args.no_title:
            save_figure(relative_resolution_figure, args.plotdir, f"{args.outputprefix}_relative_resolution_NoTitle", transparent=args.transparent)
        relative_resolution_plot.set_title(f"{args.title} Relative Energy Resolution")
        save_figure(relative_resolution_figure, args.plotdir, f"{args.outputprefix}_relative_resolution", transparent=args.transparent)


    np.savez_compressed(os.path.join(args.resultdir, f"{args.outputprefix}.npz"), **results)

if __name__ == "__main__":
    main()
