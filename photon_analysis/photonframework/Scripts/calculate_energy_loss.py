#!/usr/bin/env python3

import multiprocessing as mp
import os

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from tools.binnings import make_lin_binning, make_energy_binning_from_config, make_mc_dataset_edge_binning, combine_binnings, reduce_bins, make_log_binning
from tools.config import get_config
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d
from tools.roottree import read_tree
from tools.utilities import save_figure, set_energy_ticks
from tools.statistics import hist_mean_and_std, calculate_chisq

BRANCHES = []

def power_law(x, a, b, c, d, e, f):
    y = np.where(x<10, a + b * x**c, d + e * x**f)
    return y

def straight(x, a):
    return a + x * 0

def handle_file(arg):
    filenames, treename, chunk_size, rank, nranks, Bremsstrahlungs_Energy_branch, Energy_branch, McEnergy_branch, kwargs = arg

    energy_binning = kwargs["energy_binning"]
    relative_energy_binning = kwargs["relative_energy_binning"]
    relative_energy_binning_log = kwargs["relative_energy_binning_log"]

    relative_energy_loss_mc_histogram = Histogram(energy_binning, relative_energy_binning)
    relative_energy_loss_rec_histogram = Histogram(energy_binning, relative_energy_binning)
    relative_energy_loss_mc_log_histogram = Histogram(energy_binning, relative_energy_binning_log)
    relative_energy_loss_rec_log_histogram = Histogram(energy_binning, relative_energy_binning_log)
    energy_loss_mc_histogram = Histogram(energy_binning, energy_binning)
    energy_loss_rec_histogram = Histogram(energy_binning, energy_binning)


    for events in read_tree(filenames, treename, rank=rank, nranks=nranks, chunk_size=chunk_size, branches=BRANCHES, cache_file=False):
        bremsstrahlungs_loss = events[Bremsstrahlungs_Energy_branch] - events[Energy_branch]
        relative_energy_loss_mc_histogram.fill(events[McEnergy_branch], bremsstrahlungs_loss/events[McEnergy_branch])
        relative_energy_loss_rec_histogram.fill(events[Energy_branch], bremsstrahlungs_loss/events[Energy_branch])
        relative_energy_loss_mc_log_histogram.fill(events[McEnergy_branch], bremsstrahlungs_loss/events[McEnergy_branch])
        relative_energy_loss_rec_log_histogram.fill(events[Energy_branch], bremsstrahlungs_loss/events[Energy_branch])
        energy_loss_mc_histogram.fill(events[McEnergy_branch], bremsstrahlungs_loss)
        energy_loss_rec_histogram.fill(events[Energy_branch], bremsstrahlungs_loss)

    return relative_energy_loss_mc_histogram, relative_energy_loss_rec_histogram, relative_energy_loss_mc_log_histogram, relative_energy_loss_rec_log_histogram, energy_loss_mc_histogram, energy_loss_rec_histogram 


def make_args(filename, treename, chunk_size, nranks, bremsstrahlungs_energy_branch, energy_branch, mcenergy_branch,  **kwargs):
    for rank in range(nranks):
        yield (filename, treename, chunk_size, rank, nranks, bremsstrahlungs_energy_branch, energy_branch, mcenergy_branch, kwargs)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs=2, required=True, help="Path to config file and working directory.")
    parser.add_argument("--tree", nargs="+", required=True, help="MC Gamma Tree to read.")
    parser.add_argument("--treename", default="GammaTree", help="Name of the tree in the ROOT file to read.")
    parser.add_argument("--dataset", required=True, help="Name of the MC dataset to calculate the energy resolution for.")
    parser.add_argument("--outputprefix", default="BremsstrahlungsLoss", help="Prefix for plots and result files.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Number of events to read each step.")
    parser.add_argument("--title", default="Bremsstrahlungs loss", help="Title for plots.")
    parser.add_argument("--bremsstrahlungs-energy-branch", default="TrkTrackBestPairEnergyWithBremsstrahlungBeforeL3")
    parser.add_argument("--recon-energy-branch", default="Energy", help="The branch that contains the reconstructed Energy")
    parser.add_argument("--mc-energy-branch", default="McEnergy", help="The branch containing the MC Truth energy to use")
    parser.add_argument("--no-title", action='store_true', help='Use this to prevent titles from being generated.')
    parser.add_argument("--transparent", action="store_true", help='set this to make the background of the png transparent')

    args = parser.parse_args()
    config_filename, workdir = args.config
    config = get_config(config_filename)

    Bremsstrahlung_energy_branch = args.bremsstrahlungs_energy_branch
    Energy_branch = args.recon_energy_branch
    McEnergy_branch = args.mc_energy_branch
    BRANCHES.append(Energy_branch)
    BRANCHES.append(McEnergy_branch)
    BRANCHES.append(Bremsstrahlung_energy_branch)

    os.makedirs(args.resultdir, exist_ok=True)
    os.makedirs(args.plotdir, exist_ok=True)

    energy_binning = make_energy_binning_from_config(config)
    energy_binning_with_dataset_edges = combine_binnings((energy_binning, make_mc_dataset_edge_binning(config, args.dataset)))
    relative_energy_binning = make_lin_binning(0, 1, 100)
    relative_energy_binning_log = make_log_binning(1e-7, 1, 100)

    relative_energy_loss_mc_histogram = Histogram(energy_binning_with_dataset_edges, relative_energy_binning, labels=("$E_{MC}$ / GeV", "$E_{Bremsstrahlung}/E_{MC}$"))
    relative_energy_loss_rec_histogram = Histogram(energy_binning_with_dataset_edges, relative_energy_binning, labels=("$E_{rec}$ / GeV", "$E_{Bremsstrahlung}/E_{rec}$"))
    relative_energy_loss_mc_log_histogram = Histogram(energy_binning_with_dataset_edges, relative_energy_binning_log, labels=("$E_{MC}$ / GeV", "$E_{Bremsstrahlung}/E_{MC}$"))
    relative_energy_loss_rec_log_histogram = Histogram(energy_binning_with_dataset_edges, relative_energy_binning_log, labels=("$E_{rec}$ / GeV", "$E_{Bremsstrahlung}/E_{rec}$"))
    energy_loss_mc_histogram = Histogram(energy_binning_with_dataset_edges, energy_binning_with_dataset_edges, labels=("$E_{MC}$ / GeV", "$E_{Bremsstrahlung}$ / GeV"))
    energy_loss_rec_histogram = Histogram(energy_binning_with_dataset_edges, energy_binning_with_dataset_edges, labels=("$E_{rec}$ / GeV", "$E_{Bremsstrahlung}$ / GeV"))

    with mp.Pool(args.nprocesses) as pool:
        pool_args = make_args(args.tree, args.treename, chunk_size=args.chunk_size, nranks=args.nprocesses, bremsstrahlungs_energy_branch=Bremsstrahlung_energy_branch ,energy_branch=Energy_branch, mcenergy_branch=McEnergy_branch, energy_binning=energy_binning_with_dataset_edges, relative_energy_binning=relative_energy_binning, relative_energy_binning_log=relative_energy_binning_log)
        for rel_mc_hist, rel_rec_hist, rel_mc_log_hist, rel_rec_log_hist, mc_hist, rec_hist in pool.imap_unordered(handle_file, pool_args):
            relative_energy_loss_mc_histogram.add(rel_mc_hist)
            relative_energy_loss_rec_histogram.add(rel_rec_hist)
            relative_energy_loss_mc_log_histogram.add(rel_mc_log_hist)
            relative_energy_loss_rec_log_histogram.add(rel_rec_log_hist)
            energy_loss_mc_histogram.add(mc_hist)
            energy_loss_rec_histogram.add(rec_hist)

    results = {}

    relative_energy_loss_mc_histogram = relative_energy_loss_mc_histogram.rebin(energy_binning, relative_energy_binning)
    relative_energy_loss_rec_histogram = relative_energy_loss_rec_histogram.rebin(energy_binning, relative_energy_binning)
    relative_energy_loss_mc_log_histogram = relative_energy_loss_mc_log_histogram.rebin(energy_binning, relative_energy_binning_log)
    relative_energy_loss_rec_log_histogram = relative_energy_loss_rec_log_histogram.rebin(energy_binning, relative_energy_binning_log)
    energy_loss_mc_histogram = energy_loss_mc_histogram.rebin(energy_binning, energy_binning)
    energy_loss_rec_histogram = energy_loss_rec_histogram.rebin(energy_binning, energy_binning)

    relative_energy_loss_mc_histogram.add_to_file(results, "relative_energy_loss_mc_hist")
    relative_energy_loss_rec_histogram.add_to_file(results, "relative_energy_loss_rec_hist")
    relative_energy_loss_mc_log_histogram.add_to_file(results, "relative_energy_loss_mc_log_hist")
    relative_energy_loss_rec_log_histogram.add_to_file(results, "relative_energy_loss_rec_log_hist") 
    energy_loss_mc_histogram.add_to_file(results, "energy_loss_mc_hist") 
    energy_loss_rec_histogram.add_to_file(results, "energy_loss_rec_hist")

    rel_mc_mean, _, _ = hist_mean_and_std(relative_energy_loss_mc_histogram)
    rel_mc_fig = plt.figure(figsize=(8, 4.2))
    rel_mc_plot = rel_mc_fig.subplots(1, 1)
    plot_histogram_2d(rel_mc_plot, relative_energy_loss_mc_histogram, scale=1/relative_energy_loss_mc_histogram.values.sum(axis=1), log=True, show_overflow=False, cmap="jet")
    rel_mc_plot.plot(energy_binning.bin_centers[1:-1], rel_mc_mean, ".", markersize=1, color="black")
    rel_mc_plot.set_xlim(energy_binning.edges[1], energy_binning.edges[-2])
    rel_mc_plot.set_ylim(relative_energy_binning.edges[1], relative_energy_binning.edges[-2])
    set_energy_ticks(rel_mc_plot)
    rel_mc_fig.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
    if args.no_title:
        save_figure(rel_mc_fig, args.plotdir, f"{args.outputprefix}_relative_energy_loss_mc_NoTitle", transparent=args.transparent)
    rel_mc_plot.set_title(args.title)
    save_figure(rel_mc_fig, args.plotdir, f"{args.outputprefix}_relative_energy_loss_mc", transparent=args.transparent)

    rel_rec_mean, _, _ = hist_mean_and_std(relative_energy_loss_rec_histogram)
    rel_rec_fig = plt.figure(figsize=(8, 4.2))
    rel_rec_plot = rel_rec_fig.subplots(1, 1)
    plot_histogram_2d(rel_rec_plot, relative_energy_loss_rec_histogram, scale=1/relative_energy_loss_rec_histogram.values.sum(axis=1), log=True, show_overflow=False, cmap="jet")
    rel_rec_plot.plot(energy_binning.bin_centers[1:-1], rel_rec_mean, ".", markersize=1, color="black")
    rel_rec_plot.set_xlim(energy_binning.edges[1], energy_binning.edges[-2])
    rel_rec_plot.set_ylim(relative_energy_binning.edges[1], relative_energy_binning.edges[-2])
    set_energy_ticks(rel_rec_plot)
    rel_rec_fig.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
    if args.no_title:
        save_figure(rel_rec_fig, args.plotdir, f"{args.outputprefix}_relative_energy_loss_rec_NoTitle", transparent=args.transparent)
    rel_rec_plot.set_title(args.title)
    save_figure(rel_rec_fig, args.plotdir, f"{args.outputprefix}_relative_energy_loss_rec", transparent=args.transparent)

    rel_mc_log_mean, _, rel_mc_log_mean_error = hist_mean_and_std(relative_energy_loss_mc_log_histogram)
    valid = ~(np.isnan(rel_mc_log_mean) | np.isnan(rel_mc_log_mean_error))
    valid = valid & (energy_binning.bin_centers[1:-1] <= 1000) & (energy_binning.bin_centers[1:-1] >= 0.5)
    print(rel_mc_log_mean, flush = True)
    popt, pcov = curve_fit(straight, energy_binning.bin_centers[1:-1][valid], rel_mc_log_mean[valid], sigma = rel_mc_log_mean_error[valid], absolute_sigma = True, maxfev=10000)
    _,_,chi2 = calculate_chisq(rel_mc_log_mean[valid], straight(energy_binning.bin_centers[1:-1][valid], *popt), rel_mc_log_mean_error[valid], len(popt))
    label = f'${{{popt[0]:.3f}}}$\n'+\
            f'$\chi^2: {{{chi2:.3f}}}$'
    rel_mc_log_fig = plt.figure(figsize=(8, 4.2))
    rel_mc_log_plot = rel_mc_log_fig.subplots(1, 1)
    plot_histogram_2d(rel_mc_log_plot, relative_energy_loss_mc_log_histogram, min_value=1e-4, max_value=1e-1, scale=1/relative_energy_loss_mc_log_histogram.values.sum(axis=1), log=True, show_overflow=False, cmap="jet")
    rel_mc_log_plot.errorbar(energy_binning.bin_centers[1:-1], rel_mc_log_mean, yerr=rel_mc_log_mean_error, marker=".", ls='', markersize=1, color="black", zorder = 5, label="mean")
    rel_mc_log_plot.plot(energy_binning.bin_centers[1:-1][valid], straight(energy_binning.bin_centers[1:-1][valid], *popt), color='magenta', lw=1, zorder=3, label = label)
    rel_mc_log_plot.set_xlim(0.15, energy_binning.edges[-2])
    rel_mc_log_plot.set_ylim(1e-4, 1)
    set_energy_ticks(rel_mc_log_plot)
    rel_mc_log_fig.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
    rel_mc_log_plot.legend()
    if args.no_title:
        save_figure(rel_mc_log_fig, args.plotdir, f"{args.outputprefix}_relative_energy_loss_mc_log_NoTitle", transparent=args.transparent)
    rel_mc_log_plot.set_title(args.title)
    save_figure(rel_mc_log_fig, args.plotdir, f"{args.outputprefix}_relative_energy_loss_mc_log", transparent=args.transparent)

    rel_rec_log_mean, _, _ = hist_mean_and_std(relative_energy_loss_rec_log_histogram)
    rel_rec_log_fig = plt.figure(figsize=(8, 4.2))
    rel_rec_log_plot = rel_rec_log_fig.subplots(1, 1)
    plot_histogram_2d(rel_rec_log_plot, relative_energy_loss_rec_log_histogram, scale=1/relative_energy_loss_rec_log_histogram.values.sum(axis=1), log=True, show_overflow=False, cmap="jet")
    rel_rec_log_plot.plot(energy_binning.bin_centers[1:-1], rel_rec_log_mean, ".", markersize=1, color="black")
    rel_rec_log_plot.set_xlim(energy_binning.edges[1], energy_binning.edges[-2])
    rel_rec_log_plot.set_ylim(relative_energy_binning_log.edges[1], relative_energy_binning_log.edges[-2])
    set_energy_ticks(rel_rec_log_plot)
    rel_rec_log_fig.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)    
    if args.no_title:
        save_figure(rel_rec_log_fig, args.plotdir, f"{args.outputprefix}_relative_energy_loss_rec_log_NoTitle", transparent=args.transparent)
    rel_rec_log_plot.set_title(args.title)
    save_figure(rel_rec_log_fig, args.plotdir, f"{args.outputprefix}_relative_energy_loss_rec_log", transparent=args.transparent)


    el_mc_mean, _, el_mc_mean_error = hist_mean_and_std(energy_loss_mc_histogram)
    valid = ~(np.isnan(el_mc_mean) | np.isnan(el_mc_mean_error))
    valid = valid & (energy_binning.bin_centers[1:-1] <= 100) & (energy_binning.bin_centers[1:-1] >= 0.5)
    popt, pcov = curve_fit(power_law, energy_binning.bin_centers[1:-1][valid], el_mc_mean[valid], sigma = el_mc_mean_error[valid], absolute_sigma = True, maxfev=10000)
    _,_,chi2 = calculate_chisq(el_mc_mean[valid], power_law(energy_binning.bin_centers[1:-1][valid], *popt), el_mc_mean_error[valid], len(popt))
    label = f'E<10GeV: ${{{popt[0]:.3f}}}+{{{popt[1]:.3f}}}*E^{{{popt[2]:.3f}}}$\n'+\
            f'E>=10GeV: ${{{popt[3]:.3f}}}+{{{popt[4]:.3f}}}*E^{{{popt[5]:.3f}}}$\n'+\
            f'$\chi^2: {{{chi2:.3f}}}$'
    el_mc_fig = plt.figure(figsize=(8, 4.2))
    el_mc_plot = el_mc_fig.subplots(1, 1)
    plot_histogram_2d(el_mc_plot, energy_loss_mc_histogram, scale=1/energy_loss_mc_histogram.values.sum(axis=1), log=True, show_overflow=False, cmap="jet")
    el_mc_plot.errorbar(energy_binning.bin_centers[1:-1], el_mc_mean, yerr=el_mc_mean_error, marker=".", ls='', markersize=1, color="black", zorder = 5, label="mean")
    el_mc_plot.plot(energy_binning.bin_centers[1:-1][valid], power_law(energy_binning.bin_centers[1:-1][valid], *popt), color='magenta', lw=1, zorder=3, label = label)
    el_mc_plot.set_xlim(energy_binning.edges[1], energy_binning.edges[-2])
    el_mc_plot.set_ylim(energy_binning.edges[1], energy_binning.edges[-2])
    set_energy_ticks(el_mc_plot)
    el_mc_plot.legend()
    el_mc_fig.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
    print(popt)
    if args.no_title:
        save_figure(el_mc_fig, args.plotdir, f"{args.outputprefix}_energy_loss_mc_NoTitle", transparent=args.transparent)
    el_mc_plot.set_title(args.title)
    save_figure(el_mc_fig, args.plotdir, f"{args.outputprefix}_energy_loss_mc", transparent=args.transparent)

    res_fig = plt.figure(figsize=(8, 4.2))
    res_plot = res_fig.subplots(1,1)
    res_plot.errorbar(energy_binning.bin_centers[1:-1][valid], (el_mc_mean[valid] - power_law(energy_binning.bin_centers[1:-1][valid], *popt))/el_mc_mean_error[valid], yerr=np.ones(len(el_mc_mean[valid])), ls="", marker = '.')
    res_plot.axhline(0, color='grey')
    #set_energy_ticks(res_plot)
    res_plot.set_xlabel(r"$E_{MC}$ / GeV")
    res_plot.set_ylabel("Residuum")
    if args.no_title:
        save_figure(res_fig, args.plotdir, f"{args.outputprefix}_energy_loss_mc_residuum_NoTitle", transparent=args.transparent)
    res_plot.set_title(args.title)
    save_figure(res_fig, args.plotdir, f"{args.outputprefix}_energy_loss_mc_residuum", transparent=args.transparent)



    el_rec_mean, _, _ = hist_mean_and_std(energy_loss_rec_histogram)
    el_rec_fig = plt.figure(figsize=(8, 4.2))
    el_rec_plot = el_rec_fig.subplots(1, 1)
    plot_histogram_2d(el_rec_plot, energy_loss_rec_histogram, scale=1/energy_loss_rec_histogram.values.sum(axis=1), log=True, show_overflow=False, cmap="jet")
    el_rec_plot.plot(energy_binning.bin_centers[1:-1], el_rec_mean, ".", markersize=1, color="black")
    el_rec_plot.set_xlim(energy_binning.edges[1], energy_binning.edges[-2])
    el_rec_plot.set_ylim(energy_binning.edges[1], energy_binning.edges[-2])
    set_energy_ticks(el_rec_plot)
    el_rec_fig.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
    if args.no_title:
        save_figure(el_rec_fig, args.plotdir, f"{args.outputprefix}_energy_loss_rec_NoTitle", transparent=args.transparent)
    el_rec_plot.set_title(args.title)
    save_figure(el_rec_fig, args.plotdir, f"{args.outputprefix}_energy_loss_rec", transparent=args.transparent)


    np.savez_compressed(os.path.join(args.resultdir, f"{args.outputprefix}.npz"), **results)

if __name__ == "__main__":
    main()


