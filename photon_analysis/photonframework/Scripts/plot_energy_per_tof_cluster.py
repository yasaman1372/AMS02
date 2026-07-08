#!/usr/bin/env python3

import multiprocessing as mp
import os
import pickle

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle

from tools.binnings import make_lin_binning, make_energy_binning_from_config, make_mc_dataset_edge_binning, combine_binnings, reduce_bins, make_flux_energy_binning
from tools.config import get_config
from tools.constants import BastianPlots
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d
from tools.roottree import read_tree
from tools.utilities import load_mc_trigger_count, save_figure, set_energy_ticks, plot_2d
from tools.statistics import hist_mean_and_std

plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.minor.visible'] = True

BRANCHES = ["TofEnergyPerClusterInLayer0", "TofEnergyPerClusterInLayer1", "TofEnergieOneClusterInLayer2"]

def handle_file(arg):
    filenames, treename, chunk_size, rank, nranks, kwargs = arg

    #energy_binning = kwargs["energy_binning"]
    tof_energy_binning = kwargs["tof_energy_binning"]

    histogram = Histogram(tof_energy_binning, tof_energy_binning)

    for events in read_tree(filenames, treename, rank=rank, nranks=nranks, chunk_size=chunk_size, branches=BRANCHES, cache_file=False):
        histogram.fill(events.TofEnergyPerClusterInLayer0, events.TofEnergyPerClusterInLayer1)

    return histogram


def make_args(filename, treename, chunk_size, nranks, **kwargs):
    for rank in range(nranks):
        yield (filename, treename, chunk_size, rank, nranks, kwargs)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs=2, required=True, help="Path to config file and working directory.")
    parser.add_argument("--tree", nargs="+", required=True, help="MC Gamma Tree to read.")
    parser.add_argument("--treename", default="GammaTree", help="Name of the tree in the ROOT file to read.")
    parser.add_argument("--outputprefix", default="CutPlot", help="Prefix for plots and result files.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Number of events to read each step.")
    parser.add_argument("--title", default="Cut Plot", help="Title for plots.")
    parser.add_argument("--no-title", action='store_true', help='Use this to prevent titles from being generated.')
    parser.add_argument("--transparent", action="store_true", help='set this to make the background of the png transparent')
    parser.add_argument('--has-p-e', action="store_true", help = "set if p and/or e can be seen")
    parser.add_argument("--save-pdf", action="store_true", help= "set this to save the plots as pdf aswell.")

    args = parser.parse_args()
    config_filename, datadir = args.config
    config = get_config(config_filename)

    os.makedirs(args.resultdir, exist_ok=True)
    os.makedirs(args.plotdir, exist_ok=True)

    energy_binning = make_energy_binning_from_config(config)
    tof_energy_binning = make_lin_binning(0,10,100)

    event_histogram = Histogram(tof_energy_binning, tof_energy_binning, labels=("Energy / MeV", "Energy / MeV"))

    with mp.Pool(args.nprocesses) as pool:
        pool_args = make_args(args.tree, args.treename, chunk_size=args.chunk_size, nranks=args.nprocesses, tof_energy_binning = tof_energy_binning)
        for hist in pool.imap_unordered(handle_file, pool_args):
            event_histogram.add(hist)

    fig = plt.figure(figsize=(8, 4.2))
    plot = fig.subplots(1, 1)
    plot_histogram_2d(plot, event_histogram, min_value=None, max_value=None, label="Events", cmap = 'jet', log=True, show_overflow=False)
    plot.set_xlabel("Deposited Energy TOF Layer 0 / MeV")
    plot.set_ylabel("Deposited Energy TOF Layer 1 / MeV")
    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
    photons = Rectangle((3.2,3.2), 7-3.2, 7-3.2, color = "black", fill = False)
    plot.add_patch(photons)
    plot.text(5.1, 5.1, s = r'$\gamma$', c = 'black')
    if args.has_p_e:
        plot.text(2.5, 2, s = r"$p^+, e^{-/+}$")
    if args.no_title:
        save_figure(fig, args.plotdir, f"{args.outputprefix}_Energy_in_TOF_NoTitle", transparent=args.transparent, save_pdf=args.save_pdf)
    plot.set_title(f"{args.title}, Deposited Energy in Upper TOF Layers")
    save_figure(fig, args.plotdir, f"{args.outputprefix}_Energy_in_TOF", transparent=args.transparent, save_pdf=args.save_pdf)
    
    
    fig = plt.figure(figsize=(5.5, 4.2))
    plot = fig.subplots(1, 1)
    plot_histogram_2d(plot, event_histogram, min_value=None, max_value=None, label="Events", cmap = 'jet', log=True, show_overflow=False)
    plot.set_xlabel("Deposited Energy TOF Layer 0 / MeV")
    plot.set_ylabel("Deposited Energy TOF Layer 1 / MeV")
    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
    photons = Rectangle((3.2,3.2), 7-3.2, 7-3.2, color = "black", fill = False)
    plot.add_patch(photons)
    plot.text(5.1, 5.1, s = r'$\gamma$', c = 'black')
    if args.has_p_e:
        plot.text(2.5, 2, s = r"$p^+, e^{-/+}$")
    plot.set_box_aspect(1)
    if args.no_title:
        save_figure(fig, args.plotdir, f"{args.outputprefix}_Energy_in_TOF_square_NoTitle", transparent=args.transparent, save_pdf=args.save_pdf)
    plot.set_title(f"{args.title}, Deposited Energy in Upper TOF Layers")
    save_figure(fig, args.plotdir, f"{args.outputprefix}_Energy_in_TOF_square", transparent=args.transparent, save_pdf=args.save_pdf)

    results = {}
    event_histogram.add_to_file(results, "energy_upper_tof_clusters")
    np.savez_compressed(os.path.join(args.resultdir, f"{args.outputprefix}.npz"), **results)

if __name__ == "__main__":
    main()



