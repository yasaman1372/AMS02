#!/usr/bin/env python3

import multiprocessing as mp
import os
import pickle

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from tools.binnings import make_lin_binning, make_energy_binning_from_config, combine_binnings, reduce_bins, make_flux_energy_binning
from tools.config import get_config
from tools.constants import BastianPlots
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d
from tools.roottree import read_tree
from tools.utilities import load_mc_trigger_count, save_figure, set_energy_ticks, plot_2d
from tools.statistics import hist_mean_and_std

plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.minor.visible'] = True

BRANCHES = ["McEnergy", "Energy", "McPrimaryFinalX", "McPrimaryFinalY", "McPrimaryFinalZ", "TrkTrackBestPairMinDistanceCoordX", "TrkTrackBestPairMinDistanceCoordY", "TrkTrackBestPairMinDistanceCoordZ"]

def handle_file(arg):
    filenames, treename, chunk_size, rank, nranks, kwargs = arg

    #energy_binning = kwargs["energy_binning"]
    z_binning = kwargs["z_binning"]
    x_binning = kwargs["x_binning"]
    y_binning = kwargs["y_binning"]

    mc_histogram = Histogram(x_binning, y_binning, z_binning)
    histogram = Histogram(x_binning, y_binning, z_binning)

    for events in read_tree(filenames, treename, rank=rank, nranks=nranks, chunk_size=chunk_size, branches=BRANCHES, cache_file=False):
        mc_histogram.fill(events.McPrimaryFinalX, events.McPrimaryFinalY, events.McPrimaryFinalZ)
        histogram.fill(events.TrkTrackBestPairMinDistanceCoordX, events.TrkTrackBestPairMinDistanceCoordY, events.TrkTrackBestPairMinDistanceCoordZ)

    return mc_histogram, histogram

def make_args(filename, treename, chunk_size, nranks, **kwargs):
    for rank in range(nranks):
        yield (filename, treename, chunk_size, rank, nranks, kwargs)



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs=2, required=True, help="Path to config file and working directory.")
    parser.add_argument("--tree", nargs="+", required=True, help="MC Gamma Tree to read.")
    parser.add_argument("--treename", default="GammaTree", help="Name of the tree in the ROOT file to read.")
    parser.add_argument("--outputprefix", default="Conversion_Position", help="Prefix for plots and result files.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Number of events to read each step.")
    parser.add_argument("--title", default="Conversion Position", help="Title for plots.")
    parser.add_argument("--no-title", action='store_true', help='Use this to prevent titles from being generated.')
    parser.add_argument("--transparent", action="store_true", help='set this to make the background of the png transparent')
    parser.add_argument("--save-pdf", action="store_true", help= "set this to save the plots as pdf aswell.")
    parser.add_argument("--no-cuts", action="store_true", help = "set for no cut plot for more labels")

    args = parser.parse_args()
    config_filename, datadir = args.config
    config = get_config(config_filename)
    print(args.save_pdf)
    os.makedirs(args.resultdir, exist_ok=True)
    os.makedirs(args.plotdir, exist_ok=True)

    energy_binning = make_energy_binning_from_config(config)
    z_binning = make_lin_binning(50,175,250)
    x_binning = make_lin_binning(-120,120, 480)
    y_binning = make_lin_binning(-120,120, 480)

    event_histogram = Histogram(x_binning, y_binning, z_binning, labels=("x / cm", "y / cm", "z / cm"))
    mc_histogram = Histogram(x_binning, y_binning, z_binning, labels=("x / cm", "y / cm", "z / cm"))

    with mp.Pool(args.nprocesses) as pool:
        pool_args = make_args(args.tree, args.treename, chunk_size=args.chunk_size, nranks=args.nprocesses, x_binning=x_binning, y_binning=y_binning, z_binning=z_binning)
        for mc_hist, hist in pool.imap_unordered(handle_file, pool_args):
            mc_histogram.add(mc_hist)
            event_histogram.add(hist)

    mc_y_z_histogram = mc_histogram.project_axis(axis = 0)
    y_z_histogram = event_histogram.project_axis(axis = 0)

    mc_x_z_histogram = mc_histogram.project_axis(axis = 1)
    x_z_histogram = event_histogram.project_axis(axis = 1)

    mc_x_y_histogram = mc_histogram.project_axis(axis = 2)
    x_y_histogram = event_histogram.project_axis(axis = 2)

    fig_y_z_mc = plt.figure(figsize=(5.5, 4.2))
    plot_y_z_mc = fig_y_z_mc.subplots(1, 1)
    plot_histogram_2d(plot_y_z_mc, mc_y_z_histogram, min_value=None, max_value=None, label="Events", cmap = 'jet', log=True, show_overflow=False)
    plot_y_z_mc.set_xlabel("y / cm")
    plot_y_z_mc.set_ylabel("z / cm")
    plot_y_z_mc.set_box_aspect(1)
    fig_y_z_mc.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
    plot_y_z_mc.text(0, 68, s = "Upper TOF\nAluminum Honeycomb", color = 'k', ha = 'center', size = 'x-small', weight = 'bold')
    plot_y_z_mc.text(50, 90, s = "TRD Bulkhead", color = 'k', ha = 'center', size = 'x-small', weight = 'bold')
    plot_y_z_mc.text(0, 80, s = "TRD Lower Honeycomb", color = 'k', ha = 'center', size = 'x-small', weight = 'bold')
    plot_y_z_mc.text(0, 105, s = "Lower TRD", color = 'k', ha = 'center', size = 'x-small', weight = 'bold')
    plot_y_z_mc.text(60, 60, s = "Upper Tof Layer 1", color = 'k', ha = 'center', size = 'x-small', weight = 'bold')
    if args.no_cuts:
        plot_y_z_mc.text(0, 167.5, s = "Tracker Plane 1 Support", color = 'k', ha = 'center', size = 'x-small', weight = 'bold')
        plot_y_z_mc.text(0, 160.5, s = "Tracker Plane 1", color = 'k', ha = 'center', size = 'x-small', weight = 'bold')
        plot_y_z_mc.text(0, 147, s = "TRD Upper Honeycomb", color = 'k', ha = 'center', size = 'x-small', weight = 'bold')
    if args.no_title:
        save_figure(fig_y_z_mc, args.plotdir, f"{args.outputprefix}_mc_y_z_NoTitle", transparent=args.transparent, save_pdf=args.save_pdf)
    plot_y_z_mc.set_title(f"{args.title}, MC y/z")
    save_figure(fig_y_z_mc, args.plotdir, f"{args.outputprefix}_mc_y_z", transparent=args.transparent, save_pdf=args.save_pdf)

    fig_y_z = plt.figure(figsize=(5.5, 4.2))
    plot_y_z = fig_y_z.subplots(1, 1)
    plot_histogram_2d(plot_y_z, y_z_histogram, min_value=None, max_value=None, label="Events", cmap = 'jet', log=True, show_overflow=False)
    plot_y_z.set_xlabel("y / cm")
    plot_y_z.set_ylabel("z / cm")
    plot_y_z.set_box_aspect(1)
    fig_y_z.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
    if args.no_title:
        save_figure(fig_y_z, args.plotdir, f"{args.outputprefix}_y_z_NoTitle", transparent=args.transparent, save_pdf=args.save_pdf)
    plot_y_z.set_title(f"{args.title}, y/z")
    save_figure(fig_y_z, args.plotdir, f"{args.outputprefix}_y_z", transparent=args.transparent, save_pdf=args.save_pdf)


    fig_x_z_mc = plt.figure(figsize=(5.5, 4.2))
    plot_x_z_mc = fig_x_z_mc.subplots(1, 1)
    plot_histogram_2d(plot_x_z_mc, mc_x_z_histogram, min_value=None, max_value=None, label="Events", cmap = 'jet', log=True, show_overflow=False)
    plot_x_z_mc.set_xlabel("x / cm")
    plot_x_z_mc.set_ylabel("z / cm")
    plot_x_z_mc.set_box_aspect(1)
    fig_x_z_mc.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
    if args.no_title:
        save_figure(fig_x_z_mc, args.plotdir, f"{args.outputprefix}_mc_x_z_NoTitle", transparent=args.transparent, save_pdf = args.save_pdf)
    plot_x_z_mc.set_title(f"{args.title}, MC x/z")
    save_figure(fig_x_z_mc, args.plotdir, f"{args.outputprefix}_mc_x_z", transparent=args.transparent, save_pdf = args.save_pdf)

    fig_x_z = plt.figure(figsize=(5.5, 4.2))
    plot_x_z = fig_x_z.subplots(1, 1)
    plot_histogram_2d(plot_x_z, x_z_histogram, min_value=None, max_value=None, label="Events", cmap = 'jet', log=True, show_overflow=False)
    plot_x_z.set_xlabel("x / cm")
    plot_x_z.set_ylabel("z / cm")
    plot_x_z.set_box_aspect(1)
    fig_x_z.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
    if args.no_title:
        save_figure(fig_x_z, args.plotdir, f"{args.outputprefix}_x_z_NoTitle", transparent=args.transparent, save_pdf = args.save_pdf)
    plot_x_z.set_title(f"{args.title}, x/z")
    save_figure(fig_x_z, args.plotdir, f"{args.outputprefix}_x_z", transparent=args.transparent, save_pdf = args.save_pdf)


    fig_x_y_mc = plt.figure(figsize=(5.5, 4.2))
    plot_x_y_mc = fig_x_y_mc.subplots(1, 1)
    plot_histogram_2d(plot_x_y_mc, mc_x_y_histogram, min_value=None, max_value=None, label="Events", cmap = 'jet', log=True, show_overflow=False)
    plot_x_y_mc.set_xlabel("x / cm")
    plot_x_y_mc.set_ylabel("y / cm")
    plot_x_y_mc.set_box_aspect(1)
    fig_x_y_mc.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
    if args.no_title:
        save_figure(fig_x_y_mc, args.plotdir, f"{args.outputprefix}_mc_x_y_NoTitle", transparent=args.transparent, save_pdf = args.save_pdf)
    plot_x_y_mc.set_title(f"{args.title}, MC x/y")
    save_figure(fig_x_y_mc, args.plotdir, f"{args.outputprefix}_mc_x_y", transparent=args.transparent, save_pdf = args.save_pdf)

    fig_x_y = plt.figure(figsize=(5.5, 4.2))
    plot_x_y = fig_x_y.subplots(1, 1)
    plot_histogram_2d(plot_x_y, x_y_histogram, min_value=None, max_value=None, label="Events", cmap = 'jet', log=True, show_overflow=False)
    plot_x_y.set_xlabel("x / cm")
    plot_x_y.set_ylabel("y / cm")
    plot_x_y.set_box_aspect(1)
    fig_x_y.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
    if args.no_title:
        save_figure(fig_x_y, args.plotdir, f"{args.outputprefix}_x_y_NoTitle", transparent=args.transparent, save_pdf = args.save_pdf)
    plot_x_y.set_title(f"{args.title}, x/y")
    save_figure(fig_x_y, args.plotdir, f"{args.outputprefix}_x_y", transparent=args.transparent, save_pdf = args.save_pdf)

    results = {}
    event_histogram.add_to_file(results, 'conversion_possition')
    mc_histogram.add_to_file(results, 'mc_conversion_possition')
    np.savez_compressed(os.path.join(args.resultdir, f"{args.outputprefix}.npz"), **results)

if __name__ == "__main__":
    main()
