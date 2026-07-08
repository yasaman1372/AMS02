#!/usr/bin/env python3

import os

import numpy as np
import matplotlib.pyplot as plt

from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d, load_histograms_from_files
from tools.utilities import save_figure


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", nargs="+")
    parser.add_argument("--plotdir", default="plots")
    parser.add_argument("--outputprefix", default="TrackerVertices")
    parser.add_argument("--title", default="ISS Photon Vertices")
    parser.add_argument("--zmin", type=float, default=0)
    parser.add_argument("--log", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.plotdir, exist_ok=True)

    histograms = load_histograms_from_files(args.input_files)

    xz_hist = histograms["pair_hist_TrkTrackBestPairMinDistanceCoordX_TrkTrackBestPairMinDistanceCoordZ"]
    yz_hist = histograms["pair_hist_TrkTrackBestPairMinDistanceCoordY_TrkTrackBestPairMinDistanceCoordZ"]
    xy_hist = histograms["pair_hist_TrkTrackBestPairMinDistanceCoordX_TrkTrackBestPairMinDistanceCoordY"]


    xz_figure = plt.figure(figsize=(8, 4.2))
    xz_plot = xz_figure.subplots(1, 1)
    xz_plot.set_title(args.title)
    if args.log:
        plot_histogram_2d(xz_plot, xz_hist, log=True, min_value=1e3)
    else:
        plot_histogram_2d(xz_plot, xz_hist)
    xz_plot.set_xlabel("X / cm")
    xz_plot.set_ylabel("Z / cm")
    xz_plot.set_xlim(-100, 100)
    xz_plot.set_ylim(args.zmin, 150)
    save_figure(xz_figure, args.plotdir, f"{args.outputprefix}_xz")

    yz_figure = plt.figure(figsize=(8, 4.2))
    yz_plot = yz_figure.subplots(1, 1)
    yz_plot.set_title(args.title)
    if args.log:
        plot_histogram_2d(yz_plot, yz_hist, log=True, min_value=1e3)
    else:
        plot_histogram_2d(yz_plot, yz_hist)
    yz_plot.set_xlabel("Y / cm")
    yz_plot.set_ylabel("Z / cm")
    yz_plot.set_xlim(-100, 100)
    yz_plot.set_ylim(args.zmin, 150)
    save_figure(yz_figure, args.plotdir, f"{args.outputprefix}_yz")

    xy_figure = plt.figure(figsize=(8, 4.2))
    xy_plot = xy_figure.subplots(1, 1)
    xy_plot.set_title(args.title)
    if args.log:
        plot_histogram_2d(xy_plot, xy_hist, log=True, min_value=1e3)
    else:
        plot_histogram_2d(xy_plot, xy_hist)
    xy_plot.set_xlabel("X / cm")
    xy_plot.set_ylabel("Y / cm")
    xy_plot.set_xlim(-100, 100)
    xy_plot.set_ylim(-100, 100)
    save_figure(xy_figure, args.plotdir, f"{args.outputprefix}_xy")

if __name__ == "__main__":
    main()
