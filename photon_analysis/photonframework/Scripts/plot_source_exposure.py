#!/usr/bin/env python3

import os
import multiprocessing as mp
from datetime import datetime, timezone
import time
import re

import numpy as np
import awkward as ak
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib as mpl

from tools.binnings import make_day_binning, make_lin_binning, make_healpy_binning
from tools.histograms import Histogram, WeightedHistogram, load_histogram, plot_histogram_1d, plot_histogram_2d
from tools.roottree import read_tree
from tools.utilities import glob_re, set_energy_ticks, save_figure

from plot_time_map import make_time_ticks



def parse_datetime(datetime_str):
    if datetime_str.isnumeric():
        return float(datetime_str)
    return datetime.fromisoformat(datetime_str).replace(tzinfo=timezone.utc)


def load_altitude(arg):
    filename, effective_area_2d, nside, outputprefix, plotdir, kwargs = arg

    energy_binning, cos_theta_binning = effective_area_2d.binnings
    healpy_binning = make_healpy_binning(nside=nside)
    healpy_index = kwargs["healpy_bin_index"]

    time_range = kwargs["time_range"]
    if time_range is not None:
        min_time, max_time = map(lambda t: t.timestamp(), time_range)


    with np.load(filename) as np_file:
        start_time = np_file["start_time"].item()
        end_time = np_file["end_time"].item()
        if start_time >= 1000000000000 or end_time <= 0:
            return None
        if time_range is not None:
            if start_time > max_time or end_time < min_time:
                return None
        altitude_time_hist = load_histogram(np_file, "altitude_time")
        exposure = (altitude_time_hist.values[healpy_index,None,:] * effective_area_2d.values[:,:]).sum(axis=1)
        return (start_time, end_time, exposure)


def make_args(filenames, effective_area, nside, outputprefix, plotdir, **kwargs):
    for filename in filenames:
        yield (filename, effective_area, nside, outputprefix, plotdir, kwargs)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--altitude-histograms", required=True, nargs="+", help="Path to NPZ files containing the altitude histograms.")
    parser.add_argument("--effective-area", required=True, help="Path to NPZ file containing the effective area to be integrated.")
    parser.add_argument("--nside", type=int, default=256, help="Number of HEALPix sides to use in the skymap binning.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--outputprefix", default="ExposurePerTime", help="Prefix for plots and result files.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--time-range", type=parse_datetime, nargs=2, required=True, help="Time range to read and plot.")
    parser.add_argument("--source-position", type=float, nargs=2, metavar=("longitude", "latitude"), help="Direction to calculate the exposure series for, in degrees.")
    parser.add_argument("--title", help="Title for plots.")
    parser.add_argument("--load-exposure", help="Load exposure histograms instead of calculating them.")

    args = parser.parse_args()

    os.makedirs(args.resultdir, exist_ok=True)
    os.makedirs(args.plotdir, exist_ok=True)

    if args.load_exposure is not None:
        with np.load(args.load_exposure) as exposure_file:
            exposure_hist = load_histogram(exposure_file, "exposure")
    else:
        with np.load(args.effective_area) as effective_area_file:
            effective_area_3d = load_histogram(effective_area_file, "effective_area_3d")
            effective_area_2d = load_histogram(effective_area_file, "effective_area_2d")

        time_binning = make_day_binning(*args.time_range)
        energy_binning, _ = effective_area_2d.binnings
        exposure_hist = WeightedHistogram(energy_binning, time_binning, labels=("Energy / GeV", "Time"))

        healpy_bin_index = hp.ang2pix(args.nside, *args.source_position, lonlat=True)
        print(f"Looking for Healpy bin {healpy_bin_index}")

        altitude_histograms = [filename for pattern in args.altitude_histograms for filename in glob_re(pattern)]

        pool_args = make_args(altitude_histograms, effective_area_2d, args.nside, args.outputprefix, args.plotdir, time_range=args.time_range, healpy_bin_index=healpy_bin_index)
        progress = 0
        total = len(altitude_histograms)
        with mp.Pool(args.nprocesses) as pool:
            for return_value in pool.imap_unordered(load_altitude, pool_args):
                progress += 1
                if return_value is not None:
                    start_time, end_time, exposure_per_energy = return_value
                    mean_time = (start_time + end_time) / 2
                    exposure_hist.fill(energy_binning.bin_centers, np.full_like(exposure_per_energy, mean_time), weights=exposure_per_energy)
                print(f"{progress:>8}/{total:>8}", end="\r", flush=True)

        results = {}
        exposure_hist.add_to_file(results, "exposure")
        np.savez(os.path.join(args.resultdir, f"{args.outputprefix}_exposure_per_time.npz"), **results)

    start_timestamp, end_timestamp = map(lambda t: t.timestamp(), args.time_range)
    major_time_ticks, minor_time_ticks, major_time_labels = make_time_ticks(start_timestamp, end_timestamp)
    if len(major_time_labels) > 10:
        step = int(np.ceil(len(major_time_labels) / 10))
        for index in range(len(major_time_labels)):
            if index % step != 0:
                major_time_labels[index] = ""

    figure = plt.figure(figsize=(8, 4.2))
    plot = figure.subplots(1, 1)
    if args.title is not None:
        plot.set_title(args.title)
    plot_histogram_2d(plot, exposure_hist, label="Exposure / cm²s")
    plot.set_yticks(major_time_ticks, labels=major_time_labels)
    plot.set_yticks(minor_time_ticks, minor=True)
    set_energy_ticks(plot)
    save_figure(figure, args.plotdir, f"{args.outputprefix}_2d")

    for index, (exposure_1d, min_energy, max_energy) in enumerate(exposure_hist.project_all()):
        figure = plt.figure(figsize=(8, 4.2))
        plot = figure.subplots(1, 1)
        plot_histogram_1d(plot, exposure_1d)
        if args.title is not None:
            plot.set_title(f"{args.title}, ${min_energy:.2f} \\leq E/GeV < {max_energy:.2f}$")
        else:
            plot.set_title(f"${min_energy:.2f} \\leq E/GeV < {max_energy:.2f}$")
        plot.set_xlim(*map(lambda ts: np.datetime64(int(ts), "s"), (start_timestamp, end_timestamp)))
        plot.set_ylim(bottom=0)
        plot.set_xticks(major_time_ticks, labels=major_time_labels)
        plot.set_xticks(minor_time_ticks, minor=True)
        plot.set_ylabel("Exposure / cm²s")
        save_figure(figure, args.plotdir, f"{args.outputprefix}_1d_{index}")


if __name__ == "__main__":
    main()
