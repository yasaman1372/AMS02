#!/usr/bin/env python3

import os
import multiprocessing as mp
from datetime import datetime, timezone
import time

import numpy as np
import awkward as ak
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib as mpl

from tools.binnings import make_lin_binning, make_healpy_binning
from tools.histograms import Histogram, WeightedHistogram, load_histogram
from tools.roottree import read_tree
from tools.utilities import save_figure


def parse_datetime(datetime_str):
    if datetime_str.isnumeric():
        return float(datetime_str)
    return datetime.fromisoformat(datetime_str).replace(tzinfo=timezone.utc).timestamp()


def load_altitude(arg):
    filename, kwargs = arg

    time_range = kwargs["time_range"]
    if time_range is not None:
        min_time, max_time = time_range

    with np.load(filename) as np_file:
        start_time = np_file["start_time"].item()
        end_time = np_file["end_time"].item()
        if start_time >= 1000000000000 or end_time <= 0:
            return None
        print(datetime.fromtimestamp(start_time, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), datetime.fromtimestamp(end_time, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), flush=True)
        if time_range is not None:
            if start_time > max_time or end_time < min_time:
                return None
        altitude_time_hist = load_histogram(np_file, "altitude_time")

    return altitude_time_hist, start_time, end_time


def make_args(filenames, **kwargs):
    for filename in filenames:
        yield (filename, kwargs)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--altitude-histograms", required=True, nargs="+", help="Path to NPZ files containing the altitude histograms.")
    parser.add_argument("--resultdir", default="results", help="Directory to store the merged altitude histogram in.")
    parser.add_argument("--outputprefix", default="Altitude", help="Prefix for result files.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--time-range", type=parse_datetime, nargs=2, help="Limit the time range to read in.")

    args = parser.parse_args()

    os.makedirs(args.resultdir, exist_ok=True)

    #global_altitude_time_hist = WeightedHistogram(healpy_binning, cos_theta_binning, labels=("Healpy Index", "cos(theta)"))

    global_altitude_time_hist = None
    global_start_time = 1000000000000
    global_end_time = 0

    pool_args = make_args(args.altitude_histograms, time_range=args.time_range)
    with mp.Pool(args.nprocesses) as pool:
        for return_value in pool.imap_unordered(load_altitude, pool_args):
            print(return_value)
            if return_value is not None:
                altitude_hist, start_time, end_time = return_value
                if global_altitude_time_hist is not None:
                    global_altitude_time_hist.add(altitude_hist)
                    global_start_time = min(global_start_time, start_time)
                    global_end_time = max(global_end_time, end_time)
                else:
                    global_altitude_time_hist = altitude_hist
                    global_start_time = start_time
                    global_end_time = end_time

    results = {}
    results["start_time"] = global_start_time
    results["end_time"] = global_end_time
    global_altitude_time_hist.add_to_file(results, "altitude_time")
    np.savez(os.path.join(args.resultdir, f"{args.outputprefix}.npz"), **results)


if __name__ == "__main__":
    main()
