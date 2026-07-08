#!/usr/bin/env python3

import os
import multiprocessing as mp
from datetime import datetime
import time

import numpy as np
import awkward as ak
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib as mpl

from tools.binnings import make_lin_binning, make_healpy_binning
from tools.coordinates import convert_galactic_direction_to_ams
from tools.histograms import Histogram, WeightedHistogram, load_histogram
from tools.roottree import read_tree
from tools.utilities import save_figure


def convert_to_ams_direction(latitude, longitude, time, iss_x, iss_y, iss_z):
    iss_x = ak.to_numpy(iss_parameters[:,0])
    iss_y = ak.to_numpy(iss_parameters[:,1])
    iss_z = ak.to_numpy(iss_parameters[:,2])
    iss_vx = ak.to_numpy(iss_parameters[:,3])
    iss_vy = ak.to_numpy(iss_parameters[:,4])
    iss_vz = ak.to_numpy(iss_parameters[:,5])
    iss_yaw = ak.to_numpy(iss_parameters[:,6])
    iss_pitch = ak.to_numpy(iss_parameters[:,7])
    iss_roll = ak.to_numpy(iss_parameters[:,8])

    dx, dy, dz = convert_galactic_direction_to_ams(latitude[:,None], longitude[:,None], iss_x[None,:], iss_y[None,:], iss_z[None,:], iss_vx[None,:], iss_vy[None,:], iss_vz[None,:], iss_yaw[None,:], iss_pitch[None,:], iss_roll[None,:], time[None,:])
    return dx, dy, dz


def handle_file(arg):
    filename, treename, chunk_size, rank, nranks, kwargs = arg

    time_range = kwargs.get("time_range", None)
    time_min, time_max = None, None
    if time_range is not None:
        time_min, time_max = time_range
    nside = kwargs["nside"]

    verbose = kwargs.get("verbose", True)

    cos_theta_binning = kwargs["cos_theta_binning"]

    altitude_time_hist = WeightedHistogram(make_healpy_binning(nside=nside), cos_theta_binning, labels=("Healpy Index", "cos(theta)"), dtype=np.float32)

    pixel_indices = np.arange(hp.nside2npix(nside))
    pixel_longitude, pixel_latitude = hp.pix2ang(nside, pixel_indices, lonlat=True)
    first_time = 1000000000000
    last_time = 0

    for entries in read_tree(filename, treename, chunk_size=chunk_size, rank=rank, nranks=nranks, verbose=verbose):
        entry_time = ak.to_numpy(entries.UTCTime)
        first_time = min(first_time, np.min(entry_time))
        last_time = max(last_time, np.max(entry_time))
        entry_iss_x = ak.to_numpy(entries.ISSParameters[:,0])
        entry_iss_y = ak.to_numpy(entries.ISSParameters[:,1])
        entry_iss_z = ak.to_numpy(entries.ISSParameters[:,2])
        entry_iss_vx = ak.to_numpy(entries.ISSParameters[:,3])
        entry_iss_vy = ak.to_numpy(entries.ISSParameters[:,4])
        entry_iss_vz = ak.to_numpy(entries.ISSParameters[:,5])
        entry_iss_yaw = ak.to_numpy(entries.ISSParameters[:,6])
        entry_iss_pitch = ak.to_numpy(entries.ISSParameters[:,7])
        entry_iss_roll = ak.to_numpy(entries.ISSParameters[:,8])
        entry_livetime = ak.to_numpy(entries.TriggerLiveTime)

        for (t, livetime, iss_x, iss_y, iss_z, iss_vx, iss_vy, iss_vz, iss_yaw, iss_pitch, iss_roll) in zip(
                entry_time, entry_livetime, entry_iss_x, entry_iss_y, entry_iss_z, entry_iss_vx, entry_iss_vy, entry_iss_vz, entry_iss_yaw, entry_iss_pitch, entry_iss_roll):

            _, _, pixel_dz = convert_galactic_direction_to_ams(pixel_latitude, pixel_longitude, [iss_x], [iss_y], [iss_z], [iss_vx], [iss_vy], [iss_vz], [iss_yaw], [iss_pitch], [iss_roll], [t])
            altitude_time_hist.fill(pixel_indices, pixel_dz, weights=np.ones_like(pixel_dz) * livetime)
    
    return altitude_time_hist, first_time, last_time


def make_args(filename, treename, chunk_size, nranks, parallel, **kwargs):
    parallel_index, parallel_total = parallel
    for rank in range(parallel_index * nranks, (parallel_index + 1) * nranks):
        yield (filename, treename, chunk_size, rank, nranks * parallel_total, kwargs)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("iss_trees", nargs="+", help="Path to ISS coordinate trees to read.")
    parser.add_argument("--nside", type=int, default=256, help="Number of HEALPix sides to use in the skymap binning.")
    parser.add_argument("--time-range", type=int, nargs=2, help="Beginning and end of time range to integrate.")
    parser.add_argument("--treename", default="IssTree", help="Name of the ROOT tree in the files.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--parallel", type=int, nargs=2, default=(0, 1), help="Index and number of jobs to use in parallel.")
    parser.add_argument("--chunk-size", type=int, default=300, help="Number of entries to read in each step.")
    parser.add_argument("--verbose", action="store_true", help="Write verbose progress information.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--outputprefix", default="Altitude", help="Prefix for plots and result files.")

    args = parser.parse_args()

    parallel_index, parallel_total = args.parallel

    os.makedirs(args.resultdir, exist_ok=True)
    os.makedirs(args.plotdir, exist_ok=True)

    cos_theta_binning = make_lin_binning(0.7, 1, 150)
    healpy_binning = make_healpy_binning(nside=args.nside)
    altitude_time_hist = WeightedHistogram(healpy_binning, cos_theta_binning, labels=("Healpy Index", "cos(theta)"))

    pool_args = make_args(args.iss_trees, args.treename, args.chunk_size, args.nprocesses, args.parallel, time_range=args.time_range, verbose=args.verbose, cos_theta_binning=cos_theta_binning, nside=args.nside)

    start_time = 1000000000000
    end_time = 0

    with mp.get_context("fork" if args.verbose else "spawn").Pool(args.nprocesses) as pool:
        for hist, tmin, tmax in pool.imap_unordered(handle_file, pool_args):
            altitude_time_hist.add(hist)
            start_time = min(start_time, tmin)
            end_time = max(end_time, tmax)

    results = {}
    results["start_time"] = start_time
    results["end_time"] = end_time
    altitude_time_hist.add_to_file(results, "altitude_time")
    np.savez_compressed(os.path.join(args.resultdir, f"{args.outputprefix}_altitude_{parallel_index}.npz"), **results)



if __name__ == "__main__":
    main()
