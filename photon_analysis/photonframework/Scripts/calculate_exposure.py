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
from tools.histograms import Histogram, WeightedHistogram, load_histogram, plot_histogram_1d, plot_histogram_2d
from tools.roottree import read_tree

from calculate_exposure_map import calculate_exposure, plot_exposures

BRANCHES = ["ISSTerrestrialLongitude", "ISSTerrestrialLatitude", "AMSGalacticLongitude", "AMSGalacticLatitude", "TriggerLiveTime", "UTCTime"]


def parse_datetime(datetime_str):
    if datetime_str.isnumeric():
        return float(datetime_str)
    return datetime.fromisoformat(datetime_str).replace(tzinfo=timezone.utc).timestamp()


def handle_file(arg):
    filenames, rank, nranks, kwargs = arg

    nside = kwargs["nside"]
    npix = hp.nside2npix(nside)
    chunk_size = kwargs["chunk_size"]

    altitude_binning = kwargs["altitude_binning"]
    healpy_binning = kwargs["healpy_binning"]
    min_altitude = kwargs["min_altitude"]

    pileup_efficiency = kwargs["pileup_efficiency"]
    pileup_efficiency_nside = kwargs["pileup_efficiency_nside"]

    time_range = kwargs["time_range"]

    altitude_histogram = WeightedHistogram(healpy_binning, altitude_binning, labels=("Healpy Index", "cos(theta)"))
    min_time, max_time = None, None

    npix_step_size = 10000

    # shape npix
    ref_indices = np.arange(hp.nside2npix(nside), dtype=np.uint32)
    ref_x, ref_y, ref_z = map(lambda a: a.astype(np.float32), hp.pix2vec(nside, ref_indices))

    for seconds in read_tree(filenames, "ExposureTree", rank=rank, nranks=nranks, branches=BRANCHES, chunk_size=chunk_size):
        if time_range is not None:
            seconds = seconds[(seconds.UTCTime >= time_range[0]) & (seconds.UTCTime <= time_range[1])]
            if len(seconds) == 0:
                continue

        if min_time is None:
            min_time = np.min(seconds.UTCTime)
            max_time = np.max(seconds.UTCTime)
        else:
            min_time = min(min_time, np.min(seconds.UTCTime))
            max_time = max(max_time, np.max(seconds.UTCTime))

        seconds_weight = ak.to_numpy(seconds.TriggerLiveTime)

        if pileup_efficiency is not None:
            pileup_index = hp.ang2pix(pileup_efficiency_nside, seconds.ISSTerrestrialLongitude, seconds.ISSTerrestrialLatitude, lonlat=True)
            seconds_weight *= pileup_efficiency[pileup_index]

        # shape nseconds
        vectors = hp.ang2vec(seconds.AMSGalacticLongitude, seconds.AMSGalacticLatitude, lonlat=True).astype(np.float32)
        x = vectors[:,0]
        y = vectors[:,1]
        z = vectors[:,2]
        start_time = time.time()
        for pix_index in range(0, npix, npix_step_size):
            now = time.time()
            expected_total_seconds = (now - start_time) / max(1, pix_index) * npix
            print(f"{pix_index:>6}/{npix:>6}, {now-start_time:>5.1f}/{expected_total_seconds:>5.1f}s", flush=True)
            pixel_slice = slice(pix_index, pix_index + npix_step_size)
            # shape (nseconds, npix)
            altitude = (x[:,None] * ref_x[None,pixel_slice] + y[:,None] * ref_y[None,pixel_slice] + z[:,None] * ref_z[None,pixel_slice])
            altitude_mask = altitude >= min_altitude

            # add other weights/efficiencies here
            weights = np.broadcast_to(seconds_weight.astype(np.float32)[:,None], altitude.shape)
            healpy_indices = np.broadcast_to(ref_indices[pixel_slice][None,:], altitude.shape)

            #altitude_histogram.fill(np.ravel(healpy_indices), np.ravel(altitude), weights=np.ravel(weights))
            altitude_histogram.fill(healpy_indices[altitude_mask], altitude[altitude_mask], weights=weights[altitude_mask])

    return altitude_histogram, min_time, max_time


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--exposure-trees", required=True, nargs="+", help="Path to ROOT trees containing ISS location and AMS direction information.")
    parser.add_argument("--effective-area", required=True, help="Path to NPZ file containing the effective area to be integrated.")
    parser.add_argument("--pileup-efficiency", required=False, help="Path to NPZ file containing the pileup efficiency as a function of terrestrial location.")
    parser.add_argument("--nside", type=int, default=256, help="Number of HEALPix sides to use in the skymap binning.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--outputprefix", default="Exposure", help="Prefix for plots and result files.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--parallel", type=int, nargs=2, default=(0, 1), help="Job index and number of jobs to use in parallel.")
    parser.add_argument("--time-range", type=parse_datetime, nargs=2, help="Time range to integrate the exposure over.")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Number of seconds to handle per step.")

    args = parser.parse_args()

    os.makedirs(args.resultdir, exist_ok=True)
    os.makedirs(args.plotdir, exist_ok=True)

    with np.load(args.effective_area) as effective_area_file:
        effective_area_2d = load_histogram(effective_area_file, "effective_area_2d")

    pileup_efficiency = None
    pileup_efficiency_nside = None
    if args.pileup_efficiency is not None:
        with np.load(args.pileup_efficiency) as pileup_file:
            pileup_efficiency = pileup_file["pileup_efficiency"]
            pileup_efficiency_nside = pileup_file["nside"]

    energy_binning, altitude_binning = effective_area_2d.binnings
    healpy_binning = make_healpy_binning(args.nside)
    total_min_time, total_max_time = None, None
    min_altitude = altitude_binning.edges[1]

    parallel_index, parallel_total = args.parallel

    def _make_args():
        for rank in range(args.nprocesses):
            yield (args.exposure_trees, args.nprocesses * parallel_index + rank, args.nprocesses * parallel_total, dict(nside=args.nside, chunk_size=args.chunk_size, healpy_binning=healpy_binning, altitude_binning=altitude_binning, time_range=args.time_range, min_altitude=min_altitude, pileup_efficiency=pileup_efficiency, pileup_efficiency_nside=pileup_efficiency_nside))

    total_altitude_histogram = WeightedHistogram(healpy_binning, altitude_binning, labels=("Healpy Index", "cos(theta)"))

    with mp.Pool(args.nprocesses) as pool:
        for altitude_hist, min_time, max_time in pool.imap_unordered(handle_file, _make_args()):
            print(altitude_hist, min_time, max_time)
            total_altitude_histogram.add(altitude_hist)
            if total_min_time is None:
                total_min_time = min_time
                total_max_time = max_time
            else:
                total_min_time = min(total_min_time, min_time)
                total_max_time = max(total_max_time, max_time)

    #exposure_hist = calculate_exposure(total_altitude_histogram, effective_area_2d)

    results = {}
    total_altitude_histogram.add_to_file(results, f"altitude_time")
    #exposure_hist.add_to_file(results, "exposure")
    results["start_time"] = total_min_time
    results["end_time"] = total_max_time
    np.savez(os.path.join(args.resultdir, f"{args.outputprefix}_exposure.npz"), **results)

    #plot_exposures(exposure_hist, total_min_time, total_max_time, nside=args.nside, outputprefix=f"{args.outputprefix}_galactic", plotdir=args.plotdir, rotate=False)
    #plot_exposures(exposure_hist, total_min_time, total_max_time, nside=args.nside, outputprefix=f"{args.outputprefix}_equatorial", plotdir=args.plotdir, rotate=True)


if __name__ == "__main__":
    main()
