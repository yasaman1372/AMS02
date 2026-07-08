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

from tools.binnings import make_day_binning_from_config, make_lin_binning, make_healpy_binning, make_int_binning
from tools.config import get_config
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

    chunk_size = kwargs["chunk_size"]

    altitude_binning = kwargs["altitude_binning"]
    min_altitude = kwargs["min_altitude"]
    sources = kwargs["sources"]
    time_binning = kwargs["time_binning"]

    pileup_efficiency = kwargs["pileup_efficiency"]
    pileup_efficiency_nside = kwargs["pileup_efficiency_nside"]

    source_histograms = {source_name: WeightedHistogram(time_binning, altitude_binning, labels=("Time", "cos(theta)")) for source_name in sources}
    ref_vectors = {source_name: hp.ang2vec(longitude, latitude, lonlat=True) for source_name, (longitude, latitude) in sources.items()}
    ref_x = np.array([ref_vectors[source_name][0] for source_name in sources])
    ref_y = np.array([ref_vectors[source_name][1] for source_name in sources])
    ref_z = np.array([ref_vectors[source_name][2] for source_name in sources])

    for seconds in read_tree(filenames, "ExposureTree", rank=rank, nranks=nranks, branches=BRANCHES, chunk_size=chunk_size):
        seconds_weight = ak.to_numpy(seconds.TriggerLiveTime)

        if pileup_efficiency is not None:
            pileup_index = hp.ang2pix(pileup_efficiency_nside, seconds.ISSTerrestrialLongitude, seconds.ISSTerrestrialLatitude, lonlat=True)
            seconds_weight *= pileup_efficiency[pileup_index]

        # shape nseconds
        vectors = hp.ang2vec(seconds.AMSGalacticLongitude, seconds.AMSGalacticLatitude, lonlat=True).astype(np.float32)
        x = vectors[:,0]
        y = vectors[:,1]
        z = vectors[:,2]
        # shape (nseconds, nsources)
        altitude = (x[:,None] * ref_x[None,:] + y[:,None] * ref_y[None,:] + z[:,None] * ref_z[None,:])
        altitude_mask = altitude >= min_altitude

        for source_index, (source_name, source_histogram) in enumerate(source_histograms.items()):
            mask = altitude_mask[:,source_index]
            source_histogram.fill(seconds.UTCTime[mask], altitude[mask, source_index], weights=seconds_weight[mask])

    return source_histograms


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument("--exposure-trees", required=True, nargs="+", help="Path to ROOT trees containing ISS location and AMS direction information.")
    parser.add_argument("--effective-area", required=True, help="Path to NPZ file containing the effective area to be integrated.")
    parser.add_argument("--pileup-efficiency", required=False, help="Path to NPZ file containing the pileup efficiency as a function of terrestrial location.")
    parser.add_argument("--source", required=True, dest="sources", action="append", nargs=3, metavar=("name",  "longitude", "latitude"), help="Source name and direction to calculate the exposure for.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--outputprefix", default="Exposure", help="Prefix for plots and result files.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--parallel", type=int, nargs=2, default=(0, 1), help="Job index and number of jobs to use in parallel.")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Number of seconds to handle per step.")

    args = parser.parse_args()

    config = get_config(args.config)

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
    sources = {source_name: (float(longitude), float(latitude)) for (source_name, longitude, latitude) in args.sources}
    time_binning = make_day_binning_from_config(config)
    min_altitude = altitude_binning.edges[1]

    parallel_index, parallel_total = args.parallel

    def _make_args():
        for rank in range(args.nprocesses):
            yield (args.exposure_trees, args.nprocesses * parallel_index + rank, args.nprocesses * parallel_total, dict(chunk_size=args.chunk_size, sources=sources, time_binning=time_binning, altitude_binning=altitude_binning, min_altitude=min_altitude, pileup_efficiency=pileup_efficiency, pileup_efficiency_nside=pileup_efficiency_nside))

    source_altitude_histograms = {source_name: WeightedHistogram(time_binning, altitude_binning, labels=("Time", "cos(theta)")) for source_name in sources}

    with mp.Pool(args.nprocesses) as pool:
        for source_hists in pool.imap_unordered(handle_file, _make_args()):
            for source_name in sources:
                source_altitude_histograms[source_name].add(source_hists[source_name])

    source_exposure_histograms = {source_name: calculate_exposure(altitude_hist, effective_area_2d) for source_name, altitude_hist in source_altitude_histograms.items()}

    results = {}
    for source_name, altitude_histogram in source_altitude_histograms.items():
        altitude_histogram.add_to_file(results, f"altitude_{source_name}")
    for source_name, exposure_histogram in source_exposure_histograms.items():
        exposure_histogram.add_to_file(results, f"exposure_{source_name}")
    np.savez(os.path.join(args.resultdir, f"{args.outputprefix}_source_exposure.npz"), **results)

if __name__ == "__main__":
    main()
