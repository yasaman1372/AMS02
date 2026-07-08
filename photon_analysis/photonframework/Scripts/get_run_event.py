#!/usr/bin/env python3

import os
import multiprocessing as mp

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.projections.geo import GeoAxes
import healpy as hp
from healpy.rotator import Rotator

from tools.binnings import make_healpy_binning
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d, load_histograms_from_files, plot_2d
from tools.roottree import read_tree
from tools.utilities import save_figure
from tools.constants import SKY_REGIONS
from tools.healpy_tools import mask_sources

BRANCHES = ["GalacticLongitude", "GalacticLatitude", "RunNumber", "EventNumber"]
signal_region = SKY_REGIONS["Signal_Region"]
bg_region = SKY_REGIONS["Background_Region"]


def handle_file(arg):
    filenames, treename, chunk_size, rank, nranks, kwargs = arg

    nside = kwargs["nside"]
    healpy_binning = kwargs["healpy_binning"]
    Map = np.arange(hp.nside2npix(nside))

    signal_ids = mask_sources(Map, signal_region, mask_type=False, inclusive=False, return_indices=True)
    bg_ids = mask_sources(Map, bg_region, return_indices=True)

    s_run_number = []
    s_event_number = []
    bg_run_number = []
    bg_event_number = []

    for events in read_tree(filenames, treename, rank=rank, nranks=nranks, chunk_size=chunk_size, branches=BRANCHES, cache_file=False):
        healpy_index = hp.ang2pix(nside=nside, theta=events.GalacticLongitude, phi=events.GalacticLatitude, lonlat=True)
        RunN = events.RunNumber
        EventN = events.EventNumber
        for i, index in enumerate(healpy_index):
            if index in signal_ids:
                s_run_number.append(RunN[i])
                s_event_number.append(EventN[i])

            elif index in bg_ids:
                bg_run_number.append(RunN[i])
                bg_event_number.append(EventN[i])

    
    return s_run_number, s_event_number, bg_run_number, bg_event_number


def make_args(filename, treename, chunk_size, nranks, **kwargs):
    for rank in range(nranks):
        yield (filename, treename, chunk_size, rank, nranks, kwargs)


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--tree", nargs="+", required=True, help="ROOT files to read.")
    parser.add_argument("--treename", default="GammaTree", help="Name of the tree in the ROOT files.")
    parser.add_argument("--nside", type=int, default=128, help="Number of sides in the healpix binning.")
    parser.add_argument("--outputprefix", default="Run_event_numbers", help="Prefix for plots and result files.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Number of events to read in parallel.")

    args = parser.parse_args()
    os.makedirs(args.resultdir, exist_ok=True)

    healpy_binning = make_healpy_binning(args.nside)


    s_run = []
    s_event = []
    bg_run = []
    bg_event = []

    with mp.Pool(args.nprocesses) as pool:
        pool_args = make_args(args.tree, args.treename, chunk_size=args.chunk_size, nranks=args.nprocesses, healpy_binning=healpy_binning, nside=args.nside)
        for s_r,s_e,bg_r,bg_e in pool.imap_unordered(handle_file, pool_args):
            s_run += s_r
            s_event += s_e
            bg_run += bg_r
            bg_event += bg_e


    signal = np.column_stack((np.array(s_run),np.array(s_event)))
    bg = np.column_stack((np.array(bg_run), np.array(bg_event)))

    header = "Run, Event"

    np.savetxt(os.path.join(args.resultdir,f"{args.outputprefix}_signal.csv"), signal, delimiter=" ", header=header, fmt="%d")
    np.savetxt(os.path.join(args.resultdir,f"{args.outputprefix}_bg.csv"), bg, delimiter=" ", header=header, fmt="%d")

   

   

if __name__ == "__main__":
    main()
