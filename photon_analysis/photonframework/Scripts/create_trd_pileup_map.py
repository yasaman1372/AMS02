#!/usr/bin/env python3

import numpy as np
import multiprocessing as mp
import os
import awkward as ak
import healpy as hp
from astropy.io import fits
from scipy.integrate import quad
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm, Normalize
from matplotlib.projections.geo import GeoAxes
from matplotlib.patches import Circle

from tools.binnings import make_lin_binning, make_energy_binning_from_config, make_mc_dataset_edge_binning, combine_binnings, reduce_bins, make_flux_energy_binning, make_healpy_binning
from tools.config import get_config
from tools.constants import BastianPlots
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d
from tools.roottree import read_tree
from tools.utilities import load_mc_trigger_count, save_figure, set_energy_ticks, plot_2d
from tools.statistics import hist_mean_and_std

from plot_skymap import plot_skymap, NORMALIZATIONS



def handle_file(arg):
    filenames, treename, chunk_size, rank, nranks, nside, kwargs = arg

    BRANCHES = ["ISSTerrestrialLongitude", "ISSTerrestrialLatitude", "NTrdHits", "NTrdSegmentsXZ", "NTrdSegmentsYZ"]
    #energy_binning = kwargs["energy_binning"]
    healpy_binning = kwargs["healpy_binning"]

    histogram = Histogram(healpy_binning)
    trd_histogram = Histogram(healpy_binning)

    for events in read_tree(filenames, treename, rank=rank, nranks=nranks, chunk_size=chunk_size, branches=BRANCHES, cache_file=False):
        healpy_index = hp.ang2pix(nside=nside, events.ISSTerrestrialLongitude, events.ISSTerrestrialLatitude, lonlat=True)
        histogram.fill(healpy_index)
        ntrd_mask = (events.NTrdHits <= 10.5)
        trdsegmentXZ_mask = (events.NTrdSegmentsXZ <= 0.5)
        trdsegmentYZ_mask = (events.NTrdSegmentsYZ <= 0.5)
        segment_mask = (trdsegmentXZ_mask & trdsegmentYZ_mask)
        mask = ntrd_mask & segment_mask
        trd_histogram.fill(healpy_index[mask])

    return histogram, trd_histogram

def make_args(filename, treename, chunk_size, nranks, nside, **kwargs):
    for rank in range(nranks):
        yield (filename, treename, chunk_size, rank, nranks,nside, kwargs)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs=2, required=True, help="Path to config file and working directory.")
    parser.add_argument("--tree", nargs="+", required=True, help="Gamma tree with TRD pileup cut but no TRD cut")
    parser.add_argument("--treename", default="GammaTree", help="Name of the tree in the ROOT file to read.")
    parser.add_argument("--outputprefix", default="TRD_pileup", help="Prefix for plots and result files.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Number of events to read each step.")
    parser.add_argument("--nside", type=int, default=256, help="Number of sides in the healpix binning.")
    parser.add_argument("--title", default="TRD Pileup", help="Title for plots.")
    parser.add_argument("--no-title", action='store_true', help='Use this to prevent titles from being generated.')
    parser.add_argument("--transparent", action="store_true", help='set this to make the background of the png transparent')
    parser.add_argument("--vmin", type=float, default=0, help="Number of events per bin to use as the minimum of the colorscale.")
    parser.add_argument("--vmax", type=float, default=None, help="Number of events per bin to use as the maximum of the colorscale.")
    parser.add_argument("--colormap", choices=["viridis", "jet", "magma", "plasma", "inferno"], default="jet", help="Colormap to use.")
    parser.add_argument('--scale', default='sqrt', choices=['sqrt', 'log', 'lin'], help='Scale of the map')
    parser.add_argument("--colormap", choices=["viridis", "jet", "magma", "plasma", "inferno"], default="jet", help="Colormap to use.")
    parser.add_argument("--coord", nargs="+", type=str, default=None, choices=['G','C','E'], help="Either one of ‘G’ (Galactic), ‘E’ (Ecliptic) or ‘C’ (Celestial/Equatorial) to describe the coordinate system of the map, or a sequence of 2 of these to rotate the map from the first to the second coordinate system. The standard map is in 'G'.")
    parser.add_argument("--normalization", choices=list(NORMALIZATIONS.keys()), default="events", help="Pixel normalization to apply.")
    parser.add_argument('--save-pickle', action="store_true", help='Store the Plots as pickeld matplotlib plotts.')
    parser.add_argument('--save-pdf', action="store_true", help='Store the Plots as psf.')
    parser.add_argument("--no-colorbar", action="store_true", help='set this option if you dont want the colorbar shown')

    args = parser.parse_args()
    config_filename, datadir = args.config
    config = get_config(config_filename)


    os.makedirs(args.resultdir, exist_ok=True)
    os.makedirs(args.plotdir, exist_ok=True)

    healpy_binning = make_healpy_binning(args.nside)

    Pileup_histogram = Histogram(healpy_binning, labels=("Eventes"))
    Pileup_trd_histogram = Histogram(healpy_binning, labels=("Eventes"))

    with mp.Pool(args.nprocesses) as pool:
        pool_args = make_args(args.tree, args.treename, chunk_size=args.chunk_size, nranks=args.nprocesses, nside=args.nside, healpy_binning=healpy_binning)
        for hist, trd_hist in pool.imap_unordered(handle_file, pool_args):
            Pileup_histogram.add(hist)
            Pileup_trd_histogram.add(trd_hist)
            

    Trd_pileup = Histogram(healpy_binning, labels=("Eventes"))
    Trd_pileup.values = Pileup_trd_histogram.values / Pileup_histogram.values

    plot_skymap(Trd_pileup, resultdir=args.resultdir, plotdir=args.plotdir, outputprefix=f"{args.outputprefix}_skymap_raw", title=args.title, colormap=args.colormap, vmin=args.vmin, vmax=args.vmax, save_pickle=args.save_pickle, save_pdf = args.save_pdf, normalization="trd_pileup", transparent=args.transparent, scale=args.scale, rotate="Terrestrial")
    if args.no_title:
        plot_skymap(Trd_pileup, resultdir=args.resultdir, plotdir=args.plotdir, outputprefix=f"{args.outputprefix}_skymap_raw_NoTitle", title=None, colormap=args.colormap, vmin=args.vmin, vmax=args.vmax, save_pickle=args.save_pickle, save_pdf = args.save_pdf, normalization="trd_pileup", transparent=args.transparent, scale=args.scale, rotate="Terrestrial")

    
    pix = np.arange(hp.nside2npix(args.nside))
    latitudes = hp.pix2ang(args.nside, pix, lonlat=True)[1]
    lat_rings = sorted(list(set(latitudes)))
    Pileup_per_pixel_in_lat = np.zeros(len(lat_rings))
    Pileup_per_pixel_in_lat_std = np.zeros(len(lat_rings))

    Trd_pileup.values[1:-1] = np.nan_to_num(Trd_pileup.values[1:-1], nan=0)
    
    for i, lat in enumerate(lat_rings):
        #value_mask = Trd_pileup.values[1:-1] != 0
        lat_mask = latitudes == lat
        mask = lat_mask 
        Pileup_per_pixel_in_lat[i] = np.mean(Trd_pileup.values[1:-1][mask])
        Pileup_per_pixel_in_lat_std[i] = np.std(Trd_pileup.values[1:-1][mask])/np.sqrt(np.sum(mask))
        print(lat, Pileup_per_pixel_in_lat[i])

    results = {}
    Trd_pileup.add_to_file(results, "pileup_efficiency_hist")
    results["pileup_efficiency"] = Trd_pileup.values[1:-1]
    results["nside"] = args.nside

    np.savez(os.path.join(args.resultdir, f"{args.outputprefix}.npz"), **results)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.errorbar(lat_rings, Pileup_per_pixel_in_lat, yerr = Pileup_per_pixel_in_lat_std, marker=".", ls='', ms=1, elinewidth=1 )
    ax.set_xlabel("Latitude / deg")
    ax.set_ylabel("TRD Veto Efficiency")

    save_figure(fig, args.plotdir, f"{args.outputprefix}_per_latitude", transparent=args.transparent, save_pdf=args.save_pdf)


if __name__ == "__main__":
    main()
