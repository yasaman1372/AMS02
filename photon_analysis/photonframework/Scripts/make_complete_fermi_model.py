#!/usr/bin/env python3

from datetime import datetime
import os
import multiprocessing as mp

import matplotlib as mpl

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm, Normalize
from matplotlib.projections.geo import GeoAxes
from matplotlib.patches import Circle
import healpy as hp
from healpy.rotator import Rotator

from tools.config import get_config
from tools.constants import BASTIAN_SKYMAP_PIXEL_AREA, DATA_TIME_RANGES, SKY_REGIONS, SOURCES, TRUE_SOURCES
from tools.binnings import make_healpy_binning, make_energy_binning_from_config, make_flux_energy_binning
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d, load_histograms_from_files, plot_2d
from tools.roottree import read_tree
from tools.utilities import save_figure
from tools.healpy_tools import mask_sources, get_regions, get_sources

from plot_skymap import NORMALIZATIONS, plot_skymap

def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--source-model', required=True, help='The npz file with the source model histograms')
    parser.add_argument('--diffuse-model', required=True, help='The npz file with the diffuse model histograms')
    parser.add_argument("--outputprefix", default="SkyMap", help="Prefix for plots and result files.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--title", default="Gamma Ray Sky Map", help="Title for plots.")
    parser.add_argument("--vmin", type=float, default=1e-3, help="Number of events per bin to use as the minimum of the colorscale.")
    parser.add_argument("--vmax", type=float, default=1e0, help="Number of events per bin to use as the maximum of the colorscale.")
    parser.add_argument('--scale', default='log', choices=['sqrt', 'log', 'lin'], help='Scale of the map')
    parser.add_argument("--colormap", choices=["viridis", "jet", "magma", "plasma", "inferno"], default="jet", help="Colormap to use.")
    parser.add_argument("--coord", nargs="+", type=str, default=None, choices=['G','C','E'], help="Either one of ‘G’ (Galactic), ‘E’ (Ecliptic) or ‘C’ (Celestial/Equatorial) to describe the coordinate system of the map, or a sequence of 2 of these to rotate the map from the first to the second coordinate system. The standard map is in 'G'.")
    parser.add_argument("--s-bg-regions", nargs=2, type=str, help='Path to .reg file for signal and bg region')
    parser.add_argument("--normalization", choices=list(NORMALIZATIONS.keys()), default="events", help="Pixel normalization to apply.")
    parser.add_argument('--save-pickle', action="store_true", help='Store the Plots as pickeld matplotlib plotts.')
    parser.add_argument("--no-title", action='store_true', help='Use this to prevent titles from being generated.')
    parser.add_argument("--transparent", action="store_true", help='set this to make the background of the png transparent')
    
    args = parser.parse_args()

    os.makedirs(args.plotdir, exist_ok=True)
    os.makedirs(args.resultdir, exist_ok=True)


    Source_map = load_histograms_from_files([args.source_model])['fluxes']
    Diffuse_map = load_histograms_from_files([args.diffuse_model])['convoluted_N']

    Model = Histogram(*Source_map.binnings)

    Model.add(Source_map)
    Model.add(Diffuse_map)
    Model.values = np.nan_to_num(Model.values, nan=0)
    print(Model.values)
    print(Model.project_axis(axis=0).values)
    

    plot_skymap(Model.project_axis(axis=0), args.resultdir, args.plotdir, f'{args.outputprefix}_model', title = args.title , transparent=args.transparent, scale = 'log', vmin = args.vmin, vmax = args.vmax, normalization=args.normalization)
    if args.no_title:
        plot_skymap(Model.project_axis(axis=0), args.resultdir, args.plotdir, f'{args.outputprefix}_model_NoTitle', title = None , transparent=args.transparent, scale = 'log', vmin = args.vmin, vmax = args.vmax, normalization=args.normalization)
    for i, (hist, energy_min, energy_max) in enumerate(Model.project_all(axis=0)):
        title = f"Fermi Model, {energy_min:.4g}<=E/GeV<{energy_max:.4g}"
        plot_skymap(hist, args.resultdir, args.plotdir, f'{args.outputprefix}_model_{i}', title = title , transparent=args.transparent, scale = 'log', vmin = args.vmin, vmax = args.vmax, normalization=args.normalization)
        if args.no_title:
            plot_skymap(hist, args.resultdir, args.plotdir, f'{args.outputprefix}_model_{i}_NoTitle', title = None , transparent=args.transparent, scale = 'log', vmin = args.vmin, vmax = args.vmax, normalization=args.normalization)



    





if __name__ == "__main__":
    main()
