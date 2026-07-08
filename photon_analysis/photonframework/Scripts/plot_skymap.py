#!/usr/bin/env python3

from datetime import datetime, timezone
import os
import multiprocessing as mp

import matplotlib as mpl

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm, Normalize
from matplotlib.projections.geo import GeoAxes
from matplotlib.patches import Circle, Rectangle
import healpy as hp
from healpy.rotator import Rotator

from tools.config import get_config
from tools.constants import BASTIAN_SKYMAP_PIXEL_AREA, DATA_TIME_RANGES, SKY_REGIONS, SOURCES, TRUE_SOURCES
from tools.binnings import make_healpy_binning, make_energy_binning_from_config, make_flux_energy_binning
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d, load_histograms_from_files, plot_2d
from tools.roottree import read_tree
from tools.utilities import save_figure
from tools.healpy_tools import mask_sources, get_regions, get_sources

from calculate_point_spread_function import resolution_parametrization


def apply_smoothing(sky_histogram, nside, resolution_parameters, min_smoothing=False):
    """
    Applies energy-dependent smoothing to a 2D histogram of energy bins and Healpix indices
    """

    smoothed_skymap_values = np.zeros_like(sky_histogram.values, dtype=np.float64)
    smoothed_skymap_squared_values = np.zeros_like(sky_histogram.squared_values, dtype=np.float64)

    # Loop over each energy bin
    for energy_bin, (sky_histogram_1d, energy_min, energy_max) in enumerate(sky_histogram.project_all(axis=0, include_overflow=True)):
        # Compute the energy-dependent sigma for smoothing
        energy_center = (energy_min + energy_max) / 2
        #print(energy_center)
        sigma = np.radians(resolution_parametrization(energy_center, *resolution_parameters))

        pixel_size = hp.nside2pixarea(nside)
        min_sigma = 2 * np.sqrt(pixel_size)

        if min_smoothing and sigma < min_sigma:
            sigma = min_sigma

        # Smooth the map for this energy bin
        smoothed_slice = hp.smoothing(sky_histogram_1d.values[1:-1], sigma=sigma)
        smoothed_squared_slice = hp.smoothing(sky_histogram_1d.squared_values[1:-1], sigma=sigma)

        smoothed_skymap_values[energy_bin,1:-1] = smoothed_slice
        smoothed_skymap_squared_values[energy_bin,1:-1] = smoothed_squared_slice

    return WeightedHistogram(*sky_histogram.binnings, values=smoothed_skymap_values, squared_values=smoothed_skymap_squared_values)


def _label_energy(energy_value):
    if energy_value == 0:
        return "0 GeV"
    elif energy_value < 1:
        return f"{energy_value * 1000:.0f} MeV"
    elif energy_value < 10:
        return f"{energy_value:.2f} GeV"
    elif energy_value < 100:
        return f"{energy_value:.1f} GeV"
    elif energy_value < 1000:
        return f"{energy_value:.0f} GeV"
    elif energy_value < 10_000:
        return f"{energy_value / 1000:.2f} TeV"
    elif energy_value < 100_000:
        return f"{energy_value / 1000:.1f} TeV"
    return f"{energy_value / 1000:.0f} TeV"


def handle_file(arg):
    filenames, treename, chunk_size, rank, nranks, kwargs = arg

    nside = kwargs["nside"]
    healpy_binning = kwargs["healpy_binning"]
    energy_binning = kwargs.get("energy_binning", None)
    energy_range = kwargs.get("energy_range", None)
    min_time = kwargs.get("min_time", None)
    max_time = kwargs.get("max_time", None)

    time_range = kwargs.get("time_range", None)

    latitude_range = kwargs["latitude_range"]
    longitude_range = kwargs["longitude_range"]

    longitude_branch = kwargs.get("longitude_branch", "GalacticLongitude")
    latitude_branch = kwargs.get("latitude_branch", "GalacticLatitude")
    energy_branch = kwargs.get("energy_branch", "Energy")

    branches = [longitude_branch, latitude_branch, "TotalWeight"]
    if energy_range is not None or energy_binning is not None:
        branches.append(energy_branch)
    if time_range is not None:
        branches.append("RunNumber")

    if energy_binning is not None:
        sky_histogram = WeightedHistogram(energy_binning, healpy_binning)
    else:
        sky_histogram = WeightedHistogram(healpy_binning)

    cut_region = None
    if kwargs.get("cut_region", None) is not None:
        cut_region_filename, invert_cut_region_str = kwargs["cut_region"]
        invert_cut_region = invert_cut_region_str == "invert"
        cut_region = get_regions(cut_region_filename)
        cut_map = np.invert(mask_sources(np.ones(hp.nside2npix(nside)), cut_region, mask_type=invert_cut_region, inclusive=invert_cut_region).mask)
        in_region_indices = np.arange(len(cut_map))[cut_map]

    for events in read_tree(filenames, treename, rank=rank, nranks=nranks, chunk_size=chunk_size, branches=branches, cache_file=False):

        if time_range is not None:
            events = events[events.RunNumber <= DATA_TIME_RANGES[time_range]]
        
        else:
            if min_time is not None:
                events = events[(events.Time >= min_time)]
            if max_time is not None:
                events = events[(events.Time <= max_time)]

        if energy_range is not None:
            events = events[(events[energy_branch] >= energy_range[0]) & (events[energy_branch] <= energy_range[1])]

        if latitude_range is not None:
            events = events[(events[latitude_branch] >= latitude_range[0]) & (events[latitude_branch] <= latitude_range[1])]

        if longitude_range is not None:
            events = events[(events[longitude_branch] >= longitude_range[0]) & (events[longitude_branch] <= longitude_range[1])]

        events = events[np.isfinite(events[latitude_branch]) & np.isfinite(events[longitude_branch])]

        healpy_index = hp.ang2pix(nside=nside, theta=events[longitude_branch], phi=events[latitude_branch], lonlat=True)

        if cut_region is not None:
            region_selection = np.isin(healpy_index, in_region_indices)
            events = events[region_selection]
            healpy_index = healpy_index[region_selection]

        if energy_binning is not None:
            sky_histogram.fill(events[energy_branch], healpy_index, weights=events.TotalWeight)
        else:
            sky_histogram.fill(healpy_index, weights=events.TotalWeight)

    return sky_histogram


def calculate_signal_to_background_ratio(sky_histogram, signal_region, background_region, nside):
    pix2sr = 1 / hp.nside2pixarea(nside)
    signal_map = mask_sources(sky_histogram.values[1:-1], signal_region, mask_type=False, inclusive=False)
    background_map = mask_sources(sky_histogram.values[1:-1], background_region)
    signal = signal_map.sum() / signal_map.count() * pix2sr
    background = background_map.sum() / background_map.count() * pix2sr
    return signal / background

def no_normalization(nside):
    return 1

def normalize_per_square_degree(nside):
    return hp.nside2pixarea(nside, degrees=True)

def normalize_per_steradiant(nside):
    return hp.nside2pixarea(nside)

def normalize_per_bastian_pixel(nside):
    return hp.nside2pixarea(nside) / BASTIAN_SKYMAP_PIXEL_AREA

NORMALIZATIONS = {
    "events": (no_normalization, "Events"),
    "events-per-square-degree": (normalize_per_square_degree, "Events / ${}^{\\circ^2}$"),
    "events-per-steradiant": (normalize_per_steradiant, "Events / sr"),
    "events-per-bastian-pixel": (normalize_per_bastian_pixel, "Events / pixel"),
    "flux": (no_normalization, "$\\Phi$ / $GeV^{-1} cm^{-2} s^{-1} sr^{-1}$"),
    "trd_pileup": (no_normalization, "TRD Veto Efficiency"),
    "rel_background": (no_normalization, "Relative Background"),
    "bg_residuum": (no_normalization, "(Data - Model) / events")
}


def plot_skymap(sky_histogram, resultdir, plotdir, outputprefix, title, colormap="jet", vmin=0, vmax=None, scale=None, mask_region=None, invert_mask=False, rotate=None, resolution=30, save_pickle=False, save_fits=True, save_pdf=False, plot_sources=False, normalization="events-per-square-degree", colorbar="horizontal", latitude_range=None, longitude_range=None, transparent=False, rotate_center=None, mark_point=None, mark_point_size=None):

    fig_width = 8.5
    fig_height = 4.1
    if colorbar == "vertical":
        fig_width += 0.2
    elif colorbar == "horizontal":
        fig_height += 0.5
    if title is not None:
        fig_height += 0.5

    figsize = (fig_width, fig_height)
    subplot_kw = dict(projection="hammer")
    if latitude_range is not None or longitude_range is not None:
        subplot_kw = None
        figsize = (6, 5)
    figure = plt.figure(figsize=figsize)

    plot = figure.subplots(1, 1, subplot_kw=subplot_kw)
    nside = hp.npix2nside(len(sky_histogram.values) - 2)
    if title is not None:
        plot.set_title(title, pad=15, fontsize=10)
    theta_pixels = 90 * resolution
    phi_pixels = 180 * resolution
    theta = np.linspace(np.pi, 0, theta_pixels)
    phi = np.linspace(-np.pi, np.pi, phi_pixels)
    phi, theta = np.meshgrid(phi, theta)
    if rotate is not None and rotate != "Terrestrial":
        r = Rotator(coord=rotate, inv=True)
        theta, phi = r(theta.flatten(), phi.flatten())
        theta = theta.reshape(theta_pixels, phi_pixels)
        phi = phi.reshape(theta_pixels, phi_pixels)
    elif rotate_center is not None:
        r = Rotator(rot=rotate_center, inv=True, deg=True)
        theta, phi = r(theta.flatten(), phi.flatten())
        theta = theta.reshape(theta_pixels, phi_pixels)
        phi = phi.reshape(theta_pixels, phi_pixels)
    grid_pix = hp.ang2pix(nside, theta, phi)
    map_values = sky_histogram.values[1:-1]
    if mask_region is not None:
        map_values = mask_sources(map_values, mask_region, mask_type=not invert_mask, inclusive=not invert_mask)
    grid_map = map_values[grid_pix]
    pixel_norm_func, colorscale_label = NORMALIZATIONS[normalization]
    pixel_norm = pixel_norm_func(nside)

    theta_plot = np.linspace(np.pi, 0, theta_pixels)
    phi_plot = np.linspace(-np.pi, np.pi, phi_pixels)

    if not scale or scale == 'sqrt':
        norm = PowerNorm(0.5, vmin=vmin, vmax=vmax, clip=True)
    elif scale == 'log':
        norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
    elif scale == 'lin':
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    mesh = plot.pcolormesh(-phi_plot, -theta_plot + np.pi / 2, grid_map / pixel_norm, norm=norm, cmap=colormap)

    plot.grid(alpha=0.3)

    if colorbar:
        if colorbar == "horizontal":
            colorbar_args = dict(orientation="horizontal", fraction=0.08, pad=0.05, aspect=30, shrink=0.7)
        elif colorbar == "vertical":
            colorbar_args = dict(orientation="vertical", fraction=0.08)
        else:
            raise ValueError(f"colorbar has to be horizontal or vertical, not {colorbar!r}.")
        cbar = plt.colorbar(mesh, ax=plot, extend="max", **colorbar_args)
        cbar.set_label(colorscale_label, fontsize=12, fontweight="bold")
        cbar.ax.tick_params(labelsize=10, direction="in", length=5)

    plot.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / np.pi * 180:.0f}°"))

    if rotate_center is None:
        if rotate is None or rotate[-1] == "G":
            plot.text(1, 0, "Galactic Coordinates", transform=plot.transAxes, ha="right", va="bottom")
            plot.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{(-x / np.pi * 180) % 360:.0f}°"))
        elif rotate[-1] == "C":
            plot.text(1, 0, "ICRS Coordinates", transform=plot.transAxes, ha="right", va="bottom")
            plot.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{(-x / np.pi * 180) % 360:.0f}°"))
        elif rotate[-1] == "E":
            plot.text(1, 0, "Ecliptic Coordinates", transform=plot.transAxes, ha="right", va="bottom")
            plot.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{(-x / np.pi * 180) % 360:.0f}°"))
        elif rotate == "Terrestrial":
            plot.text(1,0,"Terrestrial Coordinates", transform=plot.transAxes, ha="right", va="bottom")
            plot.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{(x / np.pi * 180) % 360:.0f}°"))    
        else:
            raise ValueError(f"Unknown coordinate system {rotate!r}.")
    else:
        plot.text(1, 0, f"Centered at {rotate_center}°", transform=plot.transAxes, ha="right", va="bottom")
        plot.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{(-x / np.pi * 180) % 360:.0f}°"))

    plot.tick_params(axis='x', colors='white')

    if mark_point is not None:
        point_longitude, point_latitude = np.radians(mark_point)
        if point_longitude > np.pi:
            point_longitude -= 2 * np.pi
        if mark_point_size is None:
            plot.plot(-point_longitude, point_latitude, "o", color="fuchsia", fillstyle="none", zorder=5)
        else:
            mark_point_size = np.radians(mark_point_size)
            point_rect = Rectangle((-point_longitude - mark_point_size / 2, point_latitude - mark_point_size / 2), mark_point_size, mark_point_size, fill=False, color="fuchsia")
            plot.add_patch(point_rect)

    if longitude_range is not None:
        min_longitude, max_longitude = longitude_range
        min_phi, max_phi = np.radians(-max_longitude % 360), np.radians(-min_longitude % 360)
        if min_phi > np.pi:
            min_phi -= 2 * np.pi
        if max_phi > np.pi:
            max_phi -= 2 * np.pi
        plot.set_xlim(min_phi, max_phi)
    if latitude_range is not None:
        min_latitude, max_latitude = latitude_range
        plot.set_ylim(np.radians(min_latitude), np.radians(max_latitude))

    if plot_sources:
        for name, source in SOURCES.items():
            source_circle = Circle(np.radians(source[0][0:2]), np.radians(source[0][2]), color = source[1], fill = False, linewidth = 1)
            plot.add_patch(source_circle)
            plot.text(*np.radians(source[2][0:2]), s = name, c = source[1], size = 'small')
        inner_galaxy = Rectangle(np.radians([-80, -8]), np.radians(100), np.radians(16), color = 'white', fill = False)
        plot.add_patch(inner_galaxy)
        plot.text(np.radians(-45), np.radians(10), s = 'Inner Galaxy', c = 'white', size = 'small')

    if title is not None:
        figure.subplots_adjust(left=0.033, right=0.975, bottom=0.015, top=0.95)
    else:
        figure.subplots_adjust(left=0.033, right=0.975, bottom=0.0175, top=0.985)
    save_figure(figure, plotdir, f"{outputprefix}", dpi=300, save_pickle=save_pickle, save_pdf=save_pdf, transparent=transparent)

    if rotate is None:
        coord = 'G'
    else:
        coord = rotate[-1]

    if save_fits:
        if rotate is None:
            coord = 'G'
        else:
            coord = rotate[-1]

        header = {
            "BUNIT": "deg^2",       # Units of the map
            "MAPTYPE": "events",    # Description of the data
            "NSIDE": nside,         # HEALPix nside parameter
            "COORDSYS": coord,      # Coordinate system
        }

        if isinstance(map_values, np.ma.masked_array):
            map_values = map_values.filled(0)

        hp.fitsfunc.write_map(os.path.join(resultdir, f"{outputprefix}.fits"), map_values, fits_IDL=True, extra_header=header, overwrite=True)



def make_args(filename, treename, chunk_size, nranks, **kwargs):
    for rank in range(nranks):
        yield (filename, treename, chunk_size, rank, nranks, kwargs)


COLORMAP_LABELCOLORS = {
    "viridis": "white",
    "jet": "lightgray",
}


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", nargs=2, required=True, help="Path to config file and working directory.")
    parser.add_argument("--tree", nargs="+", required=True, help="ROOT files to read.")
    parser.add_argument("--treename", default="GammaTree", help="Name of the tree in the ROOT files.")
    parser.add_argument("--nside", type=int, default=128, help="Number of sides in the healpix binning.")
    parser.add_argument("--outputprefix", default="SkyMap", help="Prefix for plots and result files.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--title", default="Gamma Ray Sky Map", help="Title for plots.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Number of events to read in parallel.")
    parser.add_argument("--vmin", type=float, default=0, help="Number of events per bin to use as the minimum of the colorscale.")
    parser.add_argument("--vmax", type=float, default=None, help="Number of events per bin to use as the maximum of the colorscale.")
    parser.add_argument('--scale', default='sqrt', choices=['sqrt', 'log', 'lin'], help='Scale of the map')
    parser.add_argument("--colormap", choices=["viridis", "jet", "magma", "plasma", "inferno"], default="jet", help="Colormap to use.")
    parser.add_argument("--coord", nargs="+", type=str, default=None, choices=['G','C','E', 'Terrestrial'], help="Either one of ‘G’ (Galactic), ‘E’ (Ecliptic) or ‘C’ (Celestial/Equatorial) to describe the coordinate system of the map, or a sequence of 2 of these to rotate the map from the first to the second coordinate system. The standard map is in 'G'.")
    parser.add_argument("--s-bg-regions", nargs=2, type=str, help='Path to .reg file for signal and bg region')
    parser.add_argument("--cut-region", nargs=2, metavar=["regionfile", "invert"], help="Plot (and count) only photons within this region.")
    parser.add_argument("--latitude-range", type=float, nargs=2, metavar=["min", "max"], help="Plot (and count) only photons from within this latitude range.")
    parser.add_argument("--longitude-range", type=float, nargs=2, metavar=["min", "max"], help="Plot (and count) only photons from within this longitude range.")
    parser.add_argument("--longitude-branch", default="GalacticLongitude", help="Branch to use as the longitude.")
    parser.add_argument("--latitude-branch", default="GalacticLatitude", help="Branch to use as the latitude.")
    parser.add_argument("--not-per-energy", dest="per_energy", action="store_false", help="Make only one skymap over all energies.")
    parser.add_argument("--normalization", choices=list(NORMALIZATIONS.keys()), default="events", help="Pixel normalization to apply.")
    parser.add_argument("--psf", help="Path to a file containing a parametrized point spread function. If passed, the skymap will be smoothed with it.")
    parser.add_argument("--time-range", choices=DATA_TIME_RANGES.keys(), help="Plot the sky map only for a selected time range.")
    parser.add_argument('--time-range-limits', nargs=2, type=str, default=[None, None], help="Minimal and Maximal time (format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).")
    parser.add_argument("--per-energy-maps", action="store_true", help="Plot skymaps for every energy bin.")
    parser.add_argument('--save-pickle', action="store_true", help='Store the Plots as pickeld matplotlib plotts.')
    parser.add_argument('--save-pdf', action="store_true", help='Store the Plots as psf.')
    parser.add_argument("--sources", action="store_true", help ="Should the most notable sources be plotet onto the skymap.")
    parser.add_argument("--energy-range", nargs=2, type=float, help='Plot the Skymap for the given energy range in GeV min, max')
    parser.add_argument("--no-title", action='store_true', help='Use this to prevent titles from being generated.')
    parser.add_argument("--transparent", action="store_true", help='set this to make the background of the png transparent')
    parser.add_argument("--energy-binning", default='default', choices=['default', 'flux', 'config'], help='what energy binning should be used')
    parser.add_argument("--colorbar", choices=["horizontal", "vertical", "none"], help="Colorbar positioning, or none to not draw a colorbar.")
    parser.add_argument("--mark-point", nargs=2, type=float, help="Mark point at certain coordinates.")
    parser.add_argument("--mark-point-size", type=float, help="Replace the point marker with a rectangle of this size (in degrees).")

    args = parser.parse_args()
    config_filename, workdir = args.config
    config = get_config(config_filename)
    healpy_binning = make_healpy_binning(args.nside)

    if args.energy_binning in ("default", "config"):
        energy_binning = make_energy_binning_from_config(config)
    elif args.energy_binning == "flux":
        energy_binning = make_flux_energy_binning()

    min_time = args.time_range_limits[0]
    max_time = args.time_range_limits[1]
    if min_time is not None:
        min_time_unix =  datetime.fromisoformat(min_time).replace(tzinfo=timezone.utc).timestamp()
    else:
        min_time_unix = None
    if max_time is not None:
        max_time_unix = datetime.fromisoformat(max_time).replace(tzinfo=timezone.utc).timestamp()
    else:
        max_time_unix = None

    kwargs = {
        "nside": args.nside,
        "healpy_binning": healpy_binning,
        "time_range": args.time_range,
        "energy_binning": energy_binning,
        "energy_range": args.energy_range,
        "min_time": min_time_unix,
        "max_time": max_time_unix,
        "cut_region": args.cut_region,
        "longitude_range": args.longitude_range,
        "latitude_range": args.latitude_range,
        "longitude_branch": args.longitude_branch,
        "latitude_branch": args.latitude_branch,
    }
    if args.per_energy:
        kwargs["energy_binning"] = energy_binning
        sky_histogram = WeightedHistogram(energy_binning, healpy_binning)
    else:
        sky_histogram = WeightedHistogram(healpy_binning)

    os.makedirs(args.plotdir, exist_ok=True)
    os.makedirs(args.resultdir, exist_ok=True)


    with mp.Pool(args.nprocesses) as pool:
        pool_args = make_args(args.tree, args.treename, chunk_size=args.chunk_size, nranks=args.nprocesses, **kwargs)
        for sky_hist in pool.imap_unordered(handle_file, pool_args):
            sky_histogram.add(sky_hist)

    if args.per_energy:
        sky_histogram_1d = sky_histogram.project_axis(axis=0)
        sky_histogram_2d = sky_histogram
    else:
        sky_histogram_1d = sky_histogram
    total_events = sky_histogram_1d.values.sum()
    if args.s_bg_regions is not None:
        signal_region, background_region = args.s_bg_regions
        signal_to_background_ratio = calculate_signal_to_background_ratio(sky_histogram_1d, get_regions(signal_region), get_regions(background_region), args.nside)

    title = args.title
    if args.energy_range is not None:
        energy_min, energy_max = args.energy_range
        label_emin = _label_energy(energy_min)
        label_emax = _label_energy(energy_max)
        title = f"{title}, {label_emin} to {label_emax}"
    if args.time_range is not None:
        end_date = datetime.fromtimestamp(DATA_TIME_RANGES[args.time_range])
        title = f"{title}, until {end_date:%b %Y}"

    if args.s_bg_regions is not None:
        title = f"{title}: {total_events:.0f} events, s/bg: {signal_to_background_ratio:.2f}"
    else:
        title = f"{title}: {total_events:.0f} events"

    if args.no_title:
        title = None

    colorbar = args.colorbar
    if colorbar == "none":
        colorbar = None

    plot_kwargs = dict(
        resultdir=args.resultdir, plotdir=args.plotdir,
        colormap=args.colormap, vmin=args.vmin, vmax=args.vmax,
        rotate=args.coord, save_pickle=args.save_pickle,
        plot_sources=args.sources, normalization=args.normalization,
        transparent=args.transparent, scale=args.scale,
        colorbar=colorbar,
        mark_point=args.mark_point, mark_point_size=args.mark_point_size,
        latitude_range=args.latitude_range, longitude_range=args.longitude_range,
    )

    plot_skymap(sky_histogram_1d, outputprefix=f"{args.outputprefix}_skymap_raw", title=title, **plot_kwargs)
    if args.s_bg_regions is not None:
        plot_skymap(sky_histogram_1d, outputprefix=f"{args.outputprefix}_skymap_signal_raw", title=title, mask_region=get_regions(signal_region), invert_mask=True, **plot_kwargs)
        plot_skymap(sky_histogram_1d, outputprefix=f"{args.outputprefix}_skymap_background_raw", title=title, mask_region=get_regions(background_region), invert_mask=False, **plot_kwargs)

    if args.per_energy and args.per_energy_maps:
        for energy_bin, (sky_histogram_1d, energy_min, energy_max) in enumerate(sky_histogram_2d.project_all(axis=0, include_overflow=True)):
            plot_skymap(sky_histogram_1d, outputprefix=f"{args.outputprefix}_skymap_raw_{energy_bin}", title=None if args.no_title else f"{title}, {energy_min:.4g}<=E/GeV<{energy_max:.4g}", **plot_kwargs)

    if args.per_energy and args.psf is not None:
        with np.load(args.psf) as psf_file:
            resolution_parameters = psf_file["resolution_parameters_y"]
        smoothed_sky_histogram_2d = apply_smoothing(sky_histogram_2d, args.nside, resolution_parameters)
        smoothed_sky_histogram_1d = smoothed_sky_histogram_2d.project_axis(axis=0)
        plot_skymap(smoothed_sky_histogram_1d, outputprefix=f"{args.outputprefix}_skymap_smoothed", title=title, **plot_kwargs)
        if args.s_bg_regions is not None:
            plot_skymap(smoothed_sky_histogram_1d, outputprefix=f"{args.outputprefix}_skymap_signal_smoothed", title=title, mask_region=get_regions(signal_region), invert_mask=True, **plot_kwargs)
            plot_skymap(smoothed_sky_histogram_1d, outputprefix=f"{args.outputprefix}_skymap_background_smoothed", title=title, mask_region=get_regions(background_region), invert_mask=False, **plot_kwargs)

        if args.per_energy_maps:
            for energy_bin, (sky_histogram_1d, energy_min, energy_max) in enumerate(smoothed_sky_histogram_2d.project_all(axis=0, include_overflow=True)):
                plot_skymap(sky_histogram_1d, outputprefix=f"{args.outputprefix}_skymap_smoothed_{energy_bin}", title=f"{title}, {energy_min:.4g}<=E/GeV<{energy_max:.4g}", **plot_kwargs)

    results = {}
    sky_histogram_2d.add_to_file(results, 'raw')
    smoothed_sky_histogram_2d.add_to_file(results, 'smoothed')
    np.savez(os.path.join(args.resultdir, f"{args.outputprefix}_histograms.npz"), **results)

if __name__ == "__main__":
    main()
