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
from healpy.rotator import Rotator

from tools.binnings import make_lin_binning, make_healpy_binning
from tools.histograms import Histogram, WeightedHistogram, load_histogram
from tools.roottree import read_tree
from tools.utilities import save_figure


def parse_datetime(datetime_str):
    if datetime_str.isnumeric():
        return float(datetime_str)
    return datetime.fromisoformat(datetime_str).replace(tzinfo=timezone.utc).timestamp()


def plot_exposures(exposure_hist_2d, start_time, end_time, nside, outputprefix, plotdir, energy=None, colormap="viridis", transparent=False, save_pdf=False, no_title=False, rotate=None):
    if energy is not None:
        plot_exposure(exposure_hist_2d.project_by_value(energy, axis=1), start_time, end_time, nside, outputprefix, plotdir, title=f"Exposure at {energy} GeV", colormap=colormap, transparent=transparent, save_pdf=save_pdf, rotate=rotate)
        if no_title:
            plot_exposure(exposure_hist_2d.project_by_value(energy, axis=1), start_time, end_time, nside, f"{outputprefix}_NoTitle", plotdir, title=None, colormap=colormap, transparent=transparent, save_pdf=save_pdf, rotate=rotate, no_title=no_title)
    else:
        for index, (exposure_hist, min_energy, max_energy) in enumerate(exposure_hist_2d.project_all(axis=1)):
            plot_exposure(exposure_hist, start_time, end_time, nside, f"{outputprefix}_{index}", plotdir, title=f"Exposure, {min_energy:.4g} to {max_energy:.4g} GeV", colormap=colormap, transparent=transparent, save_pdf=save_pdf, rotate=rotate)
            if no_title:
                plot_exposure(exposure_hist, start_time, end_time, nside, f"{outputprefix}_{index}_NoTitle", plotdir, title=None, colormap=colormap, transparent=transparent, save_pdf=save_pdf, rotate=rotate, no_title=no_title)


def calculate_exposure(altitude_time_hist, effective_area_hist, partition_size=65536):
    exposure = np.zeros((altitude_time_hist.values.shape[0], effective_area_hist.values.shape[0]))
    for index in range(0, altitude_time_hist.values.shape[0], partition_size):
        exposure[index:index + partition_size,:] = (altitude_time_hist.values[index:index + partition_size,None,:] * effective_area_hist.values[None,:,:]).sum(axis=2)
    return Histogram(altitude_time_hist.binnings[0], effective_area_hist.binnings[0], values=exposure)


def plot_exposure(exposure_hist, start_time, end_time, nside, outputprefix, plotdir, title="Exposure", rotate=False, no_title=False):
    figure = plt.figure(figsize=(8, 4.2))
    plot = figure.subplots(1, 1, subplot_kw=dict(projection="hammer"))
    if not no_title:
        if title is not None:
            plot.set_title(f"{title}, {datetime.fromtimestamp(start_time, timezone.utc):%Y-%m-%d %H:%M:%S} to {datetime.fromtimestamp(end_time, timezone.utc):%Y-%m-%d %H:%M:%S}", pad=15)
        else:
            plot.set_title(f"{datetime.fromtimestamp(start_time, timezone.utc):%Y-%m-%d %H:%M:%S} to {datetime.fromtimestamp(end_time, timezone.utc):%Y-%m-%d %H:%M:%S}", pad=15)
    theta_pixels = 1800
    phi_pixels = 3600
    theta = np.linspace(np.pi, 0, theta_pixels)
    phi = np.linspace(-np.pi, np.pi, phi_pixels)
    phi, theta = np.meshgrid(phi, theta)

    if rotate is not None and rotate != "Terrestrial":
        r = Rotator(coord=rotate, inv=True)
        theta, phi = r(theta.flatten(), phi.flatten())
        theta = theta.reshape(theta_pixels, phi_pixels)
        phi = phi.reshape(theta_pixels, phi_pixels)

    grid_pix = hp.ang2pix(nside, theta, phi)
    grid_map = exposure_hist.values[grid_pix]
    theta_plot = np.linspace(np.pi, 0, theta_pixels)
    phi_plot = np.linspace(-np.pi, np.pi, phi_pixels)
    mesh = plot.pcolormesh(-phi_plot, -theta_plot + np.pi / 2, grid_map)
    plot.grid(alpha=0.3)
    plt.colorbar(mesh, ax=plot, extend="max", fraction=0.08, label="Exposure / cm²s")

    plot.xaxis.set_major_formatter(ThetaFormatterCounterclockwisePhi())
    plot.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / np.pi * 180:.0f}°"))
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

    plot.tick_params(axis='x', colors='white')

    for label in plot.get_xticklabels():
        label.set_alpha(0.6)
        label.set_color("white")

    plot.text(1, 0, "Equatorial Coordinates" if rotate else "Galactic Coordinates", transform=plot.transAxes, ha="right", va="bottom")

    figure.subplots_adjust(left=0.075, right=0.95, bottom=0.025, top=0.95)
    save_figure(figure, plotdir, f"{outputprefix}", dpi=300, save_pdf=save_pdf, transparent=transparent)


def load_altitude(arg):
    filename, effective_area_2d, nside, outputprefix, plotdir, kwargs = arg

    energy_binning, cos_theta_binning = effective_area_2d.binnings
    healpy_binning = make_healpy_binning(nside=nside)

    time_range = kwargs["time_range"]
    if time_range is not None:
        min_time, max_time = time_range

    plot_individual = kwargs["plot_individual"]
    reference_energy = kwargs["reference_energy"]

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
        if plot_individual:
            exposure_hist = calculate_exposure(altitude_time_hist, effective_area_2d)
            plot_exposures(exposure_hist, start_time, end_time, nside=nside, outputprefix=f"{outputprefix}_{datetime.fromtimestamp(start_time, timezone.utc):%Y-%m-%d}", plotdir=plotdir, energy=reference_energy, rotate=kwargs["rotate"])

    return altitude_time_hist, start_time, end_time


def make_args(filenames, effective_area, nside, outputprefix, plotdir, **kwargs):
    for filename in filenames:
        yield (filename, effective_area, nside, outputprefix, plotdir, kwargs)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--altitude-histograms", required=True, nargs="+", help="Path to NPZ files containing the altitude histograms.")
    parser.add_argument("--effective-area", required=True, help="Path to NPZ file containing the effective area to be integrated.")
    parser.add_argument("--nside", type=int, default=128, help="Number of HEALPix sides to use in the skymap binning.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--outputprefix", default="Exposure", help="Prefix for plots and result files.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--time-range", type=parse_datetime, nargs=2, help="Limit the time range to read in.")
    parser.add_argument("--plot-individual", action="store_true", help="Make plots of the exposure maps of each altitude hist (usually representing a day).")
    parser.add_argument("--reference-energy", type=float, default=10, help="Energy to plot the individual exposure maps at.")
    parser.add_argument("--no-title", action='store_true', help='Use this to prevent titles from being generated.')
    parser.add_argument("--transparent", action="store_true", help='set this to make the background of the png transparent')
    parser.add_argument("--save-pdf", action="store_true", help='Store the Plots as psf.')
    parser.add_argument("--colormap", default="viridis", help='What colormap to use')
    parser.add_argument("--coord", nargs="+", type=str, default=None, choices=['G','C','E', 'Terrestrial'], help="Either one of ‘G’ (Galactic), ‘E’ (Ecliptic) or ‘C’ (Celestial/Equatorial) to describe the coordinate system of the map, or a sequence of 2 of these to rotate the map from the first to the second coordinate system. The standard map is in 'G'.")


    args = parser.parse_args()

    os.makedirs(args.resultdir, exist_ok=True)
    os.makedirs(args.plotdir, exist_ok=True)

    with np.load(args.effective_area) as effective_area_file:
        effective_area_3d = load_histogram(effective_area_file, "effective_area_3d")
        effective_area_2d = load_histogram(effective_area_file, "effective_area_2d")

    energy_binning, cos_theta_binning = effective_area_2d.binnings
    healpy_binning = make_healpy_binning(nside=args.nside)
    global_altitude_time_hist = WeightedHistogram(healpy_binning, cos_theta_binning, labels=("Healpy Index", "cos(theta)"))

    global_start_time = 1000000000000
    global_end_time = 0

    pool_args = make_args(args.altitude_histograms, effective_area_2d, args.nside, args.outputprefix, args.plotdir, time_range=args.time_range, plot_individual=args.plot_individual, reference_energy=args.reference_energy, rotate = args.coord)
    with mp.Pool(args.nprocesses) as pool:
        for return_value in pool.imap_unordered(load_altitude, pool_args):
            if return_value is not None:
                altitude_hist, start_time, end_time = return_value
                global_altitude_time_hist.add(altitude_hist)
                global_start_time = min(global_start_time, start_time)
                global_end_time = max(global_end_time, end_time)

    # altitude is direction * cos(theta), effective area is energy * cos(theta)
    # expand to direction * energy * cos(theta), integrate over cos(theta)
    global_exposure_hist = calculate_exposure(global_altitude_time_hist, effective_area_2d)

    results = {}
    results["start_time"] = start_time
    results["end_time"] = end_time
    global_altitude_time_hist.add_to_file(results, "altitude_time")
    global_exposure_hist.add_to_file(results, "exposure")
    np.savez(os.path.join(args.resultdir, f"{args.outputprefix}_exposure.npz"), **results)

    if global_exposure.sum() > 0:
        plot_exposures(global_exposure_hist, global_start_time, global_end_time, nside=args.nside, outputprefix=f"{args.outputprefix}", colormap=args.colormap, plotdir=args.plotdir, energy=args.reference_energy, rotate=args.coord, no_title=args.no_title, transparent=args.transparent, save_pdf=args.save_pdf)


if __name__ == "__main__":
    main()
