#!/usr/bin/env python3

import os
from glob import glob

import numpy as np
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from scipy.stats import norm as gaussian
from scipy.signal import convolve

from tools.binnings import reduce_bins
import tools.constants as consts
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_2d, load_histograms_from_files
from tools.utilities import plot_2d, round_up, save_figure

Z_LABELS = (
    (consts.ECAL_Z_LOWER, consts.ECAL_Z_UPPER, "ECAL"),
    (consts.RICH_PMT_Z - 5, consts.RICH_PMT_Z + 5, "RICH PMTs"),
    (consts.RICH_PMT_Z + 5, consts.RICH_RADIATOR_Z - 5, "RICH Mirror"),
    (consts.RICH_RADIATOR_Z - 5, consts.RICH_RADIATOR_Z + 5, "RICH Radiator"),
    (63.5, 67, "ToF L1"),
    (60, 63.5, "ToF L2"),
    (-64, -60, "ToF L3"),
    (-67, -64, "ToF L4"),
    (consts.TRK_LAYER_POSITION_Z[1] + 0.5, 59.5, "TAS"),
    (consts.TRD_Z_LOWER, consts.TRD_Z_UPPER, "TRD"),
    (consts.TRK_LAYER_POSITION_Z[0] - 0.5, consts.TRK_LAYER_POSITION_Z[0] + 0.5, "Tracker Layer 1"),
    (consts.TRK_LAYER_POSITION_Z[1] - 0.5, consts.TRK_LAYER_POSITION_Z[1] + 0.5, "Tracker Layer 2"),
    (consts.TRK_LAYER_POSITION_Z[2] - 0.5, consts.TRK_LAYER_POSITION_Z[2] + 0.5, "Tracker Layer 3"),
    (consts.TRK_LAYER_POSITION_Z[3] - 0.5, consts.TRK_LAYER_POSITION_Z[3] + 0.5, "Tracker Layer 4"),
    (consts.TRK_LAYER_POSITION_Z[4] - 0.5, consts.TRK_LAYER_POSITION_Z[4] + 0.5, "Tracker Layer 5"),
    (consts.TRK_LAYER_POSITION_Z[5] - 0.5, consts.TRK_LAYER_POSITION_Z[5] + 0.5, "Tracker Layer 6"),
    (consts.TRK_LAYER_POSITION_Z[6] - 0.5, consts.TRK_LAYER_POSITION_Z[6] + 0.5, "Tracker Layer 7"),
    (consts.TRK_LAYER_POSITION_Z[7] - 0.5, consts.TRK_LAYER_POSITION_Z[7] + 0.5, "Tracker Layer 8"),
    (consts.TRK_LAYER_POSITION_Z[8] - 0.5, consts.TRK_LAYER_POSITION_Z[8] + 0.5, "Tracker Layer 9"),
)

def concatenate_colormaps(cmap_low, cmap_high, crossover):
    if isinstance(cmap_low, str):
        cmap_low = mpl.colormaps.get_cmap(cmap_low)
    if isinstance(cmap_high, str):
        cmap_high = mpl.colormaps.get_cmap(cmap_high)
    assert 0 < crossover < 1
    def _make_channel(channel):
        def _channel(value):
            if isinstance(value, (int, float)):
                value = np.array([value])
            low = value < crossover
            high = ~low
            low_value = value / crossover
            high_value = (value - crossover) / (1 - crossover)
            return (low[:,None] * cmap_low(low_value) + high[:,None] * cmap_high(high_value))[:,channel]
        return _channel
    r = _make_channel(0)
    g = _make_channel(1)
    b = _make_channel(2)
    a = _make_channel(3)
    return mpl.colors.LinearSegmentedColormap(name="concat", segmentdata=dict(red=r, green=g, blue=b, alpha=a))


def gaussian_smooth(hist, sigmas, n_sigmas=10):
    if isinstance(sigmas, (float, int)):
        sigmas = np.ones(len(hist.binnings)) * sigmas
    assert len(hist.binnings) == len(sigmas)
    bin_widths = np.array([binning.edges[2] - binning.edges[1] for binning in hist.binnings])
    n_kernel_bins = np.round(2 * n_sigmas * np.array(sigmas) / bin_widths).astype(np.int32)
    kernel_binnings = [np.arange(-n_bins / 2 * bin_width, (n_bins / 2 + 1) * bin_width, bin_width) for (n_bins, bin_width) in zip(n_kernel_bins, bin_widths)]
    kernel = np.ones(n_kernel_bins)
    for dimension, (bin_edges, sigma) in enumerate(zip(kernel_binnings, sigmas)):
        norm = gaussian(loc=0, scale=sigma)
        probabilities = np.diff(norm.cdf(bin_edges))
        kernel *= probabilities[tuple([slice(None) if axis == dimension else None for axis in range(hist.dimensions)])]
    smoothed_values = convolve(hist.values, kernel, mode="same")
    hist.values = smoothed_values
    return smoothed_values

def smooth_layers(z_values, hists, sigma, n_sigmas=5):
    stepsize = z_values[1] - z_values[0]
    steps = int(n_sigmas * sigma / stepsize)

    new_hists = []
    for index, (z, hist) in enumerate(zip(z_values, hists)):
        sel = slice(max(index - steps, 0), min(index + steps, len(hists)))
        zs = z_values[sel]
        hs = hists[sel]
        edges = np.concatenate((zs - stepsize / 2, [zs[-1] + stepsize / 2]))
        norm = gaussian(loc=z, scale=sigma)
        prob = np.diff(norm.cdf(edges))
        new_values = np.zeros_like(hist.values)
        for h, p in zip(hs, prob):
            new_values += p * h.values
        new_hists.append(Histogram(*hist.binnings, values=new_values, labels=hist.labels))
    return new_hists


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--iss-files", nargs="+", help="Files with ISS histograms to draw.")
    parser.add_argument("--mc-files", nargs="+", help="Files with MC histograms to draw.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--outputprefix", default="VertexMap", help="Prefix for plots and result files.")
    parser.add_argument("--title", default="Vertices", help="Title to put on each plot.")
    parser.add_argument("--iss-title", default="ISS", help="Title to put on each ISS plot.")
    parser.add_argument("--mc-title", default="MC", help="Title to put on each MC plot.")
    parser.add_argument("--rebin", type=int, help="Reduce resolution by rebinning by this factor.")
    parser.add_argument("--hist-name", default="hist", help="Name of histogram to load.")
    parser.add_argument("--smooth", type=float, help="Gaussian width to smooth entries by.")
    parser.add_argument("--smooth-z", type=float, help="Gaussian width to smooth layers by.")
    parser.add_argument("--nmax", type=float, help="Maximum of the color scale.")
    parser.add_argument("--zrange", type=float, nargs=2, help="Minimum and maximum z value to plot.")
    parser.add_argument("--width", type=float, help="Maximum X and Y value to show.")

    args = parser.parse_args()

    os.makedirs(args.plotdir, exist_ok=True)

    print("loading histograms")
    iss_hists = load_histograms_from_files(args.iss_files)
    mc_hists = load_histograms_from_files(args.mc_files)

    print("loading z positions")
    z_slices = None
    for filenames in (args.iss_files, args.mc_files):
        for filename in filenames:
            with np.load(filename) as np_file:
                if z_slices is None:
                    z_slices = np_file["z"]
                else:
                    assert np.all(z_slices == np_file["z"])

    iss_slice_hists = []
    mc_slice_hists = []

    iss_z_sum = []
    mc_z_sum = []
    used_z_slices = []

    #cmap_gb = mpl.colors.LinearSegmentedColormap(
    cmap = concatenate_colormaps("gist_yarg", "hot", 0.1)
    cmap_ratio = "seismic"

    print("rebinning and smoothing")
    for z in z_slices:
        if args.zrange:
            if z < args.zrange[0] or z > args.zrange[1]:
                continue

        used_z_slices.append(z)
        for hists, slice_hists, z_sum in ((iss_hists, iss_slice_hists, iss_z_sum), (mc_hists, mc_slice_hists, mc_z_sum)):
            hist = hists[f"{args.hist_name}_{z:.2f}"]
            if args.rebin is not None:
                x_binning = reduce_bins(hist.binnings[0], args.rebin)
                y_binning = reduce_bins(hist.binnings[1], args.rebin)
                hist = hist.rebin(x_binning, y_binning)
            if args.smooth is not None:
                gaussian_smooth(hist, args.smooth)
            slice_hists.append(hist)
            z_sum.append(hist.values.sum())

    iss_z_sum, mc_z_sum = map(np.array, (iss_z_sum, mc_z_sum))
    iss_norm = iss_z_sum.sum()
    mc_norm = mc_z_sum.sum()

    print("smoothing z")
    if args.smooth_z:
        iss_slice_hists = smooth_layers(used_z_slices, iss_slice_hists, sigma=args.smooth_z)
        mc_slice_hists = smooth_layers(used_z_slices, mc_slice_hists, sigma=args.smooth_z)

    print("plotting z")
    z_figure = plt.figure(figsize=(8, 6.2))
    z_plot = z_figure.subplots(1, 1)
    z_plot.set_title(args.title)
    z_plot.plot(used_z_slices, iss_z_sum / iss_norm, "-", label=args.iss_title)
    z_plot.plot(used_z_slices, mc_z_sum / mc_norm, "-", label=args.mc_title)
    z_plot.set_ylim(bottom=0)
    z_plot.set_xlabel("z / cm")
    z_plot.set_ylabel("Normalized Vertices")
    z_plot.legend()
    z_figure.subplots_adjust(left=0.15, right=0.95, bottom=0.12, top=0.9)
    save_figure(z_figure, args.plotdir, f"{args.outputprefix}_z")


    iss_n_max = max([np.max(h.values) for h in iss_slice_hists]) if args.nmax is None else args.nmax
    mc_n_max = max([np.max(h.values) for h in mc_slice_hists]) if args.nmax is None else args.nmax
    iss_color_max = round_up(iss_n_max, log=True)
    mc_color_max = round_up(mc_n_max, log=True)

    #print("saving data")
    ##figure_3d = plt.figure(figsize=(8, 6))
    ##plot_3d = figure_3d.add_subplot(projection="3d")
    #os.makedirs(args.resultdir, exist_ok=True)
    #for slice_hists, prefix, n_max in ((iss_slice_hists, "iss", iss_n_max), (mc_slice_hists, "mc", mc_n_max)):
    #    all_x, all_y, all_z, all_transparency = [], [], [], []
    #    for z, hist in zip(used_z_slices, slice_hists):
    #        x = hist.binnings[0].bin_centers
    #        y = hist.binnings[1].bin_centers
    #        x, y = map(lambda a: a.flatten(), np.meshgrid(x, y))
    #        transparency = (np.maximum(hist.values, 0) / n_max).flatten()
    #        sel = transparency > 0
    #        x, y, transparency = map(lambda a: a[sel], (x, y, transparency))
    #        color = np.zeros_like(x)
    #        rgba = np.stack((color, color, color, transparency), axis=1)
    #        z = np.full_like(x, z)
    #        all_x.append(x)
    #        all_y.append(y)
    #        all_z.append(z)
    #        all_transparency.append(transparency)
    #        #plot_3d.scatter(x, y, z, c=rgba, s=1)
    #    #save_figure(figure_3d, args.plotdir, f"{args.outputprefix}_3d", save_pickle=True)
    #    all_x, all_y, all_z, all_transparency = map(np.concatenate, (all_x, all_y, all_z, all_transparency))
    #    np.savez_compressed(os.path.join(args.resultdir, f"{args.outputprefix}_{prefix}_points.npz"), x=all_x, y=all_y, z=all_z, amplitude=all_transparency)


    print("making plots")
    for index, (z, iss_hist, mc_hist) in enumerate(zip(used_z_slices, iss_slice_hists, mc_slice_hists)):

        labels = []
        for z_min, z_max, label in Z_LABELS:
            if z >= z_min and z <= z_max:
                labels.append(label)

        triple_figure = plt.figure(figsize=(16, 8))
        gs = GridSpec(nrows=4, ncols=5, left=0.025, right=0.975, bottom=0.05, top=1, wspace=0, hspace=0, width_ratios=(5, 0.5, 5, 0.5, 5), height_ratios=(2, 5, 0.5, 0.5))
        iss_plot = triple_figure.add_subplot(gs[1, 0])
        ratio_plot = triple_figure.add_subplot(gs[1, 2])
        mc_plot = triple_figure.add_subplot(gs[1, 4])
        z_plot = triple_figure.add_subplot(gs[0, 0])
        iss_colorbar_plot = triple_figure.add_subplot(gs[3, 0])
        ratio_colorbar_plot = triple_figure.add_subplot(gs[3, 2])
        mc_colorbar_plot = triple_figure.add_subplot(gs[3, 4])

        iss_normed = np.maximum(iss_hist.values / iss_norm, 0)
        mc_normed = np.maximum(mc_hist.values / mc_norm, 0)
        ratio = (iss_normed - mc_normed) / (iss_normed + mc_normed)
        ratio_hist = Histogram(*iss_hist.binnings, values=ratio, labels=iss_hist.labels)

        iss_plot.text(0, 0, f"Z = {z:.2f} cm", transform=iss_plot.transAxes, ha="left", va="bottom")
        mc_plot.text(1, 0, ", ".join(labels), transform=mc_plot.transAxes, ha="right", va="bottom")
        iss_plot.text(0.5, 0.99, args.iss_title, transform=iss_plot.transAxes, ha="center", va="top")
        mc_plot.text(0.5, 0.99, args.mc_title, transform=mc_plot.transAxes, ha="center", va="top")
        #ratio_plot.text(0.5, 0.99, "Ratio", transform=ratio_plot.transAxes, ha="center", va="top")
        ratio_plot.set_title("(ISS-MC)/(ISS+MC)")
        plot_histogram_2d(iss_plot, iss_hist, show_overflow=False, mask_zeros=True, min_value=0, max_value=iss_color_max, cmap=cmap, colorbar_ax=iss_colorbar_plot, colorbar_orientation="horizontal")
        plot_histogram_2d(mc_plot, mc_hist, show_overflow=False, mask_zeros=True, min_value=0, max_value=mc_color_max, cmap=cmap, colorbar_ax=mc_colorbar_plot, colorbar_orientation="horizontal")
        plot_histogram_2d(ratio_plot, ratio_hist, show_overflow=False, mask_zeros=False, min_value=-1, max_value=1, cmap=cmap_ratio, colorbar_ax=ratio_colorbar_plot, colorbar_orientation="horizontal")

        z_plot.plot(used_z_slices, iss_z_sum / iss_norm, "-", color="tab:blue")
        z_plot.plot(used_z_slices, mc_z_sum / mc_norm, "-", color="tab:orange")
        z_plot.set_ylim(bottom=0)
        z_plot.axvline(z, alpha=0.8, color="gray")

        iss_plot.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        mc_plot.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        ratio_plot.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        z_plot.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

        if args.width:
            iss_plot.set_xlim(-args.width, args.width)
            iss_plot.set_ylim(-args.width, args.width)
            mc_plot.set_xlim(-args.width, args.width)
            mc_plot.set_ylim(-args.width, args.width)
            ratio_plot.set_xlim(-args.width, args.width)
            ratio_plot.set_ylim(-args.width, args.width)
        #triple_figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        save_figure(triple_figure, args.plotdir, f"{args.outputprefix}_triple_{index:0>5}", dpi=300)


        #figure = plt.figure(figsize=(8, 6))
        #plot = figure.subplots(1, 1)
        #figure.suptitle(f"Z = {z:.2f} cm")
        #plot.text(1, 0, ", ".join(labels), transform=plot.transAxes, ha="right", va="bottom")
        #plot.text(0.5, 0.99, args.title, transform=plot.transAxes, ha="center", va="top")
        #plot_histogram_2d(plot, hist, show_overflow=False, mask_zeros=False, min_value=0, max_value=color_max, cmap=cmap)
        #plot_z(figure, z)
        #if args.width:
        #    plot.set_xlim(-args.width, args.width)
        #    plot.set_ylim(-args.width, args.width)
        #save_figure(figure, args.plotdir, f"{args.outputprefix}_lin_{z:.2f}", dpi=300)

        #figure = plt.figure(figsize=(8, 6))
        #plot = figure.subplots(1, 1)
        #figure.suptitle(f"Z = {z:.2f} cm")
        #plot.text(1, 0, ", ".join(labels), transform=plot.transAxes, ha="right", va="bottom")
        #plot.text(0.5, 0.99, args.title, transform=plot.transAxes, ha="center", va="top")
        #plot_histogram_2d(plot, hist, show_overflow=False, mask_zeros=False, log=True, min_value=5e-2, max_value=color_max, cmap=cmap)
        #plot_z(figure, z)
        #if args.width:
        #    plot.set_xlim(-args.width, args.width)
        #    plot.set_ylim(-args.width, args.width)
        #save_figure(figure, args.plotdir, f"{args.outputprefix}_log_{z:.2f}", dpi=300)

        #figure = plt.figure(figsize=(8, 6))
        #plot = figure.subplots(1, 1)
        #figure.suptitle(f"Z = {z:.2f} cm")
        #plot.text(1, 0, ", ".join(labels), transform=plot.transAxes, ha="right", va="bottom")
        #plot.text(0.5, 0.99, args.title, transform=plot.transAxes, ha="center", va="top")
        #plot_histogram_2d(plot, hist, show_overflow=False, mask_zeros=False, min_value=0, cmap=cmap)
        #plot_z(figure, z)
        #if args.width:
        #    plot.set_xlim(-args.width, args.width)
        #    plot.set_ylim(-args.width, args.width)
        #save_figure(figure, args.plotdir, f"{args.outputprefix}_noscale_{z:.2f}", dpi=300)

        #clean_figure = plt.figure(figsize=(5, 5))
        #clean_plot = clean_figure.subplots(1, 1)
        #clean_plot.text(0, 0, f"Z = {z:.2f} cm", transform=clean_plot.transAxes, ha="left", va="bottom")
        #clean_plot.text(1, 0, ", ".join(labels), transform=clean_plot.transAxes, ha="right", va="bottom")
        #clean_plot.text(0.5, 0.99, args.title, transform=clean_plot.transAxes, ha="center", va="top")
        #plot_histogram_2d(clean_plot, hist, show_overflow=False, mask_zeros=True, min_value=0, max_value=color_max, cmap=cmap, colorbar=False)
        #plot_z(clean_figure, z)
        #if args.width:
        #    clean_plot.set_xlim(-args.width, args.width)
        #    clean_plot.set_ylim(-args.width, args.width)
        #clean_figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        #save_figure(clean_figure, args.plotdir, f"{args.outputprefix}_clean_lin_{index:0>5}", dpi=300)

        #clean_figure = plt.figure(figsize=(5, 5))
        #clean_plot = clean_figure.subplots(1, 1)
        #clean_plot.text(0, 0, f"Z = {z:.2f} cm", transform=clean_plot.transAxes, ha="left", va="bottom")
        #clean_plot.text(1, 0, ", ".join(labels), transform=clean_plot.transAxes, ha="right", va="bottom")
        #clean_plot.text(0.5, 0.99, args.title, transform=clean_plot.transAxes, ha="center", va="top")
        #plot_histogram_2d(clean_plot, hist, show_overflow=False, mask_zeros=True, log=True, min_value=5e-2, max_value=color_max, cmap=cmap, colorbar=False)
        #plot_z(clean_figure, z)
        #if args.width:
        #    clean_plot.set_xlim(-args.width, args.width)
        #    clean_plot.set_ylim(-args.width, args.width)
        #clean_figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        #save_figure(clean_figure, args.plotdir, f"{args.outputprefix}_clean_log_{index:0>5}", dpi=300)




if __name__ == "__main__":
    main()
