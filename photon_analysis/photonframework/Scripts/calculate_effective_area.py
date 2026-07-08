#!/usr/bin/env python3

import multiprocessing as mp
import os
import pickle


import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from tools.binnings import make_lin_binning, make_energy_binning_from_config, make_mc_dataset_edge_binning, combine_binnings, reduce_bins, make_flux_energy_binning
from tools.config import get_config
from tools.constants import BastianPlots
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d
from tools.roottree import read_tree
from tools.utilities import load_mc_trigger_count, save_figure, set_energy_ticks, plot_2d
from tools.statistics import hist_mean_and_std


BRANCHES = ["McEnergy", "McTheta", "McPhi"]

def handle_file(arg):
    filenames, treename, chunk_size, rank, nranks, kwargs = arg

    energy_binning = kwargs["energy_binning"]
    cos_theta_binning = kwargs["cos_theta_binning"]
    phi_binning = kwargs["phi_binning"]

    histogram = Histogram(energy_binning, cos_theta_binning, phi_binning)

    for events in read_tree(filenames, treename, rank=rank, nranks=nranks, chunk_size=chunk_size, branches=BRANCHES, cache_file=False):
        cos_theta = -np.cos(events.McTheta)
        histogram.fill(events.McEnergy, cos_theta, events.McPhi)

    return histogram


def make_args(filename, treename, chunk_size, nranks, **kwargs):
    for rank in range(nranks):
        yield (filename, treename, chunk_size, rank, nranks, kwargs)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs=2, required=True, help="Path to config file and working directory.")
    parser.add_argument("--tree", nargs="+", required=True, help="MC Gamma Tree to read.")
    parser.add_argument("--treename", default="GammaTree", help="Name of the tree in the ROOT file to read.")
    parser.add_argument("--dataset", required=True, help="Name of the MC dataset to calculate the effective area for.")
    parser.add_argument("--outputprefix", default="EffectiveArea", help="Prefix for plots and result files.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Number of events to read each step.")
    parser.add_argument("--title", default="Effective Area", help="Title for plots.")
    parser.add_argument("--comparison", choices=list(BastianPlots.keys()), help="Version of bastians data to compare to.")
    parser.add_argument("--no-title", action='store_true', help='Use this to prevent titles from being generated.')
    parser.add_argument("--transparent", action="store_true", help='set this to make the background of the png transparent')
    parser.add_argument("--energy-binning", default='skymap', choices=['skymap', 'flux'], help='Which energy binning should be used')

    args = parser.parse_args()
    config_filename, datadir = args.config
    config = get_config(config_filename)

    os.makedirs(args.resultdir, exist_ok=True)
    os.makedirs(args.plotdir, exist_ok=True)

    ENERGY_BINNINGS = {'skymap': make_energy_binning_from_config(config),
                       'flux': make_flux_energy_binning()}

    energy_binning = ENERGY_BINNINGS[args.energy_binning]
    energy_binning_with_dataset_edges = combine_binnings((energy_binning, make_mc_dataset_edge_binning(config, args.dataset)))
    cos_theta_binning = make_lin_binning(0.7, 1, 150)
    cos_theta_2d_binning = make_lin_binning(0.7, 1, 150)
    phi_binning = make_lin_binning(-np.pi, np.pi, 60)
    
    cos_theta_rebinning = make_lin_binning(0.7, 1, 15)

    comparison = None
    if args.comparison is not None:
        comparison = BastianPlots[args.comparison]["effective area"]

    trigger_count = load_mc_trigger_count(os.path.join(datadir, "data", args.dataset, "triggers", "triggers.txt"), energy_binning_with_dataset_edges)
    event_histogram = Histogram(energy_binning_with_dataset_edges, cos_theta_binning, phi_binning, labels=("E / GeV", "$cos(\\theta)$", "$\\phi$"))

    with mp.Pool(args.nprocesses) as pool:
        pool_args = make_args(args.tree, args.treename, chunk_size=args.chunk_size, nranks=args.nprocesses, energy_binning=energy_binning_with_dataset_edges, cos_theta_binning=cos_theta_binning, phi_binning=phi_binning)
        for hist in pool.imap_unordered(handle_file, pool_args):
            event_histogram.add(hist)

    event_histogram_2d = event_histogram.project_axis(axis=2)
    event_histogram_2d_rebin = event_histogram_2d.rebin(energy_binning_with_dataset_edges, cos_theta_rebinning, method="sum")
    event_histogram_2d = event_histogram_2d.rebin(energy_binning_with_dataset_edges, cos_theta_2d_binning, method="sum")
    event_histogram = event_histogram.rebin(energy_binning_with_dataset_edges, cos_theta_2d_binning, phi_binning, method='sum')

    event_histogram_1d = event_histogram.project_axis(axis=2).project_axis(axis=1)

    bin_size_cos_theta = cos_theta_2d_binning.edges[2] - cos_theta_2d_binning.edges[1]
    bin_size_cos_theta_rb = cos_theta_rebinning.edges[2] - cos_theta_rebinning.edges[1]
    bin_size_phi = phi_binning.edges[2] - phi_binning.edges[1]

    effective_area_3d = 3.9**2 * np.pi * event_histogram.values[1:-1,1:-1,1:-1] / (trigger_count.values[1:-1,None,None] * bin_size_cos_theta * bin_size_phi)
    effective_area_2d = 3.9**2 / 2 * event_histogram_2d.values[1:-1,1:-1] / (trigger_count.values[1:-1,None] * bin_size_cos_theta)

    effective_area_3d_hist = event_histogram * (1e4 * 3.9**2 * np.pi / (trigger_count.values[:,None,None] * bin_size_cos_theta * bin_size_phi))
    effective_area_2d_hist = event_histogram_2d * (1e4 * 3.9**2 / (2 * trigger_count.values[:,None] * bin_size_cos_theta))
    effective_area_2d_hist_rb = event_histogram_2d_rebin * (1e4 * 3.9**2 / (2 * trigger_count.values[:,None] * bin_size_cos_theta_rb))
    effective_acceptence_1d_hist = event_histogram_1d * (1e4 * 3.9**2 * np.pi / (trigger_count.values[:]))

    effective_area_3d_hist = effective_area_3d_hist.rebin(energy_binning, cos_theta_2d_binning, phi_binning, method="mean")
    effective_area_2d_hist = effective_area_2d_hist.rebin(energy_binning, cos_theta_2d_binning, method="mean")
    effective_acceptence_1d_hist = effective_acceptence_1d_hist.rebin(energy_binning, method = "mean")
    effective_area_2d_hist_rb = effective_area_2d_hist_rb.rebin(energy_binning, cos_theta_rebinning, method="mean")



    theta_phi_hist = event_histogram.project_axis(axis=0)
    phi_correction = 2 * np.pi / bin_size_phi * theta_phi_hist.values[1:-1,1:-1] / np.sum(theta_phi_hist.values[1:-1,1:-1], axis=1)[:,None]

    print(type(phi_correction))
    results = {}
    effective_area_3d_hist.add_to_file(results, "effective_area_3d")
    effective_area_2d_hist.add_to_file(results, "effective_area_2d")
    effective_area_2d_hist_rb.add_to_file(results, "effective_area_2d_rb")
    effective_acceptence_1d_hist.add_to_file(results, 'effective_acceptance_1d')

    area_figure = plt.figure(figsize=(8, 4.2))
    area_plot = area_figure.subplots(1, 1)
    plot_histogram_2d(area_plot, effective_area_2d_hist, show_overflow=False, label="Effective Area / $cm^2$", min_value=0, cmap="jet")
    set_energy_ticks(area_plot)
    area_figure.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
    if args.no_title:
        save_figure(area_figure, args.plotdir, f"{args.outputprefix}_2d_NoTitle", transparent=args.transparent)
    area_plot.set_title(args.title)
    save_figure(area_figure, args.plotdir, f"{args.outputprefix}_2d", transparent=args.transparent)

    phi_correction_figure = plt.figure(figsize=(8, 4.2))
    phi_correction_plot = phi_correction_figure.subplots(1, 1)
    plot_2d(phi_correction_plot, phi_correction, cos_theta_2d_binning.edges[1:-1], phi_binning.edges[1:-1], min_value=0, max_value=2, colorbar_label="Correction Factor", cmap = 'jet')
    phi_correction_plot.set_xlabel("$cos\\theta$")
    phi_correction_plot.set_ylabel("$\\phi$")
    results["phi_correction"] = phi_correction
    phi_correction_figure.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
    if args.no_title:
        save_figure(phi_correction_figure, args.plotdir, f"{args.outputprefix}_phi_correction_NoTitle", transparent=args.transparent)
    phi_correction_plot.set_title(f"{args.title}, $\\phi$ correction factor")
    save_figure(phi_correction_figure, args.plotdir, f"{args.outputprefix}_phi_correction", transparent=args.transparent)

    np.savez_compressed(os.path.join(args.resultdir, f"{args.outputprefix}.npz"), **results)

    
    figure_acceptence_1d = plt.figure(figsize=(8,4.2))
    plot_acceptence_1d = figure_acceptence_1d.subplots(1,1)
    plot_histogram_1d(plot_acceptence_1d, effective_acceptence_1d_hist, show_overflow=False, label_y="Effective Acceptance / $cm^2 sr$", label="This work")
    plot_acceptence_1d.set_ylim(bottom=0)
    plot_acceptence_1d.set_xlim(energy_binning.edges[1], energy_binning.edges[-2])
    if comparison is not None:
        plot_acceptence_1d.plot(comparison["acceptance"]["x"], comparison["acceptance"]["y"], "k--", label = "B. Beischer")
        plot_acceptence_1d.set_ylim(*comparison["acceptance"]["ylim"])
    plot_acceptence_1d.legend()
    if args.no_title:
        save_figure(figure_acceptence_1d, args.plotdir, f"{args.outputprefix}_Effective_Acceptance_NoTitle", transparent=args.transparent)
    plot_acceptence_1d.set_title(f'{args.title}, Effective Acceptance')
    save_figure(figure_acceptence_1d, args.plotdir, f"{args.outputprefix}_Effective_Acceptance", transparent=args.transparent)


    for index, (area_hist_1d, energy_min, energy_max) in enumerate(effective_area_2d_hist_rb.project_all(axis=0)):
        figure_1d = plt.figure(figsize=(8, 4.2))
        plot_1d = figure_1d.subplots(1, 1)
        plot_histogram_1d(plot_1d, area_hist_1d, show_overflow=False, label_y="Effective Area / $cm^2$", label="This work")
        
        plot_1d.set_ylim(bottom=0)
        plot_1d.set_xlim(cos_theta_rebinning.edges[1], cos_theta_rebinning.edges[-2])
        if comparison is not None:
            if energy_min <= 1 and energy_max >= 1:
                plot_1d.plot(comparison["1GeV"]["x"], comparison["1GeV"]["y"], "k--", label="B. Beischer")
                plot_1d.set_ylim(*comparison["1GeV"]["ylim"])
        plot_1d.legend()
        if args.no_title:
            save_figure(figure_1d, args.plotdir, f"{args.outputprefix}_per_cos_theta_{index+1}_rebinned_NoTitle", transparent=args.transparent)
        plot_1d.set_title(f"{args.title}, ${energy_min:.2f}<= E/GeV<{energy_max:.2f}$")
        save_figure(figure_1d, args.plotdir, f"{args.outputprefix}_per_cos_theta_{index+1}_rebinned", transparent=args.transparent)


    for index, ((area_hist_1d, energy_min, energy_max), (area_hist_2d, _, _)) in enumerate(zip(effective_area_2d_hist.project_all(axis=0), effective_area_3d_hist.project_all(axis=0))):
        figure_1d = plt.figure(figsize=(8, 4.2))
        plot_1d = figure_1d.subplots(1, 1)
        plot_histogram_1d(plot_1d, area_hist_1d, show_overflow=False, label_y="Effective Area / $cm^2$", label="This work")
        plot_1d.set_ylim(bottom=0)
        plot_1d.set_xlim(cos_theta_2d_binning.edges[1], cos_theta_2d_binning.edges[-2])
        if comparison is not None:
            if energy_min <= 1 and energy_max >= 1:
                plot_1d.plot(comparison["1GeV"]["x"], comparison["1GeV"]["y"], "k--", label="B. Beischer")
                plot_1d.set_ylim(*comparison["1GeV"]["ylim"])
        plot_1d.legend()
        if args.no_title:
            save_figure(figure_1d, args.plotdir, f"{args.outputprefix}_per_cos_theta_{index+1}_NoTitle", transparent=args.transparent)
        plot_1d.set_title(f"{args.title}, ${energy_min:.2f}<= E/GeV<{energy_max:.2f}$")
        save_figure(figure_1d, args.plotdir, f"{args.outputprefix}_per_cos_theta_{index+1}", transparent=args.transparent)

        figure_2d = plt.figure(figsize=(8, 4.2))
        plot_theta_phi = figure_2d.subplots(1, 1)
        plot_histogram_2d(plot_theta_phi, area_hist_2d, transpose=True, show_overflow=False, label="Effective Area / $cm^2$", cmap="jet")
        figure_2d.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
        if args.no_title:
            save_figure(figure_2d, args.plotdir, f"{args.outputprefix}_per_cos_theta_phi_{index+1}_NoTitle", transparent=args.transparent)
        plot_theta_phi.set_title(f"{args.title}, ${energy_min:.2f}<= E/GeV<{energy_max:.2f}$")
        save_figure(figure_2d, args.plotdir, f"{args.outputprefix}_per_cos_theta_phi_{index+1}", transparent=args.transparent)

        figure_2d_polar = plt.figure(figsize=(7, 6))
        plot_theta_phi = figure_2d_polar.subplots(1, 1, subplot_kw=dict(projection="polar"))
        values = area_hist_2d.values[::-1,:].transpose()
        edges_x = area_hist_2d.binnings[1].edges
        edges_y = 1 - area_hist_2d.binnings[0].edges[::-1]
        cos_ticks = np.arange(0, 40, 10)  # Larger intervals (e.g., 0, 10, 20, ..., 50 degrees)
        cos_theta_ticks = 1 - np.cos(cos_ticks / 180 * np.pi)
        cos_theta_tick_labels = [f"{tick:.0f}°" for tick in cos_ticks]
        plot_2d(plot_theta_phi, values[1:-1,1:-1], edges_x[1:-1], edges_y[1:-1], colorbar_label="Effective Area / $cm^2$", cmap="jet", min_value=0)
        plot_theta_phi.set_rticks(cos_theta_ticks, cos_theta_tick_labels)
        plot_theta_phi.set_rticks([0.2, 0.4, 0.6, 0.8], ["10","20","30","40"])
        plot_theta_phi.set_rlim(0, 1 - np.cos(30 / 180 * np.pi))
        #plot_histogram_2d(plot_theta_phi, area_hist_2d, transpose=True, show_overflow=False, label="Effective Area / $cm^2$")
        figure_2d_polar.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.85)
        if args.no_title:
            save_figure(figure_2d_polar, args.plotdir, f"{args.outputprefix}_per_cos_theta_phi_{index+1}_polar_NoTitle", transparent=args.transparent)
        plot_theta_phi.set_title(f"{args.title}, ${energy_min:.2f}<= E/GeV<{energy_max:.2f}$")
        save_figure(figure_2d_polar, args.plotdir, f"{args.outputprefix}_per_cos_theta_phi_{index+1}_polar", transparent=args.transparent)

    for index, (area_hist_1d, cos_theta_min, cos_theta_max) in enumerate(effective_area_2d_hist_rb.project_all(axis=1)):
        figure_1d = plt.figure(figsize=(8, 4.2))
        plot_1d = figure_1d.subplots(1, 1)
        plot_histogram_1d(plot_1d, area_hist_1d, show_overflow=False, label_y="Effective Area / $cm^2$", label="This work")
        plot_1d.set_ylim(bottom=0)
        plot_1d.set_xlim(energy_binning.edges[1], energy_binning.edges[-2])
        if comparison is not None:
            if cos_theta_min <= 1 and cos_theta_max >= 1:
                plot_1d.plot(comparison["perpendicular"]["x"], comparison["perpendicular"]["y"], "k--", label="B. Beischer")
        plot_1d.legend()
        if args.no_title:
            save_figure(figure_1d, args.plotdir, f"{args.outputprefix}_per_energy_{index+1}_rebinned_NoTitle", transparent=args.transparent)
        plot_1d.set_title(f"{args.title}, ${cos_theta_min:.3f}<= cos\\theta<{cos_theta_max:.3f}$")
        save_figure(figure_1d, args.plotdir, f"{args.outputprefix}_per_energy_{index+1}_rebinned", transparent=args.transparent)



    for index, ((area_hist_1d, cos_theta_min, cos_theta_max), (area_hist_2d, _, _)) in enumerate(zip(effective_area_2d_hist.project_all(axis=1), effective_area_3d_hist.project_all(axis=1))):
        figure_1d = plt.figure(figsize=(10, 6))
        plot_1d = figure_1d.subplots(1, 1)
        plot_histogram_1d(plot_1d, area_hist_1d, show_overflow=False, label_y="Effective Area / $cm^2$", label="This work")
        plot_1d.set_ylim(bottom=0)
        plot_1d.set_xlim(energy_binning.edges[1], energy_binning.edges[-2])
        if comparison is not None:
            if cos_theta_min <= 1 and cos_theta_max >= 1:
                plot_1d.plot(comparison["perpendicular"]["x"], comparison["perpendicular"]["y"], "k--", label="B. Beischer")
        plot_1d.legend()
        if args.no_title:
            save_figure(figure_1d, args.plotdir, f"{args.outputprefix}_per_energy_{index+1}_NoTitle", transparent=args.transparent)
        plot_1d.set_title(f"{args.title}, ${cos_theta_min:.3f}<= cos\\theta<{cos_theta_max:.3f}$")
        save_figure(figure_1d, args.plotdir, f"{args.outputprefix}_per_energy_{index+1}", transparent=args.transparent)

        figure_2d = plt.figure(figsize=(8, 4.2))
        plot_energy_phi = figure_2d.subplots(1, 1)
        plot_histogram_2d(plot_energy_phi, area_hist_2d, show_overflow=False, label="Effective Area / $cm^2$", cmap = "jet")
        figure_2d.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
        if args.no_title:
            save_figure(figure_2d, args.plotdir, f"{args.outputprefix}_per_energy_phi_{index+1}_NoTitle", transparent=args.transparent)
        plot_energy_phi.set_title(f"{args.title}, ${cos_theta_min:.3f}<= cos\\theta<{cos_theta_max:.3f}$")
        save_figure(figure_2d, args.plotdir, f"{args.outputprefix}_per_energy_phi_{index+1}", transparent=args.transparent)



if __name__ == "__main__":
    main()
