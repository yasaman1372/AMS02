#!/usr/bin/env python3

import os

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from iminuit import Minuit
from iminuit.cost import LeastSquares

from tools.binnings import make_energy_binning#make_lin_binning, make_energy_binning, make_mc_dataset_edge_binning, combine_binnings, reduce_bins
from tools.config import get_config
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d, load_histograms_from_files
from tools.roottree import read_tree
from tools.utilities import load_mc_trigger_count, save_figure, set_energy_ticks, plot_2d
from tools.statistics import hist_mean_and_std, hist_percentile

from tools.constants import BastianVertexPlots as BVP
BVPEA = BVP["effective area"]
BVPER = BVP["energy resolution"]
BVPPSF = BVP["psf"]

import argparse

plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.minor.visible'] = True

def resolution_parametrization(energy, a, b, c):
    return a / (energy + b) + c

def calculate_resolution_from_percentile(hist_2d, confidence_level=0.6827, plotdir="plots", outputprefix="Percentile"):
    low_edge = hist_percentile(hist_2d, axis=1, percentile=0.5 - confidence_level / 2, bin_point="low", interpolate=True)
    high_edge = hist_percentile(hist_2d, axis=1, percentile=0.5 + confidence_level / 2, bin_point="high", interpolate=True)
    return (high_edge[1:-1] - low_edge[1:-1]) / 2, None


def fit_resolution_parametrization(energy_values, resolution_values, resolution_errors=None, min_energy=None, max_energy=None):

    if resolution_errors is None:
        resolution_errors = np.ones_like(resolution_values)
    mask = np.ones_like(energy_values, dtype=bool)
    if min_energy is not None:
        mask = mask & (energy_values >= min_energy)
    if max_energy is not None:
        mask = mask & (energy_values <= max_energy)
    cost = LeastSquares(energy_values[mask], resolution_values[mask], resolution_errors[mask], resolution_parametrization)

    m = Minuit(cost, a=1, b=0.5, c=0.1)
    m.limits["a"] = (0, None)
    m.limits["b"] = (0, None)
    m.limits["c"] = (0, None)
    m.migrad()
    if m.valid:
        m.hesse()
    print(m)
    parameter_dict = dict(zip(m.parameters, m.values))
    parameter_error_dict = dict(zip(m.parameters, m.errors))
    return parameter_dict, parameter_error_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overallinputdir", default=None, help="If set this will be added infront of all input directories. Else the complied path should be provieded in the individual input dirs")
    parser.add_argument("--effareafile", default=None, help="File with the effective area histograms")
    parser.add_argument("--energyfile", default=None, help="File with the energy resolution histograms")
    parser.add_argument("--psffile", default=None, help="File with the psf histograms")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument('--outputprefix', default='Comparison', help='Prefix for plots and result files.')
    parser.add_argument("--no-title", action='store_true', help='Use this to prevent titles from being generated.')
    parser.add_argument("--transparent", action="store_true", help='set this to make the background of the png transparent')

    args = parser.parse_args()
    
    os.makedirs(args.plotdir, exist_ok=True)

    if args.effareafile == None and args.energyfile == None and args.psffile == None:
        raise ValueError('At least one of effareafile, energyfile, or psffile musst be set.')

    Effareafile = None
    Energyresolutionfile = None
    PSFfile = None
    EA = False
    ER = False
    PSF = False
    if args.overallinputdir != None:
        if args.effareafile != None:
            Effareafile = os.path.join(args.overallinputdir, args.effareafile)
            EA = True
        if args.energyfile != None:
            Energyresolutionfile = os.path.join(args.overallinputdir, args.energyfile)
            ER = True
        if args.psffile != None:
            PSFfile = os.path.join(args.overallinputdir, args.psffile)
            PSF = True          
    else:
        if args.effareafile != None:
            Effareafile = args.effareafile
            EA = True
        if args.energyfile != None:
            Energyresolutionfile = args.energyfile
            ER = True
        if args.psffile != None:
            PSFfile = args.psffile
            PSF = True

    if EA:
        EA_histograms = load_histograms_from_files([Effareafile])

    if ER: 
        ER_histograms = load_histograms_from_files([Energyresolutionfile])

    if PSF:
        PSF_histograms = load_histograms_from_files([PSFfile])


    if EA:

        EA_3d_hist = EA_histograms['effective_area_3d']
        EA_2d_hist = EA_histograms['effective_area_2d']
        EA_2d_hist_rb = EA_histograms['effective_area_2d_rb']
        EA_1d_acceptance_hist = EA_histograms['effective_acceptance_1d']
        area_figure = plt.figure(figsize=(5.5, 4.2))
        area_plot = area_figure.subplots(1, 1)
        plot_histogram_2d(area_plot, EA_2d_hist, show_overflow=False, label="Effective Area / $cm^2$ $sr$", cmap = "jet")
        set_energy_ticks(area_plot)
        area_plot.set_box_aspect(1)
        area_figure.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
        if args.no_title:
            save_figure(area_figure, args.plotdir, f"{args.outputprefix}_Effectiv_Area_NoTitle", transparent=args.transparent)
        area_plot.set_title("Effective Area")
        save_figure(area_figure, args.plotdir, f"{args.outputprefix}_Effectiv_Area", transparent=args.transparent)

        area_figure = plt.figure(figsize=(8, 4.2))
        area_plot = area_figure.subplots(1, 1)
        plot_histogram_2d(area_plot, EA_2d_hist, show_overflow=False, label="Effective Area / $cm^2$", cmap = "jet")
        set_energy_ticks(area_plot)
        area_figure.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
        if args.no_title:
            save_figure(area_figure, args.plotdir, f"{args.outputprefix}_Effectiv_Area_no_square_NoTitle", transparent=args.transparent)
        area_plot.set_title("Effective Area")
        save_figure(area_figure, args.plotdir, f"{args.outputprefix}_Effectiv_Area_no_square", transparent=args.transparent)


        area_figure = plt.figure(figsize=(5.5, 4.2))
        area_plot = area_figure.subplots(1, 1)
        cbar = plot_histogram_2d(area_plot, EA_2d_hist, show_overflow=False, label="Effective Area / $cm^2$", cmap = "jet", min_value = BVPEA["2D"]["colorbar"][0], max_value = BVPEA["2D"]["colorbar"][1])
        set_energy_ticks(area_plot)
        area_plot.set_box_aspect(1)
        cbar.set_ticks([0,20,40,60,80,100,120,140,160,180])
        area_figure.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
        if args.no_title:
            save_figure(area_figure, args.plotdir, f"{args.outputprefix}_Effectiv_Area_BastianZ_NoTitle", transparent=args.transparent)
        area_plot.set_title("Effective Area")
        save_figure(area_figure, args.plotdir, f"{args.outputprefix}_Effectiv_Area_BastianZ", transparent=args.transparent)

        acceptance_fig = plt.figure(figsize=(8, 4.2))
        acceptance_plot = acceptance_fig.subplots(1,1)
        plot_histogram_1d(acceptance_plot, EA_1d_acceptance_hist, show_overflow=False, label_y="Effective Acceptance / $cm^2 sr$", label='This Analysis', color = 'red')
        acceptance_plot.plot(BVPEA["acceptance"]['x'], BVPEA["acceptance"]["y"], color = 'tab:blue', label = "B. Beischer")
        acceptance_plot.set_ylim(bottom=0)
        acceptance_plot.set_xlim(*(0.05, 1000))
        acceptance_plot.legend()
        set_energy_ticks(acceptance_plot)
        acceptance_plot.yaxis.set_major_locator(mticker.MaxNLocator(steps=[1,2,5,10]))
        if args.no_title:
            save_figure(acceptance_fig, args.plotdir, f"{args.outputprefix}_Effective_Acceptance_NoTitle", transparent=args.transparent)
        acceptance_plot.set_title("Effective Acceptance")
        save_figure(acceptance_fig, args.plotdir, f"{args.outputprefix}_Effective_Acceptance", transparent=args.transparent)


        area_hist_1d_1GeV, energy_min, energy_max = EA_2d_hist_rb.project_by_value(1, axis = 0, return_bin_edges=True)
        figure_1d = plt.figure(figsize=(8, 4.2))
        plot_1d = figure_1d.subplots(1, 1)
        plot_histogram_1d(plot_1d, area_hist_1d_1GeV, show_overflow=False, label_y="Effective Area / $cm^2$", label = 'This Analysis', color = 'red')
        plot_1d.plot(BVPEA["1GeV"]["x"],BVPEA["1GeV"]["y"], color = 'tab:blue', label = 'B. Beischer')
        plot_1d.set_ylim(bottom=0)
        plot_1d.set_xlim(BVPEA["1GeV"]["xlim"][0], BVPEA["1GeV"]["xlim"][1])
        plot_1d.legend()
        plot_1d.yaxis.set_major_locator(mticker.MaxNLocator(steps=[1,2,5,10]))
        if args.no_title:
            save_figure(figure_1d, args.plotdir, f"{args.outputprefix}_Effective_Area_at_1GeV_NoTitle", transparent=args.transparent)
        plot_1d.set_title(f"Effective Area, ${energy_min:.2f}<= E/GeV<{energy_max:.2f}$")
        save_figure(figure_1d, args.plotdir, f"{args.outputprefix}_Effective_Area_at_1GeV", transparent=args.transparent)

        area_hist_1d_1GeV, energy_min, energy_max = EA_2d_hist_rb.project_by_value(1, axis = 0, return_bin_edges=True)
        figure_1d = plt.figure(figsize=(8, 4.2))
        plot_1d = figure_1d.subplots(1, 1)
        plot_histogram_1d(plot_1d, area_hist_1d_1GeV, style='curve', marker='', show_overflow=False, label_y="Effective Area / $cm^2$", label = 'This Analysis', color = 'red')
        plot_1d.plot(BVPEA["1GeV"]["x"],BVPEA["1GeV"]["y"], color = 'tab:blue', label = 'B. Beischer')
        plot_1d.set_ylim(bottom=0)
        plot_1d.set_xlim(BVPEA["1GeV"]["xlim"][0], BVPEA["1GeV"]["xlim"][1])
        plot_1d.set_ylabel("Effective Area / $cm^2$")
        plot_1d.yaxis.set_major_locator(mticker.MaxNLocator(steps=[1,2,5,10]))
        plot_1d.legend()
        if args.no_title:
            save_figure(figure_1d, args.plotdir, f"{args.outputprefix}_Effective_Area_at_1GeV_curve_NoTitle", transparent=args.transparent)
        plot_1d.set_title(f"Effective Area, ${energy_min:.2f}<= E/GeV<{energy_max:.2f}$")
        save_figure(figure_1d, args.plotdir, f"{args.outputprefix}_Effective_Area_at_1GeV_curve", transparent=args.transparent)

        polar_2d_hist = EA_3d_hist.project_by_value(2, axis = 0)
        figure_2d_polar = plt.figure(figsize=(7, 6))
        plot_theta_phi = figure_2d_polar.subplots(1, 1, subplot_kw=dict(projection="polar"))
        values = polar_2d_hist.values[::-1,:].transpose()
        edges_x = polar_2d_hist.binnings[1].edges
        edges_y = 1 - polar_2d_hist.binnings[0].edges[::-1]
        cos_ticks = np.arange(5) * 10
        cos_theta_ticks = 1 - np.cos(cos_ticks / 180 * np.pi)
        cos_theta_tick_labels = [f"{tick:.0f}°" for tick in cos_ticks]
        plot_2d(plot_theta_phi, values[1:-1,1:-1], edges_x[1:-1], edges_y[1:-1], colorbar_label="Effective Area / $cm^2$", cmap = 'jet')
        plot_theta_phi.set_rticks(cos_theta_ticks, cos_theta_tick_labels)
        plot_theta_phi.set_rlim(0, 1 - np.cos(40 / 180 * np.pi))
        #plot_histogram_2d(plot_theta_phi, area_hist_2d, transpose=True, show_overflow=False, label="Effective Area / $cm^2$", cmap = "jet")
        figure_2d_polar.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.85)
        if args.no_title:
            save_figure(figure_2d_polar, args.plotdir, f"{args.outputprefix}_EffectiveArea_per_cos_theta_phi_polar_at_2GeV_NoTitle", transparent=args.transparent)   
        plot_theta_phi.set_title(f"Effective Area, ${energy_min:.2f}<= E/GeV<{energy_max:.2f}$")
        save_figure(figure_2d_polar, args.plotdir, f"{args.outputprefix}_EffectiveArea_per_cos_theta_phi_polar_at_2GeV", transparent=args.transparent) 

        figure_2d_polar = plt.figure(figsize=(7, 6))
        plot_theta_phi = figure_2d_polar.subplots(1, 1, subplot_kw=dict(projection="polar"))
        values = polar_2d_hist.values[::-1,:].transpose()
        edges_x = polar_2d_hist.binnings[1].edges
        edges_y = 1 - polar_2d_hist.binnings[0].edges[::-1]
        cos_ticks = np.arange(5) * 10
        cos_theta_ticks = 1 - np.cos(cos_ticks / 180 * np.pi)
        cos_theta_tick_labels = [f"{tick:.0f}°" for tick in cos_ticks]
        plot_2d(plot_theta_phi, values[1:-1,1:-1], edges_x[1:-1], edges_y[1:-1], colorbar_label="Effective Area / $cm^2$", cmap = 'jet', min_value = BVPEA["polar"]["colorbar"][0], max_value = BVPEA["polar"]["colorbar"][1])
        plot_theta_phi.set_rticks(cos_theta_ticks, cos_theta_tick_labels)
        plot_theta_phi.set_rlim(0, 1 - np.cos(40 / 180 * np.pi))
        #plot_histogram_2d(plot_theta_phi, area_hist_2d, transpose=True, show_overflow=False, label="Effective Area / $cm^2$", cmap = "jet")
        figure_2d_polar.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.85)
        if args.no_title:
            save_figure(figure_2d_polar, args.plotdir, f"{args.outputprefix}_EffectiveArea_per_cos_theta_phi_polar_at_2GeV_BastianZ_NoTitle", transparent=args.transparent)
        plot_theta_phi.set_title(f"Effective Area, ${energy_min:.2f}<= E/GeV<{energy_max:.2f}$")
        save_figure(figure_2d_polar, args.plotdir, f"{args.outputprefix}_EffectiveArea_per_cos_theta_phi_polar_at_2GeV_BastianZ", transparent=args.transparent)



        area_hist_1d_perpendicular, cos_min, cos_max = EA_2d_hist_rb.project_by_value(0.99999, axis = 1, return_bin_edges=True)
        figure_1d = plt.figure(figsize=(8, 4.2))
        plot_1d = figure_1d.subplots(1, 1)
        plot_histogram_1d(plot_1d, area_hist_1d_perpendicular, show_overflow=False, label_y="Effective Area / $cm^2$", label = 'This Analysis', color = 'red')
        plot_1d.plot(BVPEA["perpendicular"]["x"], BVPEA["perpendicular"]["y"], color = 'tab:blue', label = 'B. Beischer')
        plot_1d.set_ylim(bottom=0)
        plot_1d.set_xlim(BVPEA["perpendicular"]["xlim"][0], BVPEA["perpendicular"]["xlim"][1])
        plot_1d.legend()
        set_energy_ticks(plot_1d)
        plot_1d.yaxis.set_major_locator(mticker.MaxNLocator(steps=[1,2,5,10]))
        if args.no_title:
            save_figure(figure_1d, args.plotdir, f"{args.outputprefix}_Effective_Area_perpendicular_NoTitle", transparent=args.transparent)
        plot_1d.set_title(f"Effective Area, ${cos_min:.2f}<= cos(\Theta)<{cos_max:.2f}$")
        save_figure(figure_1d, args.plotdir, f"{args.outputprefix}_Effective_Area_perpendicular", transparent=args.transparent)


    if ER:

        ER_migration_hist = ER_histograms['migration_hist']
        ER_relative_difference_hist = ER_histograms['relative_difference_hist']
        energy_binning = make_energy_binning(0.05, 1000, 0.0430103)

        migration_figure = plt.figure(figsize=(5.5, 4.2))
        migration_plot = migration_figure.subplots(1, 1)
        plot_histogram_2d(migration_plot, ER_migration_hist, scale=1 / ER_migration_hist.values.sum(axis=1), log=True, show_overflow=False, cmap = "jet", min_value = BVPER["migration"]["colorbar"][0], max_value = BVPER["migration"]["colorbar"][1])
        migration_plot.plot(energy_binning.edges[1:-1], energy_binning.edges[1:-1], "-", color="gray", alpha=0.5, linewidth=1)
        migration_plot.set_xlim(energy_binning.edges[1], energy_binning.edges[-2])
        migration_plot.set_ylim(energy_binning.edges[1], energy_binning.edges[-2]) 
        set_energy_ticks(migration_plot)
        set_energy_ticks(migration_plot, axis="y")
        migration_plot.set_box_aspect(1)
        migration_figure.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
        if args.no_title:
            save_figure(migration_figure, args.plotdir, f"{args.outputprefix}_EnergyResolution_migration_NoTitle", transparent=args.transparent)
        migration_plot.set_title("Energy Migration")
        save_figure(migration_figure, args.plotdir, f"{args.outputprefix}_EnergyResolution_migration", transparent=args.transparent)

        with np.load(Energyresolutionfile) as f:
            reldiff_mean = f["relative_bias"]
            reldiff_std = f["relative_resolution"]

        relative_difference_figure = plt.figure(figsize=(5.5, 4.2))
        relative_difference_plot = relative_difference_figure.subplots(1, 1)
        plot_histogram_2d(relative_difference_plot, ER_relative_difference_hist, scale=1 / ER_relative_difference_hist.values.sum(axis=1), log=True, show_overflow=False, cmap = "jet", min_value = BVPER["resolution"]["colorbar"][0], max_value = BVPER["resolution"]["colorbar"][1])
        relative_difference_plot.plot(energy_binning.edges[1:-1], np.zeros_like(energy_binning.edges[1:-1]), "-", color="gray", alpha=0.5, linewidth=1)
        relative_difference_plot.plot(energy_binning.bin_centers[1:-1], reldiff_mean, ".", markersize=1, color="black")
        relative_difference_plot.plot(energy_binning.bin_centers[1:-1], reldiff_mean + reldiff_std, "^", markersize=1, color="black")
        relative_difference_plot.plot(energy_binning.bin_centers[1:-1], reldiff_mean - reldiff_std, "v", markersize=1, color="black")
        relative_difference_plot.set_xlim(energy_binning.edges[1], energy_binning.edges[-2])
        relative_difference_plot.set_ylim(-1,1)
        set_energy_ticks(relative_difference_plot)
        relative_difference_plot.set_box_aspect(1)
        relative_difference_figure.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
        relative_difference_plot.yaxis.set_major_locator(mticker.MaxNLocator(steps=[1,2,5,10]))
        if args.no_title:
            save_figure(relative_difference_figure, args.plotdir, f"{args.outputprefix}_EnergyResolution_relative_difference_NoTitle", transparent=args.transparent)
        relative_difference_plot.set_title("Relative Energy Difference")
        save_figure(relative_difference_figure, args.plotdir, f"{args.outputprefix}_EnergyResolution_relative_difference", transparent=args.transparent)


        ER_relative_difference_hist_2GeV, min_e, max_e = ER_relative_difference_hist.project_by_value(2,axis = 0, return_bin_edges=True)
        relative_difference_2GeV_figure = plt.figure(figsize=(8, 4.2))
        relative_difference_2GeV_plot = relative_difference_2GeV_figure.subplots(1, 1)
        plot_histogram_1d(relative_difference_2GeV_plot, ER_relative_difference_hist_2GeV, show_overflow=False, label_y = "Events", color = 'red')
        #relative_difference_plot.set_xlabel(r'$(E_{rec} - E_{MC})/E_{MC}$')
        relative_difference_2GeV_plot.set_ylim(bottom=0)
        relative_difference_2GeV_plot.set_xlim(-1,1)
        relative_difference_2GeV_plot.axvline(reldiff_mean[37], color = 'tab:blue')
        relative_difference_2GeV_plot.axvline(reldiff_mean[37] + reldiff_std[37], color = 'orange')
        relative_difference_2GeV_plot.axvline(reldiff_mean[37] - reldiff_std[37], color = 'orange')
        relative_difference_2GeV_plot.xaxis.set_major_locator(mticker.MaxNLocator(steps=[1,2,5,10]))
        if args.no_title:
            save_figure(relative_difference_2GeV_figure, args.plotdir, f"{args.outputprefix}_EnergyResolution_relative_difference_at_2GeV_NoTitle", transparent=args.transparent)
        relative_difference_2GeV_plot.set_title(f"Relative Energy Difference, ${min_e:.2f}<= E/GeV<{max_e:.2f}$")   
        save_figure(relative_difference_2GeV_figure, args.plotdir, f"{args.outputprefix}_EnergyResolution_relative_difference_at_2GeV", transparent=args.transparent)

        relative_bias_figure = plt.figure(figsize=(8, 4.2))
        relative_bias_plot = relative_bias_figure.subplots(1, 1)
        relative_bias_plot.plot(energy_binning.bin_centers[1:-1], reldiff_mean, "o", markersize=1, color="red", label = 'This Analysis')
        relative_bias_plot.plot(BVPER["mean"]["x"], BVPER["mean"]["y"], color = 'tab:blue', label = 'B. Beischer')
        relative_bias_plot.set_xlabel("$E_{MC} / GeV$")
        relative_bias_plot.set_ylabel("Relative Bias")
        relative_bias_plot.semilogx()
        relative_bias_plot.set_xlim(1.5*1e-1, energy_binning.edges[-2])
        relative_bias_plot.set_ylim(-0.5,0.5)
        set_energy_ticks(relative_bias_plot)
        relative_bias_plot.legend()
        if args.no_title:
            save_figure(relative_bias_figure, args.plotdir, f"{args.outputprefix}_EnergyResolution_relative_bias_NoTitle", transparent=args.transparent)
        relative_bias_plot.set_title("Relative Energy Bias")
        save_figure(relative_bias_figure, args.plotdir, f"{args.outputprefix}_EnergyResolution_relative_bias", transparent=args.transparent)


        relative_resolution_figure = plt.figure(figsize=(8, 4.2))
        relative_resolution_plot = relative_resolution_figure.subplots(1, 1)
        relative_resolution_plot.plot(energy_binning.bin_centers[1:-1], reldiff_std, "o", markersize=1, color="red", label = 'This Analysis')
        relative_resolution_plot.plot(BVPER["rms"]["x"], BVPER["rms"]["y"], color = 'tab:blue', label = 'B. Beischer')
        relative_resolution_plot.set_xlabel("$E_{MC} / GeV$")
        relative_resolution_plot.set_ylabel("Relative Resolution")
        relative_resolution_plot.set_xlim(1.5*1e-1, energy_binning.edges[-2])
        relative_resolution_plot.set_ylim(0,0.6)
        relative_resolution_plot.semilogx()
        set_energy_ticks(relative_resolution_plot)
        relative_resolution_plot.legend()
        if args.no_title:
            save_figure(relative_resolution_figure, args.plotdir, f"{args.outputprefix}_EnergyResolution_relative_resolution_NoTitle", transparent=args.transparent)
        relative_resolution_plot.set_title("Relative Energy Resolution")
        save_figure(relative_resolution_figure, args.plotdir, f"{args.outputprefix}_EnergyResolution_relative_resolution", transparent=args.transparent)


    if PSF:
        
        PSF_hist = PSF_histograms['histogram_3d']
        PSF_hist_2d_xy, energy_min, energy_max = PSF_hist.project_by_value(2, axis=0, return_bin_edges=True)
        figure_2d = plt.figure(figsize=(6, 4.2))
        plot_2d_PSF = figure_2d.subplots(1, 1)
        plot_histogram_2d(plot_2d_PSF, PSF_hist_2d_xy, show_overflow=False, cmap = "jet", min_value = BVPPSF["alphaX_alphaY"]["colorbar"][0], max_value = BVPPSF["alphaX_alphaY"]["colorbar"][1])
        
        plot_2d_PSF.set_box_aspect(1)
        plot_2d_PSF.set_xlim(-7.5,7.5)
        plot_2d_PSF.set_ylim(-7.5,7.5)
        figure_2d.subplots_adjust(left=0.125, right=0.85, bottom=0.125, top=0.9)
        plot_2d_PSF.yaxis.set_major_locator(mticker.MaxNLocator(steps=[1,2,5,10]))
        plot_2d_PSF.xaxis.set_major_locator(mticker.MaxNLocator(steps=[1,2,5,10]))
        plot_2d_PSF.set_box_aspect(1)
        if args.no_title:
            save_figure(figure_2d, args.plotdir, f"{args.outputprefix}_PointSpreadFunction_at_2GeV_BastianZ_NoTitle", transparent=args.transparent)
        plot_2d_PSF.set_title(f"Point Spread Function, ${energy_min:.2f}<=E/GeV<{energy_max:.2f}$")
        save_figure(figure_2d, args.plotdir, f"{args.outputprefix}_PointSpreadFunction_at_2GeV_BastianZ", transparent=args.transparent)

        figure_2d = plt.figure(figsize=(6, 4.2))
        plot_2d_PSF = figure_2d.subplots(1, 1)
        plot_histogram_2d(plot_2d_PSF, PSF_hist_2d_xy, show_overflow=False, cmap = "jet")
        plot_2d_PSF.set_xlim(-7.5,7.5)
        plot_2d_PSF.set_ylim(-7.5,7.5)
        plot_2d_PSF.yaxis.set_major_locator(mticker.MaxNLocator(steps=[1,2,5,10]))
        plot_2d_PSF.xaxis.set_major_locator(mticker.MaxNLocator(steps=[1,2,5,10]))
        figure_2d.subplots_adjust(left=0.125, right=0.85, bottom=0.125, top=0.9)
        plot_2d_PSF.set_box_aspect(1)
        if args.no_title:
            save_figure(figure_2d, args.plotdir, f"{args.outputprefix}_PointSpreadFunction_at_2GeV_NoTitle", transparent=args.transparent)
        plot_2d_PSF.set_title(f"Point Spread Function, ${energy_min:.2f}<=E/GeV<{energy_max:.2f}$")
        save_figure(figure_2d, args.plotdir, f"{args.outputprefix}_PointSpreadFunction_at_2GeV", transparent=args.transparent)

        figure_x = plt.figure(figsize=(5.5, 4.2))
        plot_x = figure_x.subplots(1, 1)
        hist_x = PSF_hist.project_axis(axis=2)
        alpha_x_resolution, alpha_x_resolution_uncertainty = calculate_resolution_from_percentile(hist_x, plotdir=args.plotdir, outputprefix=f"{args.outputprefix}_x")
        alpha_x_mask = np.isfinite(alpha_x_resolution)
        alpha_x_parameters, alpha_x_parameter_errors = fit_resolution_parametrization(energy_binning.bin_centers[1:-1][alpha_x_mask], alpha_x_resolution[alpha_x_mask], alpha_x_resolution_uncertainty[alpha_x_mask] if alpha_x_resolution_uncertainty is not None else None, min_energy=0.15, max_energy=1000)
        fit_label = f"$\\sigma={alpha_x_parameters['a']:.4f}^\\circ / (E/GeV + {alpha_x_parameters['b']:.4f}) + {alpha_x_parameters['c']:.4f}^\\circ$"
        plot_histogram_2d(plot_x, hist_x, scale=1 / hist_x.values.sum(axis=1), log=True, show_overflow=False, cmap = "jet", min_value = BVPPSF["E_alphaX"]["colorbar"][0], max_value = BVPPSF["E_alphaX"]["colorbar"][1])
        plot_x.plot(energy_binning.bin_centers[1:-1], +alpha_x_resolution, "^", markersize=1, color="black")
        plot_x.plot(energy_binning.bin_centers[1:-1], -alpha_x_resolution, "v", markersize=1, color="black")
        plot_x.plot(energy_binning.bin_centers[1:-1], resolution_parametrization(energy_binning.bin_centers[1:-1], **alpha_x_parameters), "-", color="red", linewidth=1, label=fit_label)
        fit_label_bastian = f'B. Beischer: $\\sigma={BVPPSF["E_alphaX"]["fitpara"][0]:.3f}^\\circ / (E/GeV + {BVPPSF["E_alphaX"]["fitpara"][1]:.4f}) + {BVPPSF["E_alphaX"]["fitpara"][2]:.4f}^\\circ$'
        plot_x.plot(energy_binning.bin_centers[1:-1], resolution_parametrization(energy_binning.bin_centers[1:-1], BVPPSF["E_alphaX"]["fitpara"][0], BVPPSF["E_alphaX"]["fitpara"][1], BVPPSF["E_alphaX"]["fitpara"][2]), "-", color="w", linewidth=1, alpha=0.9, label=fit_label_bastian)
        set_energy_ticks(plot_x)
        plot_x.set_xlim(1.5*1e-1, 1000)
        plot_x.set_ylim(-7.5, 7.5)
        plot_x.legend(fontsize=6)
        plot_x.set_box_aspect(1)
        figure_x.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
        if args.no_title:
            save_figure(figure_x, args.plotdir, f"{args.outputprefix}_PointSpreadFunction_alpha_x_NoTitle", transparent=args.transparent)
        plot_x.set_title("Point Spread Function, Alpha X")
        save_figure(figure_x, args.plotdir, f"{args.outputprefix}_PointSpreadFunction_alpha_x", transparent=args.transparent)

        fig_x_comp = plt.figure(figsize=(8, 4.2))
        plt_x_comp = fig_x_comp.subplots(1,1)
        plt_x_comp.plot(energy_binning.bin_centers[1:-1], +alpha_x_resolution, "^", markersize=1, color="black")
        fit_label_this = f"This Analysis: $\\sigma={alpha_x_parameters['a']:.4f}^\\circ / (E/GeV + {alpha_x_parameters['b']:.4f}) + {alpha_x_parameters['c']:.4f}^\\circ$"
        fit_label_bastian = f'B. Beischer: $\\sigma={BVPPSF["E_alphaX"]["fitpara"][0]:.3f}^\\circ / (E/GeV + {BVPPSF["E_alphaX"]["fitpara"][1]:.4f}) + {BVPPSF["E_alphaX"]["fitpara"][2]:.4f}^\\circ$'
        plt_x_comp.plot(energy_binning.bin_centers[1:-1], resolution_parametrization(energy_binning.bin_centers[1:-1], **alpha_x_parameters), "-", color="red", linewidth=1, label=fit_label_this)
        plt_x_comp.plot(energy_binning.bin_centers[1:-1], resolution_parametrization(energy_binning.bin_centers[1:-1], BVPPSF["E_alphaX"]["fitpara"][0], BVPPSF["E_alphaX"]["fitpara"][1], BVPPSF["E_alphaX"]["fitpara"][2]), "-", color="green", linewidth=1, label=fit_label_bastian)
        plt_x_comp.semilogx()
        plt_x_comp.legend()
        fig_x_comp.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
        if args.no_title:
            save_figure(fig_x_comp, args.plotdir, f"{args.outputprefix}_PSF_alpha_x_comparison_NoTitle", transparent=args.transparent)
        plt_x_comp.set_title("PDF Alpha X comparison")
        save_figure(fig_x_comp, args.plotdir, f"{args.outputprefix}_PSF_alpha_x_comparison", transparent=args.transparent)

        fig_x_comp = plt.figure(figsize=(8, 4.2))
        plt_x_comp = fig_x_comp.subplots(1,1)
        plt_x_comp.plot(energy_binning.bin_centers[1:-1], +alpha_x_resolution, "^", markersize=1, color="black")
        fit_label_this = f"This Analysis: $\\sigma={alpha_x_parameters['a']:.4f}^\\circ / (E/GeV + {alpha_x_parameters['b']:.4f}) + {alpha_x_parameters['c']:.4f}^\\circ$"
        fit_label_bastian = f'B. Beischer: $\\sigma={BVPPSF["E_alphaX"]["fitpara"][0]:.3f}^\\circ / (E/GeV + {BVPPSF["E_alphaX"]["fitpara"][1]:.4f}) + {BVPPSF["E_alphaX"]["fitpara"][2]:.4f}^\\circ$'
        plt_x_comp.plot(energy_binning.bin_centers[1:-1], resolution_parametrization(energy_binning.bin_centers[1:-1], **alpha_x_parameters), "-", color="red", linewidth=1, label=fit_label_this)
        plt_x_comp.plot(energy_binning.bin_centers[1:-1], resolution_parametrization(energy_binning.bin_centers[1:-1], BVPPSF["E_alphaX"]["fitpara"][0], BVPPSF["E_alphaX"]["fitpara"][1], BVPPSF["E_alphaX"]["fitpara"][2]), "-", color="green", linewidth=1, label=fit_label_bastian)
        plt_x_comp.semilogx()
        plt_x_comp.legend()
        fig_x_comp.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
        if args.no_title:
            save_figure(fig_x_comp, args.plotdir, f"{args.outputprefix}_alpha_x_comparison_NoTitle", transparent=args.transparent)
        plt_x_comp.set_title("PDF Alpha X comparison")
        save_figure(fig_x_comp, args.plotdir, f"{args.outputprefix}_alpha_x_comparison", transparent=args.transparent)

        figure_y = plt.figure(figsize=(5.5, 4.2))
        plot_y = figure_y.subplots(1, 1)
        hist_y = PSF_hist.project_axis(axis=1)
        alpha_y_resolution, alpha_y_resolution_uncertainty = calculate_resolution_from_percentile(hist_y, plotdir=args.plotdir, outputprefix=f"{args.outputprefix}_y")
        alpha_y_mask = np.isfinite(alpha_y_resolution)
        alpha_y_parameters, alpha_y_parameter_errors = fit_resolution_parametrization(energy_binning.bin_centers[1:-1][alpha_y_mask], alpha_y_resolution[alpha_y_mask], alpha_y_resolution_uncertainty[alpha_y_mask] if alpha_y_resolution_uncertainty is not None else None, min_energy=0.15, max_energy=1000)
        fit_label = f"$\\sigma={alpha_y_parameters['a']:.4f}^\\circ / (E/GeV + {alpha_y_parameters['b']:.4f}) + {alpha_y_parameters['c']:.4f}^\\circ$"
        plot_histogram_2d(plot_y, hist_y, scale=1 / hist_y.values.sum(axis=1), log=True, show_overflow=False, cmap = "jet", min_value = BVPPSF["E_alphaY"]["colorbar"][0], max_value = BVPPSF["E_alphaY"]["colorbar"][1])
        plot_y.plot(energy_binning.bin_centers[1:-1], +alpha_y_resolution, "^", markersize=1, color="black")
        plot_y.plot(energy_binning.bin_centers[1:-1], -alpha_y_resolution, "v", markersize=1, color="black")
        plot_y.plot(energy_binning.bin_centers[1:-1], resolution_parametrization(energy_binning.bin_centers[1:-1], **alpha_y_parameters), "-", color="red", linewidth=1, label=fit_label)
        fit_label_bastian = f'B. Beischer: $\\sigma={BVPPSF["E_alphaY"]["fitpara"][0]:.3f}^\\circ / (E/GeV + {BVPPSF["E_alphaY"]["fitpara"][1]:.4f}) + {BVPPSF["E_alphaY"]["fitpara"][2]:.4f}^\\circ$'
        plot_y.plot(energy_binning.bin_centers[1:-1], resolution_parametrization(energy_binning.bin_centers[1:-1], BVPPSF["E_alphaY"]["fitpara"][0], BVPPSF["E_alphaY"]["fitpara"][1], BVPPSF["E_alphaY"]["fitpara"][2]), "-", color="w", linewidth=1, alpha=0.9, label=fit_label_bastian)
        set_energy_ticks(plot_y)
        plot_y.set_xlim(1.5*1e-1, 1000)
        plot_y.set_ylim(-7.5, 7.5)
        plot_y.legend(fontsize=6)
        plot_y.set_box_aspect(1)
        figure_y.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
        if args.no_title:
            save_figure(figure_y, args.plotdir, f"{args.outputprefix}_PointSpreadFunction_alpha_y_NoTitle", transparent=args.transparent)
        plot_y.set_title("Point Spread Function, Alpha Y")
        save_figure(figure_y, args.plotdir, f"{args.outputprefix}_PointSpreadFunction_alpha_y", transparent=args.transparent)
        
        fig_y_comp = plt.figure(figsize=(8, 4.2))
        plt_y_comp = fig_y_comp.subplots(1,1)
        
        plt_y_comp.plot(energy_binning.bin_centers[1:-1], +alpha_y_resolution, "^", markersize=1, color="black")
        fit_label_this = f"This Analysis: $\\sigma={alpha_y_parameters['a']:.4f}^\\circ / (E/GeV + {alpha_y_parameters['b']:.4f}) + {alpha_y_parameters['c']:.4f}^\\circ$"
        fit_label_bastian = f'B. Beischer: $\\sigma={BVPPSF["E_alphaY"]["fitpara"][0]:.3f}^\\circ / (E/GeV + {BVPPSF["E_alphaY"]["fitpara"][1]:.4f}) + {BVPPSF["E_alphaY"]["fitpara"][2]:.4f}^\\circ$'
        plt_y_comp.plot(energy_binning.bin_centers[1:-1], resolution_parametrization(energy_binning.bin_centers[1:-1], **alpha_y_parameters), "-", color="red", linewidth=1, label=fit_label_this)
        plt_y_comp.plot(energy_binning.bin_centers[1:-1], resolution_parametrization(energy_binning.bin_centers[1:-1], BVPPSF["E_alphaY"]["fitpara"][0], BVPPSF["E_alphaY"]["fitpara"][1], BVPPSF["E_alphaY"]["fitpara"][2]), "-", color="green", linewidth=1, label=fit_label_bastian)
        plt_y_comp.semilogx()
        plt_y_comp.legend()
        fig_y_comp.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9)
        if args.no_title:
            save_figure(fig_y_comp, args.plotdir, f"{args.outputprefix}_PSF_alpha_y_comparison_NoTitle", transparent=args.transparent)
        plt_y_comp.set_title("PDF Alpha Y comparison")
        save_figure(fig_y_comp, args.plotdir, f"{args.outputprefix}_PSF_alpha_y_comparison", transparent=args.transparent)





if __name__ == "__main__":
    main()
