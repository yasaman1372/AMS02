#!/usr/bin/env python3

from collections import defaultdict
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from tools.config import get_config
from tools.histograms import WeightedHistogram, plot_histogram_1d, plot_histogram_2d
from tools.sample import Sample
from tools.statistics import calculate_efficiency_and_error_weighted
from tools.utilities import plot_steps, shaded_steps, set_energy_ticks, round_up, round_down, make_tab_palette, save_figure
from tools.selection import Cut, Selection
from tools.binnings import Binning, make_log_binning_with_known_edge
from create_tag_and_probe_histograms import ResultHists


search_binning = Binning([0.1,0.5,1,10,100,1000], log=True)

def extract_cut_values(cut_config, cut_binning, return_intervals=False):
    cut_values = []
    removed_intervals = []
    binning_edges = cut_binning.edges[1:-1]
    is_int_binning = np.all(np.abs(binning_edges[1:] - binning_edges[:-1] - 1) < 1e-7)
    if "min" in cut_config:
        cut_value = cut_config["min"]
        if is_int_binning and cut_value == int(cut_value):
            cut_values.append(cut_value - 0.5)
            removed_intervals.append((binning_edges[0], cut_value - 0.5))
        else:
            cut_values.append(cut_value)
            removed_intervals.append((binning_edges[0], cut_value))
    if "max" in cut_config:
        cut_value = cut_config["max"]
        if is_int_binning and cut_value == int(cut_value):
            cut_values.append(cut_value + 0.5)
            removed_intervals.append((cut_value + 0.5, binning_edges[-1]))
        else:
            cut_values.append(cut_value)
            removed_intervals.append((cut_value, binning_edges[-1]))
    if "bool" in cut_config:
        if cut_config["bool"]:
            cut_values.extend([0.5, 1.5])
            removed_intervals.append((-0.5, 0.5))
        else:
            cut_values.extend([-0.5, 0.5])
            removed_intervals.append((0.5, 1.5))
    if "mask" in cut_config:
        target = cut_config["value"]
        mask = cut_config["mask"]
        passes = [(value & mask) == target for value in cut_binning.bin_centers.astype(np.int32)]
        for passes_below, passes_above, edge in zip(passes[:-1], passes[1:], binning_edges):
            if passes_below != passes_above:
                cut_values.append(edge)
        for passes, value in zip(passes, cut_binning.bin_centers):
            if not passes:
                removed_intervals.append((value - 0.5, value + 0.5))
    if return_intervals:
        return cut_values, removed_intervals
    return cut_values


def plot_efficiency(config, workdir, json_file, sampel_name, iss_dataset, mc_datasets, plotdir, resultdir, prefix, title, energy_estimator="McMomentum", search_binning=search_binning ,mask_datasets=()):
    with open(json_file, 'r') as file:
        efficiency_data = json.load(file)
    
    # mc_datasets = [dataset for dataset in efficiency_data[energy_estimator]["mc_datasets"] if dataset not in mask_datasets]
    
    sample = Sample.load(config, sampel_name , workdir)
    binnings = sample.binnings
    rig_search_binning = search_binning
    rig_min = rig_search_binning.edges[1]
    rig_max = rig_search_binning.edges[-2]

    selections = {}
    for cut_key in efficiency_data[energy_estimator]["cuts"]:
        selection_name, cut_name = cut_key.split(",")
        if selection_name not in selections:
            selections[selection_name] = []
        selections[selection_name].append(cut_name)

    #with open(os.path.join(resultdir, f"selection_{prefix}.tex"), "w") as tex_file:
    #    for selection_name, cut_names in selections.items():
    #        tex_file.write(f"\\input{{selection/{efficiency_name}/{selection_name}}}\n")
    #        #for cut_name in cut_names:
    #        #    tex_file.write(f"\\input{{selection/{efficiency_name}/{selection_name}_{cut_name}}}\n")

    for selection_name, cut_names in selections.items():
        selection_config = config["selections"][selection_name]
        flipped_selection = selection_config.get("flipped_selection", None)
        for cut_name in cut_names:
            cut_data = efficiency_data[energy_estimator]["cuts"][f"{selection_name},{cut_name}"]
            rigidity_binning = Binning(np.array(cut_data["energy"]), log=True)
            rigidities = rigidity_binning.bin_centers
            efficiency_iss, efficiency_error_iss = map(np.array, cut_data["iss"])
            cut_config = config["selections"][selection_name]["cuts"][cut_name]
            cut = Cut.load(cut_config, cut_name, binnings[cut_name], rigidity_binning, energy_estimator, config, workdir, fill_hists=False)
            cut_label = cut.label

            if flipped_selection is not None:
                alt_cut_config = config["selections"][flipped_selection]["cuts"][cut_name]
                alt_cut = Cut.load(alt_cut_config, cut_name, binnings[cut_name], rigidity_binning, energy_estimator, config, workdir, cut_name, fill_hists=False)
                if alt_cut.label != cut_label:
                    cut_label = f"{cut_label} ({alt_cut.label})"

            if cut_name == "TrkDoubleLayerPatternY":
                cut_label = "Hit in every double layer"
            cut_values, cut_intervals = extract_cut_values(cut_config, binnings[cut_name], return_intervals=True)

            iss_passed_hist = iss_dataset.tagged_and_passed_hists_2d[energy_estimator][selection_name,cut_name]
            iss_failed_hist = iss_dataset.tagged_and_failed_hists_2d[energy_estimator][selection_name,cut_name]
            print(selection_name,cut_name)
            
            cut_figure_2d = plt.figure(figsize=(8,4.2))
            cut_plot_2d = cut_figure_2d.subplots(1, 1)

            cut_figure_1d = plt.figure(figsize=(8,4.2))
            cut_plot_1d = cut_figure_1d.subplots(1, 1)

            efficiency_figure_1d = plt.figure(figsize=(8,4.2))
            efficiency_plot_1d = efficiency_figure_1d.subplots(1, 1)

            cut_figure = plt.figure(figsize=(8, 6.2))
            cut_plot, efficiency_plot = cut_figure.subplots(2, 1, sharex=True, gridspec_kw=dict(hspace=0))
            combined_hist = iss_passed_hist + iss_failed_hist
            combined_hist_1d = combined_hist.project_by_value(rig_min, rig_max, axis=0)
            plot_histogram_2d(cut_plot, combined_hist, scale=1 / combined_hist.values.sum(axis=1), colorbar=False, show_overflow_y=False, log=True)#, cmap = "jet")
            plot_histogram_2d(cut_plot_2d, combined_hist, scale=1 / combined_hist.values.sum(axis=1), show_overflow_y=False, log=True, cmap = "jet")
            plot_histogram_1d(cut_plot_1d, combined_hist_1d, style="iss", label="ISS")
            for cut_value in cut_values:
                cut_plot.axhline(cut_value, color="red", linewidth=1)
                cut_plot_2d.axhline(cut_value, color="red", linewidth=1)
                cut_plot_1d.axvline(cut_value, color="red", linewidth=1)

            efficiency_plot.errorbar(rigidities, efficiency_iss * 100, efficiency_error_iss * 100, fmt=".", label="ISS")
            efficiency_plot_1d.errorbar(rigidities, efficiency_iss * 100, efficiency_error_iss * 100, fmt=".", label="ISS")
            mc_efficiencies = []

            for mc_dataset_name, (mc_dataset_label, mc_dataset) in mc_datasets.items():
                efficiency_mc, efficiency_error_mc, mc_cut_key_str = cut_data[mc_dataset_name]
                efficiency_mc, efficiency_error_mc = map(np.array, (efficiency_mc, efficiency_error_mc))
                efficiency_plot.errorbar(rigidities, efficiency_mc * 100, efficiency_error_mc * 100, fmt=".", label=mc_dataset_label)
                efficiency_plot_1d.errorbar(rigidities, efficiency_mc * 100, efficiency_error_mc * 100, fmt=".", label=mc_dataset_label)
                mc_efficiencies.append(efficiency_mc)

                mc_passed_hist = mc_dataset.tagged_and_passed_hists_2d[energy_estimator][selection_name, cut_name]
                mc_failed_hist = mc_dataset.tagged_and_failed_hists_2d[energy_estimator][selection_name, cut_name]
                
                mc_combined_hist = mc_passed_hist + mc_failed_hist
                mc_combined_hist_1d = mc_combined_hist.project_by_value(rig_min, rig_max, axis=0)
                plot_histogram_1d(cut_plot_1d, mc_combined_hist_1d, scale=combined_hist_1d.values.sum() / mc_combined_hist_1d.values.sum(), style="mc", label=mc_dataset_label)

            cut_plot_1d.set_ylim(bottom=0)

            for cut_interval_min, cut_interval_max in cut_intervals:
                rmin = combined_hist.binnings[0].edges[1]
                rmax = combined_hist.binnings[0].edges[-2]
                ymin, ymax = cut_plot_1d.get_ylim()
                cut_plot.fill_between([rmin, rmax], [cut_interval_min, cut_interval_min], [cut_interval_max, cut_interval_max], color="gray", alpha=0.25, zorder=-1, edgecolor="none")
                cut_plot_2d.fill_between([rmin, rmax], [cut_interval_min, cut_interval_min], [cut_interval_max, cut_interval_max], color="gray", alpha=0.25, zorder=-1, edgecolor="none")
                cut_plot_1d.fill_betweenx([ymin, ymax], [cut_interval_min, cut_interval_min], [cut_interval_max, cut_interval_max], color="gray", alpha=0.25, zorder=-1, edgecolor="none")

            efficiency_plot.set_xscale("log")
            efficiency_plot.legend()
            efficiency_plot.set_xlabel("Rigidity / GV")
            efficiency_plot.set_ylabel("Efficiency / %")
            efficiency_plot.set_ylim(60, 110)
            efficiency_plot.set_xlim(rig_min, rig_max)
            set_energy_ticks(efficiency_plot)
            cut_plot.get_xaxis().set_visible(False)
            save_figure(cut_figure, plotdir, f"{prefix}_cut_{selection_name}_{cut_name}", dpi=300, save_pickle=True)

            cut_plot_2d.set_xlabel("Rigidity / GV")
            cut_plot_2d.set_ylabel(cut_name)
            cut_plot.set_ylabel(cut_name)
            cut_plot_2d.set_xlim(rig_min, rig_max)
            set_energy_ticks(cut_plot_2d)
            save_figure(cut_figure_2d, plotdir, f"{prefix}_cut_{selection_name}_{cut_name}_2d", dpi=300, save_pickle=True)

            
            binning = binnings[cut_name]
            xmin = binning.edges[1]
            xmax = binning.edges[-2]
            cut_plot_1d.set_xlim(xmin, xmax)
            cut_plot_1d.legend()
            cut_plot_1d.set_xlabel(cut_name)
            cut_plot_1d.ticklabel_format(axis="y", scilimits=(-3, 3), style="scientific", useMathText=True)
            save_figure(cut_figure_1d, plotdir, f"{prefix}_cut_{selection_name}_{cut_name}_1d", dpi=300, save_pickle=True)

            efficiency_plot_1d.set_title(cut_label)
            efficiency_plot_1d.legend()
            efficiency_plot_1d.set_xscale("log")
            efficiency_plot_1d.set_xlabel("Rigidity / GV")
            efficiency_plot_1d.set_ylabel("Efficiency / %")
            efficiency_plot_1d.set_ylim(60, 110)
            efficiency_plot_1d.set_xlim(rig_min, rig_max)
            set_energy_ticks(efficiency_plot_1d)
            save_figure(efficiency_figure_1d, plotdir, f"{prefix}_cut_{selection_name}_{cut_name}_efficiency", dpi=300, save_pickle=True)

            rig_sel = (rigidities >= rig_min) & (rigidities <= rig_max)
            print(prefix, selection_name, cut_name, f"{np.mean(efficiency_iss[rig_sel]):.2f}", *[f"{np.mean(efficiency[rig_sel]):.2f}" for efficiency in mc_efficiencies])

   

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=2, nargs=2, help="Path to analysis config file and workdir.")
    parser.add_argument("--sample", required=True, help="Sample to study selection efficiency for.")
    parser.add_argument("--iss-dataset", required=True, help="ISS dataset used for comparison.")
    parser.add_argument("--mc-dataset", dest="mc_datasets", action="append", required=True, help="MC dataset used for comparison.")
    parser.add_argument("--energy-estimators", nargs="+", help="Energy estimators to calculate efficiency as a function of.")
    parser.add_argument("--inputdir", required=True, help="Directory to load efficiency histograms from.")
    parser.add_argument("--energy-range", type=float, nargs=2, default=(1, 1000), help="Minimum and maximum energy.")
    parser.add_argument("--outputprefix", default="Efficiency", help="Prefix for the output file.")
    parser.add_argument("--resultdir", default="results", help="Directory to store result files in.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")

    args = parser.parse_args()

    config_file, workdir = args.config
    config = get_config(config_file)
    sample = Sample.load(config, args.sample, workdir)

    energy_estimators = args.energy_estimators
    if energy_estimators is None:
        energy_estimators = (sample.estimators["Energy"],)

    energy_min, energy_max = args.energy_range

    os.makedirs(args.resultdir, exist_ok=True)
    os.makedirs(args.plotdir, exist_ok=True)

    iss_dataset = ResultHists.load(args.inputdir, f"{args.iss_dataset}")
    mc_datasets = {mc_dataset: (config["datasets"][mc_dataset]["label"], ResultHists.load(args.inputdir, f"{mc_dataset}")) for mc_dataset in args.mc_datasets}

    cuts = {(selection_name, cut_name): cut for selection_name, selection in sample.selections.items() for cut_name, cut in selection.cuts.items()}

    default_tagger = "default"

    palette = make_tab_palette()
    iss_color = palette.get_color()
    mc_colors = {mc_dataset_name: palette.get_color() for mc_dataset_name in args.mc_datasets}
    true_color = palette.get_color()

    result_data = {energy_estimator: {"mc_datasets": args.mc_datasets, "cuts": {}, "taggers": {}} for energy_estimator in energy_estimators}

    for energy_estimator in energy_estimators:
        print(energy_estimator)
        efficiency_ratios_per_tagger = defaultdict(lambda: [])
        for cut_key, cut in cuts.items():
            key_str = "_".join(cut_key)
            tagger = cut.tagger or default_tagger

            iss_tagged_and_passed = iss_dataset.tagged_and_passed_hists[energy_estimator][cut_key]
            iss_tagged_and_failed = iss_dataset.tagged_and_failed_hists[energy_estimator][cut_key]

            iss_tag_and_probe_efficiency, iss_tag_and_probe_efficiency_error = calculate_efficiency_and_error_weighted(iss_tagged_and_passed.values, iss_tagged_and_failed.values, iss_tagged_and_passed.squared_values, iss_tagged_and_failed.squared_values)
            iss_tag_and_probe_mask = np.invert(np.isfinite(iss_tag_and_probe_efficiency))
            iss_tag_and_probe_efficiency = np.ma.masked_where(iss_tag_and_probe_mask, iss_tag_and_probe_efficiency)
            iss_tag_and_probe_efficiency_error = np.ma.masked_where(iss_tag_and_probe_mask, iss_tag_and_probe_efficiency_error)

            energy_binning = iss_tagged_and_passed.binnings[0]
            energy_edges = energy_binning.edges
            energy_points = energy_binning.bin_centers

            energy_bin_min, energy_bin_max = energy_binning.get_indices(np.array([energy_min, energy_max]))

            mc_only = np.all(iss_tag_and_probe_mask[energy_bin_min:energy_bin_max+1])
            min_efficiency = np.min((iss_tag_and_probe_efficiency - iss_tag_and_probe_efficiency_error)[energy_bin_min:energy_bin_max+1])
            max_ratio = 1.01
            min_ratio = 0.99

            cut_result = {
                "energy": list(energy_edges),
                "tagger": tagger,
                "iss": (list(iss_tag_and_probe_efficiency.data), list(iss_tag_and_probe_efficiency_error.data)),
            }

            figure = plt.figure(figsize=(12, 6.15))
            figure.suptitle(cut.label)
            gs = GridSpec(8, 1, hspace=0)
            plot = figure.add_subplot(gs[:8 if mc_only else 6,:])
            if not mc_only:
                ratio_plot = figure.add_subplot(gs[6:,:], sharex=plot)
                plot.errorbar(energy_points, iss_tag_and_probe_efficiency, iss_tag_and_probe_efficiency_error, fmt=".", label="ISS", color=iss_color)

            for mc_dataset_name, (mc_dataset_label, mc_dataset) in mc_datasets.items():
                mc_cut_key = cut_key
                if config["datasets"][mc_dataset_name]["charge"] < 0:
                    selection_name, cut_name = cut_key
                    if "flipped_selection" in config["selections"][selection_name]:
                        mc_cut_key = (config["selections"][selection_name]["flipped_selection"], cut_name)
                mc_triggered_and_passed = mc_dataset.triggered_and_passed_hists[energy_estimator][mc_cut_key]
                mc_triggered_and_failed = mc_dataset.triggered_and_failed_hists[energy_estimator][mc_cut_key]
                mc_tagged_and_passed = mc_dataset.tagged_and_passed_hists[energy_estimator][mc_cut_key]
                mc_tagged_and_failed = mc_dataset.tagged_and_failed_hists[energy_estimator][mc_cut_key]

                mc_efficiency, mc_efficiency_error = calculate_efficiency_and_error_weighted(mc_triggered_and_passed.values, mc_triggered_and_failed.values, mc_triggered_and_passed.squared_values, mc_triggered_and_failed.squared_values)
                mc_tag_and_probe_efficiency, mc_tag_and_probe_efficiency_error = calculate_efficiency_and_error_weighted(mc_tagged_and_passed.values, mc_tagged_and_failed.values, mc_tagged_and_passed.squared_values, mc_tagged_and_failed.squared_values)
                mc_mask = np.invert(np.isfinite(mc_efficiency))
                mc_efficiency = np.ma.masked_where(mc_mask, mc_efficiency)
                mc_efficiency_error = np.ma.masked_where(mc_mask, mc_efficiency_error)
                mc_tag_and_probe_mask = np.invert(np.isfinite(mc_tag_and_probe_efficiency))
                mc_tag_and_probe_efficiency = np.ma.masked_where(mc_tag_and_probe_mask, mc_tag_and_probe_efficiency)
                mc_tag_and_probe_efficiency_error = np.ma.masked_where(mc_tag_and_probe_mask, mc_tag_and_probe_efficiency_error)

                cut_result[mc_dataset_name] = (list(mc_tag_and_probe_efficiency.data), list(mc_tag_and_probe_efficiency_error.data), ":".join(mc_cut_key))

                if np.all(mc_tag_and_probe_efficiency == 1) and np.all(iss_tag_and_probe_efficiency == 1):
                    print(f"Warning: Efficiency of {cut.label!r} is exactly one, redundant cut or bad tag selection.")

                if mc_only:
                    plot_steps(plot, energy_edges, mc_efficiency, label=f"{mc_dataset_label}", color=true_color)
                plot_steps(plot, energy_edges, mc_tag_and_probe_efficiency, label=f"{mc_dataset_label}", color=mc_colors[mc_dataset_name])
                shaded_steps(plot, energy_edges, mc_tag_and_probe_efficiency - mc_tag_and_probe_efficiency_error, mc_tag_and_probe_efficiency + mc_tag_and_probe_efficiency_error, color=mc_colors[mc_dataset_name])


                if not mc_only:
                    iss_mc_efficiency_ratio = iss_tag_and_probe_efficiency / mc_tag_and_probe_efficiency 
                    iss_mc_efficiency_ratio_error = np.sqrt((iss_tag_and_probe_efficiency_error / iss_tag_and_probe_efficiency)**2 + (mc_tag_and_probe_efficiency_error / mc_tag_and_probe_efficiency)**2) * iss_mc_efficiency_ratio
                    ratio_plot.errorbar(energy_points, iss_mc_efficiency_ratio, iss_mc_efficiency_ratio_error, fmt=".", color=mc_colors[mc_dataset_name])

                    efficiency_ratios_per_tagger[tagger].append((cut.label, iss_mc_efficiency_ratio, iss_mc_efficiency_ratio_error))
                    max_ratio = max(max_ratio, round_up(np.max(np.abs(iss_mc_efficiency_ratio) + iss_mc_efficiency_ratio_error) * 1.05, digits=2))
                    min_ratio = min(min_ratio, round_down(np.min(np.abs(iss_mc_efficiency_ratio) - iss_mc_efficiency_ratio_error) / 1.05, digits=2))

                min_efficiency = min(min_efficiency, np.min((mc_tag_and_probe_efficiency - mc_tag_and_probe_efficiency_error)[energy_bin_min:energy_bin_max+1]))

            if not mc_only:
                ratio_plot.axhline(0, alpha=0.8, color="darkgray")

                plot.get_xaxis().set_visible(False)
                ratio_plot.set_xlabel(f"{energy_estimator} / GeV")
                ratio_plot.set_ylabel("$\\epsilon_{MC} / \\epsilon_{ISS}$")
                ratio_plot.set_ylim(min_ratio, max_ratio)
                ratio_plot.yaxis.tick_right()
                ratio_plot.yaxis.set_label_position("right")
            else:
                plot.set_xlabel(f"{energy_estimator} / GeV")

            plot.set_xscale("log")
            plot.set_xlim(energy_min, energy_max)
            set_energy_ticks(plot)
            plot.set_ylabel("Efficiency")
            if min_efficiency > 0.98:
                plot.set_ylim(0.975, 1.005)
            elif min_efficiency > 0.955:
                plot.set_ylim(0.95, 1.01)
            elif min_efficiency > 0.925:
                plot.set_ylim(0.9, 1.02)
            elif min_efficiency > 0.8:
                plot.set_ylim(0.75, 1.05)
            elif min_efficiency > 0.55:
                plot.set_ylim(0.5, 1.05)
            else:
                plot.set_ylim(0, 1.1)
            plot.legend()

            figure.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
            save_figure(figure, args.plotdir, f"{args.outputprefix}_{energy_estimator}_cut_{key_str}")

            result_data[energy_estimator]["cuts"][",".join(cut_key)] = cut_result

        for tagger, efficiency_ratios in efficiency_ratios_per_tagger.items():
            efficiency_ratio_labels = [r[0] for r in efficiency_ratios]
            efficiency_ratio_values = np.array([r[1] for r in efficiency_ratios])
            efficiency_ratio_errors = np.array([r[2] for r in efficiency_ratios])
            efficiency_ratio_relative_errors = efficiency_ratio_errors / efficiency_ratio_values

            figure = plt.figure(figsize=(12, 6.15))
            figure.suptitle(tagger)
            gs = GridSpec(8, 1, hspace=0)
            plot = figure.add_subplot(gs[:6,:])
            ratio_plot = figure.add_subplot(gs[6:,:], sharex=plot)

            tagger_iss_tagged_and_passed = iss_dataset.tagged_and_passed_hists[energy_estimator][tagger]
            tagger_iss_tagged_and_failed = iss_dataset.tagged_and_failed_hists[energy_estimator][tagger]
            tagger_iss_efficiency, tagger_iss_efficiency_error = calculate_efficiency_and_error_weighted(tagger_iss_tagged_and_passed.values, tagger_iss_tagged_and_failed.values, tagger_iss_tagged_and_passed.squared_values, tagger_iss_tagged_and_failed.squared_values)
            iss_mask = np.invert(np.isfinite(tagger_iss_efficiency))
            tagger_iss_efficiency = np.ma.masked_where(iss_mask, tagger_iss_efficiency)
            tagger_iss_efficiency_error = np.ma.masked_where(iss_mask, tagger_iss_efficiency_error)

            energy_binning = iss_dataset.tagged_and_passed_hists[energy_estimator][tagger].binnings[0]
            energy_edges = energy_binning.edges
            energy_points = energy_binning.bin_centers

            plot.errorbar(energy_points, tagger_iss_efficiency, tagger_iss_efficiency_error, fmt=".", label="ISS tag&probe", color=iss_color)

            efficiency_ratio_data = {}
            mc_efficiency_data = {}

            for mc_dataset_name, (_, mc_dataset) in mc_datasets.items():
                tagger_mc_tagged_and_passed = mc_dataset.tagged_and_passed_hists[energy_estimator][tagger]
                tagger_mc_tagged_and_failed = mc_dataset.tagged_and_failed_hists[energy_estimator][tagger]

                tagger_mc_efficiency, tagger_mc_efficiency_error = calculate_efficiency_and_error_weighted(tagger_mc_tagged_and_passed.values, tagger_mc_tagged_and_failed.values, tagger_mc_tagged_and_passed.squared_values, tagger_mc_tagged_and_failed.squared_values)
                mc_mask = np.invert(np.isfinite(tagger_mc_efficiency))
                tagger_mc_efficiency = np.ma.masked_where(mc_mask, tagger_mc_efficiency)
                tagger_mc_efficiency_error = np.ma.masked_where(mc_mask, tagger_mc_efficiency_error)

                mc_efficiency_data[mc_dataset_name] = (list(energy_points[~mc_mask]), list(tagger_mc_efficiency.compressed()), list(tagger_mc_efficiency_error.compressed()))

                iss_mc_efficiency_ratio = tagger_iss_efficiency / tagger_mc_efficiency
                iss_mc_efficiency_ratio_error = iss_mc_efficiency_ratio * np.sqrt((tagger_iss_efficiency_error / tagger_iss_efficiency)**2 + (tagger_mc_efficiency_error / tagger_mc_efficiency)**2)
                efficiency_ratio_data[mc_dataset_name] = (list(energy_points[~(iss_mask | mc_mask)]), list(iss_mc_efficiency_ratio.compressed()), list(iss_mc_efficiency_ratio_error.compressed()))

                max_ratio = round_up(np.max(np.abs(1 - iss_mc_efficiency_ratio) + iss_mc_efficiency_ratio_error) * 1.05)
                if max_ratio == 0:
                    max_ratio = 1e-3

                shaded_steps(plot, energy_edges, tagger_mc_efficiency - tagger_mc_efficiency_error, tagger_mc_efficiency + tagger_mc_efficiency_error, label=f"MC tag&probe {mc_dataset_name}", color=mc_colors[mc_dataset_name], alpha=0.5)

                ratio_plot.errorbar(energy_points, (iss_mc_efficiency_ratio - 1) * 100, iss_mc_efficiency_ratio_error * 100, fmt=".", color=mc_colors[mc_dataset_name])
                ratio_plot.axhline(0, alpha=0.8, color="darkgray")

                ratio_plot.set_xlabel(f"{energy_estimator} / GeV")
                plot.set_xscale("log")
                set_energy_ticks(plot)
                plot.get_xaxis().set_visible(False)
                plot.set_ylabel("Efficiency")
                plot.set_ylim(0, 1.1)
                ratio_plot.set_ylabel("$\\epsilon_{ISS}/\\epsilon_{MC}-1 / \\%$")
                #ratio_plot.set_ylim(-max_ratio * 100, max_ratio * 100)
                ratio_plot.set_ylim(-25, 25)
                ratio_plot.yaxis.tick_right()
                ratio_plot.yaxis.set_label_position("right")
            plot.legend()

            save_figure(figure, args.plotdir, f"{args.outputprefix}_{energy_estimator}_tagger_{tagger}")

            result_data[energy_estimator]["taggers"][tagger] = {"iss": (list(energy_points[~iss_mask]), list(tagger_iss_efficiency.compressed()), list(tagger_iss_efficiency_error.compressed())), "mc": mc_efficiency_data, "ratio": efficiency_ratio_data}

            ratio_figure = plt.figure(figsize=(12, 6.15))
            ratio_figure.suptitle(f"Efficiency ratios {tagger}")
            ratio_plot = ratio_figure.subplots(1, 1)

            palette = make_tab_palette()
            total_color = palette.get_color()

            if len(efficiency_ratio_labels) > 1:
                plot_steps(ratio_plot, energy_edges, iss_mc_efficiency_ratio, color=total_color, label="Total", zorder=4)
                shaded_steps(ratio_plot, energy_edges, values=iss_mc_efficiency_ratio, errors=iss_mc_efficiency_ratio_error, color=total_color, alpha=0.5)
            for label, values, errors in zip(efficiency_ratio_labels, efficiency_ratio_values, efficiency_ratio_errors):
                plot_steps(ratio_plot, energy_edges, values, label=label, color=palette.get_color())

            ratio_plot.plot(energy_edges, np.ones_like(energy_edges), color="gray", alpha=0.5, linewidth=1)
            ratio_plot.set_ylabel("ISS / MC efficiency ratio")
            ratio_plot.set_xlabel(f"{energy_estimator} / GeV")
            ratio_plot.set_xscale("log")
            set_energy_ticks(ratio_plot)
            ratio_plot.yaxis.tick_right()
            ratio_plot.yaxis.set_label_position("right")
            ratio_plot.legend()

            save_figure(ratio_figure, args.plotdir, f"{args.outputprefix}_{energy_estimator}_group_{tagger}")

    with open(os.path.join(args.resultdir, f"{args.outputprefix}.json"), "w") as result_file:
        json.dump(result_data, result_file)

    for energy_estimator in energy_estimators:
        plot_efficiency(config, workdir, os.path.join(args.resultdir, f"{args.outputprefix}.json"), args.sample, iss_dataset, mc_datasets, args.plotdir, args.resultdir, args.outputprefix, 'title', energy_estimator=energy_estimator, search_binning=search_binning ,mask_datasets=())


if __name__ == "__main__":
    main()
