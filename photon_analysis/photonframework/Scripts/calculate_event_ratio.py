#!/usr/bin/env python3

from glob import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from tools.binnings import Binning
from tools.histograms import WeightedHistogram, plot_histogram_1d, plot_histogram_2d
from tools.utilities import plot_steps, save_figure, set_energy_ticks


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", nargs=2, required=True, help="Path to config and workdir")
    parser.add_argument("--iss-files", nargs="+", required=True, help="Path to files containing ISS data event histograms.")
    parser.add_argument("--mc-species", nargs="+", dest="mc_species", action="append", help="Species and path to files containing its event histograms.")
    parser.add_argument("--energy-estimator", default="TrkRigidityAbsAll", help="Trk Rigidity variable name")
    parser.add_argument("--mc-energy-estimator", default="McAbsRigidity", help="Trk Rigidity variable name")
    parser.add_argument("--outputprefix", default="McEventRatioWeight", help="Prefix for result files and plots.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--resultdir", default="results", help="Directory to store result files in.")

    args = parser.parse_args()

    os.makedirs(args.plotdir, exist_ok=True)
    os.makedirs(args.resultdir, exist_ok=True)

    results = {}

    iss_rigidity_hist = None
    for pattern in args.iss_files:
        for filename in glob(pattern):
            with np.load(filename) as iss_file:
                hist = WeightedHistogram.from_file(iss_file, f"hist_{args.energy_estimator}")
                if iss_rigidity_hist is None:
                    iss_rigidity_hist = hist
                else:
                    iss_rigidity_hist += hist
    rig_binning = iss_rigidity_hist.binnings[0]
    results["rigidity_edges"] = list(rig_binning.edges)
    results["weight_type"] = "event_ratio"
    results["weights"] = {}

    for species, *patterns in args.mc_species:
        print(species)
        mc_rigidity_hist = None
        mc_true_rigidity_hist = None
        for pattern in patterns:
            for filename in glob(pattern):
                with np.load(filename) as mc_file:
                    rig_hist = WeightedHistogram.from_file(mc_file, f"hist_{args.energy_estimator}")
                    mc_rig_hist = WeightedHistogram.from_file(mc_file, f"hist_{args.mc_energy_estimator}")
                    if mc_rigidity_hist is None:
                        mc_rigidity_hist = rig_hist
                        mc_true_rigidity_hist = mc_rig_hist
                    else:
                        mc_rigidity_hist += rig_hist
                        mc_true_rigidity_hist += mc_rig_hist

        assert mc_rigidity_hist.binnings[0] == mc_true_rigidity_hist.binnings[0] and iss_rigidity_hist.binnings[0] == mc_rigidity_hist.binnings[0]

        mc_events_per_true_rig = mc_true_rigidity_hist.values
        mc_events_per_reconstructed_rig = mc_rigidity_hist.values
        iss_events_per_reconstructed_rig = iss_rigidity_hist.values
        
        log_bin_width = np.log(rig_binning.edges[1:]) - np.log(rig_binning.edges[:-1])
        mc_weights = iss_events_per_reconstructed_rig / mc_events_per_true_rig
        flat_mc_weights = log_bin_width / log_bin_width[1:-1].sum() * iss_events_per_reconstructed_rig[1:-1].sum() / mc_events_per_true_rig

        figure = plt.figure(figsize=(12, 6.15))
        event_plot, weight_plot = figure.subplots(2, 1, sharex=True, gridspec_kw=dict(hspace=0))
        plot_histogram_1d(event_plot, iss_rigidity_hist, style="iss", label="ISS")
        plot_histogram_1d(event_plot, mc_rigidity_hist, style="mc", label=f"MC {species}")
        plot_histogram_1d(event_plot, mc_true_rigidity_hist, style="mc", label=f"MC {species} (true)")
        event_plot.legend()
        plot_steps(weight_plot, rig_binning.edges, mc_weights, label="Weights")
        plot_steps(weight_plot, rig_binning.edges, flat_mc_weights, label="Flat weights")
        weight_plot.legend()
        weight_plot.set_xlabel("Rigidity / GV")
        weight_plot.set_ylabel(f"{species} MC weight")

        save_figure(figure, args.plotdir, f"{args.outputprefix}_{species}")

        results["weights"][species] = dict(weights=list(mc_weights), flat_weights=list(flat_mc_weights))

    with open(os.path.join(args.resultdir, f"weights.json"), "w") as result_file:
        json.dump(results, result_file)

if __name__ == "__main__":
    main()
