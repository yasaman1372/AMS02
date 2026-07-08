#!/usr/bin/env python3

import os

import numpy as np
import awkward as ak
import matplotlib as mpl
import matplotlib.pyplot as plt

from tools.config import get_config
from tools.sample import Sample
from tools.utilities import save_figure, plot_2d

BRANCHES = ["RunNumber", "EventNumber", "TrkClustersX", "TrkClustersY", "TrkClusterLayersX", "TrkClusterLayersY", "TrkClusterOffsetsX", "TrkClusterOffsetsY", "TrkClusterCoordinatesY", "TrkClusterLengthsX", "TrkClusterLengthsY", "TrkTrackFitCoordinates", "TrkNTracks", "TrkHitClusterIndicesX", "TrkHitClusterIndicesY", "TrkHitTrackIndices"]

HALF_SIZE = 4200
STRIP_PITCH_IN_CM = 110e-4

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True, help="Path to ROOT trees to read in.")
    parser.add_argument("--treename", default="PhotonTree", help="Name of the tree in the ROOT files.")
    parser.add_argument("--config", nargs=2, required=True, help="Path to config file and workdir.")
    parser.add_argument("--sample", required=True, help="Sample name to load.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--outputprefix", default="TrackerClusters", help="Prefix for plots and result files.")
    parser.add_argument("--n-images", type=int, default=10, help="Number of cluster images to create.")

    args = parser.parse_args()

    config_file, workdir = args.config
    config = get_config(config_file)

    sample = Sample.load(config, args.sample, workdir)

    os.makedirs(args.plotdir, exist_ok=True)
    image_index = 0

    amplitude_cmap = mpl.colormaps.get_cmap("plasma_r")
    length_cmap = mpl.colormaps.get_cmap("tab10")

    for events in sample.read_tree(args.tree, args.treename, BRANCHES, rank=0, nranks=1):
        for event in events.get_array():
            figure = plt.figure(figsize=(8, 4.2))
            plot = figure.subplots(1, 1)
            for layer in range(1, 10):
                plot.axhline(layer, color="tab:gray", alpha=0.8, linewidth=1)
            #image = np.zeros((2 * HALF_SIZE, 9))
            hit_indices = np.arange(len(event.TrkHitTrackIndices))
            associated_hit_indices = hit_indices[event.TrkHitTrackIndices >= 0]
            #associated_clusters_x = event.TrkHitClusterIndicesX[associated_hit_indices]
            associated_clusters_y = event.TrkHitClusterIndicesY[associated_hit_indices]
            #associated_clusters_x = associated_clusters_x[associated_clusters_x >= 0]
            associated_clusters_y = associated_clusters_y[associated_clusters_y >= 0]
            for index, (layer, coord, offset, length, amplitudes) in enumerate(zip(event.TrkClusterLayersY, event.TrkClusterCoordinatesY, event.TrkClusterOffsetsY, event.TrkClusterLengthsY, event.TrkClustersY)):
                #image[HALF_SIZE + offset:HALF_SIZE + offset + length,layer - 1] = amplitudes
                scale = 15
                #plot.bar(np.arange(length) * scale + offset - scale * length / 2, -np.log10(amplitudes) / 1.5, width=scale, bottom=layer, color=amplitude_cmap((np.log10(amplitudes) - 0.5)))
                strip_coords = coord + np.arange(length) * STRIP_PITCH_IN_CM * scale
                #color = amplitude_cmap((np.log10(amplitudes) - 0.5))
                color = length_cmap((length - 1) / 10)
                alpha = 0.9 if index in associated_clusters_y else 0.3
                plot.bar(strip_coords - scale * length * STRIP_PITCH_IN_CM / 2, -amplitudes / 40, width=scale * STRIP_PITCH_IN_CM, bottom=layer, color=color, alpha=alpha)
            #plot_2d(plot, image, np.arange(2 * HALF_SIZE + 1) - HALF_SIZE - 0.5, np.arange(10) + 0.5, log=True)
            for trackindex in range(event.TrkNTracks):
                track_y = event.TrkTrackFitCoordinates[27 * trackindex + 1:27 * (trackindex + 1) + 1:3]
                track_layers = np.arange(1, 10)
                plot.plot(track_y[1:8], track_layers[1:8], "-", alpha=0.5)

            plot.set_title(f"{event.RunNumber} {event.EventNumber}")
            plot.set_xlim(-50, 50)
            plot.set_ylim(0.5, 9.5)
            plot.yaxis.set_inverted(True)
            plot.set_ylabel("Layer")
            plot.set_xlabel("Y / cm")
            save_figure(figure, args.plotdir, f"{args.outputprefix}_{image_index}", dpi=600)
            image_index += 1




if __name__ == "__main__":
    main()
