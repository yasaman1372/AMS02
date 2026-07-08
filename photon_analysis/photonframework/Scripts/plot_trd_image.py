#!/usr/bin/env python3

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from tools.constants import MC_PARTICLE_NAMES, TRD_MAX_X_UPPER, TRD_MAX_X_LOWER, TRD_MAX_Y_UPPER, TRD_MAX_Y_LOWER
from tools.utilities import save_figure, plot_2d


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfile")
    parser.add_argument("--plotdir", default="plots")
    parser.add_argument("--outputprefix", default="TrdImage")
    parser.add_argument("--max-images", type=int, default=100)

    args = parser.parse_args()

    os.makedirs(args.plotdir, exist_ok=True)

    trd_width_lower = max(TRD_MAX_X_LOWER, TRD_MAX_Y_LOWER)
    trd_width_upper = max(TRD_MAX_X_UPPER, TRD_MAX_Y_UPPER)

    trd_outline = np.array([
        [-trd_width_lower, -0.5], 
        [-trd_width_upper, 19.5],
        [trd_width_upper, 19.5],
        [trd_width_lower, -0.5], 
    ])

    with np.load(args.inputfile) as np_file:
        batch = np_file["batches"]
        energies = np_file["energies"]
        mc_momenta = np_file["mc_momenta"]
        particle_ids = np_file["mc_particle_ids"]
        xy_steps = np_file["xy"]
        z_steps = np_file["z"]
        print(batch.shape)

        for index, (image, energy, mc_momentum, particle_id) in enumerate(zip(batch, energies, mc_momenta, particle_ids)):
            if index >= args.max_images:
                break
            print(index, len(batch), end="\r")
            if particle_id in MC_PARTICLE_NAMES:
                title = f"{mc_momentum:.2f} GeV MC {MC_PARTICLE_NAMES[particle_id]}"
            else:
                title = f"{energy:.2f} GeV event"
            figure = plt.figure(figsize=(8, 4.2))
            plot = figure.subplots(1, 1)
            plot.pcolormesh(xy_steps, z_steps, image.transpose(), cmap="hot_r")
            for layer in (4, 8, 12, 16):
                width = trd_width_lower + (trd_width_upper - trd_width_lower) / 20 * layer
                plot.plot([-width, width], [layer - 0.5, layer - 0.5], "-", color="tab:gray", alpha=0.5, linewidth=1)
            plot.add_patch(Polygon(trd_outline, closed=True, fill=False, edgecolor="black", linewidth=1))
            plot.set_title(title)
            plot.set_xlabel("X/Y / cm")
            plot.set_ylabel("TRD Layer")
            plot.spines["top"].set_visible(False)
            plot.spines["bottom"].set_visible(False)
            plot.spines["left"].set_visible(False)
            plot.spines["right"].set_visible(False)
            plot.tick_params(axis="y", left=False)
            plot.set_xticks([-75, -50, -25, 0, 25, 50, 75])
            plot.yaxis.set_visible(False)
            figure.subplots_adjust(left=0.0125, right=0.9825, bottom=0.125, top=0.925)
            save_figure(figure, args.plotdir, f"{args.outputprefix}_{index}")



if __name__ == "__main__":
    main()
