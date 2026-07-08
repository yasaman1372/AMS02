#!/usr/bin/env python3

import tracemalloc

import glob
import multiprocessing as mp
import os

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

from tools.binnings import Binning, make_lin_binning
from tools.config import get_config
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_2d
from tools.roottree import read_tree, clear_cache
from tools.sample import Sample
from tools.utilities import save_figure

from tools.constants import RICH_RADIATOR_Z, RICH_PMT_Z, ECAL_Z_UPPER, ECAL_Z_LOWER, TOF_Z, TRD_Z_UPPER, TRD_Z_LOWER, TRK_LAYER_POSITION_Z 


def handle_file(arg):
    filename, treename, chunk_size, rank, nranks, kwargs = arg

    tracemalloc.start()

    config = kwargs["config"]
    workdir = kwargs["workdir"]
    verbose = kwargs.get("verbose", True)
    sample_name = kwargs["sample"]
    outputprefix = kwargs.get("outputprefix", "Vertex")
    resultdir = kwargs.get("resultdir", "results")

    resolution = kwargs.get("resolution", 0.1)
    z_depth = kwargs.get("z_depth", 1)
    z_min = kwargs.get("z_min", -175)
    z_max = kwargs.get("z_max", 200)
    z_positions = kwargs.get("z", np.linspace(z_min, z_max, int((z_max - z_min) / z_depth + 1)))
    max_radius = kwargs.get("xy_max", 150)
    min_hits = kwargs.get("min_hits", 3)
    min_hits_total = kwargs.get("min_hits_total", 6)
    max_distance = kwargs.get("max_distance", 1)
    min_invariant_mass = kwargs.get("min_invariant_mass", 0.01)

    xy_binning = make_lin_binning(-max_radius, max_radius, int(2 * max_radius / resolution))

    hists = {
        z: Histogram(xy_binning, xy_binning, labels=("X / cm", "Y / cm"), dtype=np.float32)
        for z in z_positions
    }

    sample = Sample.load(config, sample_name, workdir, fill_selection_hists=False)

    coordinates_branch = "TrkTrackPairMinDistanceCoordinates"
    angle_branch = "TrkTrackPairMinDistanceAngles"
    distance_branch = "TrkTrackPairMinDistances"
    invariant_mass_branch = "TrkTrackPairInvariantMasses"
    nhits_first_branch = "TrkTrackPairFirstTrackHits"
    nhits_second_branch = "TrkTrackPairSecondTrackHits"

    branches = [coordinates_branch, angle_branch, distance_branch, invariant_mass_branch, nhits_first_branch, nhits_second_branch]

    for events in sample.read_tree(filename, treename, branches=branches, rank=rank, nranks=nranks, chunk_size=chunk_size, verbose=verbose, resultdir=resultdir, prefix=outputprefix):
        coord_x = ak.flatten(events[coordinates_branch][:,0::3])
        coord_y = ak.flatten(events[coordinates_branch][:,1::3])
        coord_z = ak.flatten(events[coordinates_branch][:,2::3])
        distance = ak.flatten(events[distance_branch])
        angle = ak.flatten(events[angle_branch])
        invariant_mass = ak.flatten(events[invariant_mass_branch])
        nhits_first = ak.flatten(events[nhits_first_branch])
        nhits_second = ak.flatten(events[nhits_second_branch])

        selection = (
            (nhits_first >= min_hits) &
            (nhits_second >= min_hits) &
            (nhits_first + nhits_second >= min_hits_total) &
            (distance <= max_distance) &
            (invariant_mass > min_invariant_mass)
        )

        for slice_z, hist in hists.items():
            slice_selection = selection & (coord_z >= slice_z - z_depth / 2) & (coord_z < slice_z + z_depth / 2)
            slice_x = coord_x[slice_selection]
            slice_y = coord_y[slice_selection]
            hist.fill(slice_x, slice_y)

    results = {"z": np.array(list(hists.keys()))}
    for z, hist in hists.items():
        hist.add_to_file(results, f"hist_{z:.2f}")

    np.savez_compressed(os.path.join(resultdir, f"{outputprefix}_{rank}.npz"), **results)
    tracemalloc.stop()
    return rank


def make_args(filename, treename, chunk_size, nranks, parallel, **kwargs):
    parallel_index, parallel_total = parallel
    assert parallel_index >= 0 and parallel_index < parallel_total
    for rank in range(parallel_index * nranks, (parallel_index + 1) * nranks):
        yield (filename, treename, chunk_size, rank, (nranks * parallel_total), kwargs)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", nargs="+", required=True, help="Path to tree file(s).")
    parser.add_argument("--treename", help="Tree name in file.")
    parser.add_argument("--config", required=True, nargs=2, help="Path to analysis config file and workdir.")
    parser.add_argument("--sample", required=True, help="Sample selection from config to apply.")
    parser.add_argument("--outputprefix", default="VertexMap", help="Prefix for plot and result files.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of parallel processes.")
    parser.add_argument("--parallel", type=int, nargs=2, default=(0, 1), help="Index of this job and total number of parallel jobs.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Number of events to read per chunk.")
    parser.add_argument("--quiet", "-q", dest="verbose", action="store_false", help="Do not output progress info.")
    parser.add_argument("--max-distance", type=float, default=0.5, help="Maximum vertex distance to be used.")
    parser.add_argument("--min-angle", type=float, default=0, help="Minimum vertex angle to be used.")
    parser.add_argument("--z-range", type=float, nargs=2, default=(-175, 200), help="Range of z coordinates to scan.")
    parser.add_argument("--slice-depth", type=float, default=1, help="Depth of each plane.")
    parser.add_argument("--xy-max", type=float, default=150, help="Maximum value of x and y coordinates.")
    parser.add_argument("--xy-resolution", type=float, default=0.2, help="X/Y resolution.")

    args = parser.parse_args()

    config_file, workdir = args.config
    config = get_config(config_file)

    treename = args.treename or config["analysis"].get("treename", "PhotonTree")

    parallel = args.parallel
    parallel_index, parallel_total = parallel
    prefix = f"{args.outputprefix}_{parallel_index}"

    os.makedirs(args.resultdir, exist_ok=True)
    os.makedirs(args.plotdir, exist_ok=True)

    z_min, z_max = args.z_range

    clear_cache(treename)

    result_hists = {}

    with mp.get_context("fork" if args.verbose else "spawn").Pool(args.nprocesses) as pool:
        pool_args = make_args(args.tree, treename, args.chunk_size, args.nprocesses, parallel, config=config, sample=args.sample, workdir=workdir, resultdir=args.resultdir, plotdir=args.plotdir, outputprefix=prefix, verbose=args.verbose, max_distance=args.max_distance, min_angle=args.min_angle, resolution=args.xy_resolution, z_depth=args.slice_depth, z_min=z_min, z_max=z_max, xy_max=args.xy_max)
        for rank in pool.imap_unordered(handle_file, pool_args):
            rank_filename = os.path.join(args.resultdir, f"{prefix}_{rank}.npz")
            with np.load(rank_filename) as np_file:
                for z in np_file["z"]:
                    key = f"hist_{z:.2f}"
                    hist = Histogram.from_file(np_file, key)
                    if key in result_hists:
                        result_hists[z].add(hist)
                    else:
                        result_hists[z] = hist
            os.remove(rank_filename)

    results = {"z": np.array(sorted(result_hists.keys()))}
    for z, hist in result_hists.items():
        key = f"hist_{z:.2f}"
        hist.add_to_file(results, key)
    np.savez_compressed(os.path.join(args.resultdir, f"{prefix}.npz"), **results)

    for z, hist in result_hists.items():
        figure = plt.figure(figsize=(8, 6))
        plot = figure.subplots(1, 1)
        figure.suptitle(f"Slice Z = {z:.2f} cm")
        plot_histogram_2d(plot, hist, show_overflow=False)
        save_figure(figure, args.plotdir, f"{prefix}_{z:.2f}")
 

if __name__ == "__main__":
    main()
