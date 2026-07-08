#!/usr/bin/env python3

import contextlib
import os
import multiprocessing as mp
import pickle

import numpy as np
import awkward as ak
import healpy as hp
import optuna
import uproot
from optuna.samplers import TPESampler
from astropy import units
from astropy.coordinates import SkyCoord

from tools.binnings import make_healpy_binning
from tools.config import get_config
from tools.histograms import Histogram
from tools.sample import Sample
from tools.roottree import read_tree
from tools.selection import Cut, Selection
from tools.healpy_tools import get_regions, count_pixels_in_region
from variables.trackpair import make_trk_track_best_pair_min_distance_points_to_region_from_file as make_points_to_region
from tools.coordinates import convert_galactic_to_equatorial_coordinates

from plot_skymap import plot_skymap


def calculate_angular_distance(latitude, longitude, ref_latitude, ref_longitude):
    direction = hp.ang2vec(longitude, latitude, lonlat=True)
    ref_direction = hp.ang2vec(ref_longitude, ref_latitude, lonlat=True)[None,:]
    return np.degrees(np.arccos(np.sum(direction * ref_direction, axis=1)))


def calculate_excess_and_significance(signal_events, background_events, signal_weight, background_weight):
    signal_uncertainty = np.sqrt(signal_events)
    background_events_in_signal_region = background_events * (signal_weight / background_weight)
    background_uncertainty = np.sqrt(background_events_in_signal_region)
    excess = signal_events - background_events_in_signal_region
    uncertainty = np.sqrt(signal_uncertainty**2 + background_uncertainty**2)
    signal_to_background_ratio = (signal_events - background_events_in_signal_region) / background_events_in_signal_region
    return excess, excess / uncertainty, signal_to_background_ratio


def load_events(filenames, selection_branches, signal_position, signal_window_size, background_region, background_window_size, nside, treename="GammaTree"):
    branches = selection_branches + ["GalacticLatitude", "GalacticLongitude", "Time", "Energy"]
    signal_latitude, signal_longitude = signal_position
    signal_ra, signal_dec = convert_galactic_to_equatorial_coordinates(signal_latitude, signal_longitude)
    is_in_background_region = make_points_to_region(None, background_region, False, "GalacticLatitude", "GalacticLongitude", nside=nside)
    signal_arrays = []
    background_arrays = []
    for filename in filenames:
        with uproot.open(filename) as root_file:
            tree = root_file[treename]
            events = tree.arrays(expressions=branches)
            events = events[np.isfinite(events.GalacticLatitude) & np.isfinite(events.GalacticLongitude)]
            events_ra, events_dec = convert_galactic_to_equatorial_coordinates(events.GalacticLatitude, events.GalacticLongitude)
            signal_arrays.append(ak.to_packed(events[calculate_angular_distance(events.GalacticLatitude, events.GalacticLongitude, signal_latitude, signal_longitude) <= signal_window_size]))
            background_arrays.append(ak.to_packed(events[is_in_background_region(events) & (np.abs(events_dec - signal_dec) <= background_window_size)]))
    signal_events = ak.concatenate(signal_arrays)
    background_events = ak.concatenate(background_arrays)
    return signal_events, background_events


def initialize_process(input_queue, output_queue, kwargs):
    global signal_events
    global background_events
    global check_value
    filenames, check_value = input_queue.get()

    signal_events, background_events = load_events(filenames, selection_branches=kwargs["selection_branches"], signal_position=kwargs["signal_position"], signal_window_size=kwargs["signal_window_size"], background_region=kwargs["background_region"], background_window_size=kwargs["background_window_size"], nside=kwargs["nside"], treename=kwargs["treename"])
    #output_queue.put((len(signal_events), len(background_events)))
    output_queue.put((len(signal_events), len(background_events), (np.array(signal_events.GalacticLatitude), np.array(signal_events.GalacticLongitude), np.array(background_events.GalacticLatitude), np.array(background_events.GalacticLongitude))))


def process_files(arg):
    global signal_events
    global background_events
    global check_value
    kwargs = arg


    config = kwargs["config"]
    workdir = kwargs["workdir"]
    sample_name = kwargs["sample"]
    modified_selection_configs = kwargs["selections"]

    for selection_name, new_cuts in modified_selection_configs.items():
        config["selections"][selection_name]["cuts"].update(new_cuts)

    sample = Sample.load(config, sample_name, workdir, fill_selection_hists=False)

    passed_signal_events = sample.apply(signal_events, derived_branches=[], is_mc=False)
    passed_background_events = sample.apply(background_events, derived_branches=[], is_mc=False)

    return len(passed_signal_events), len(passed_background_events), check_value



def modify_selection_config(trial, selection_name, selection_config, variables_to_modify):
    new_cuts = {}
    for variable, cut_config in selection_config["cuts"].items():
        if variable not in variables_to_modify:
            new_cuts[variable] = cut_config
            continue

        properties_to_modify = variables_to_modify[variable]["properties"]
        variable_type = variables_to_modify[variable]["type"]
        if variable_type == "int":
            suggest = lambda prop, args: trial.suggest_int(f"{selection_name}_{variable}_{prop}", args[0], args[1])
        elif variable_type == "float":
            suggest = lambda prop, args: trial.suggest_float(f"{selection_name}_{variable}_{prop}", args[0], args[1])
        elif variable_type == "bool":
            suggest = lambda prop, args: bool(trial.suggest_int(f"{selection_name}_{variable}_{prop}", 0, 1))
        if "use" in properties_to_modify:
            use = bool(trial.suggest_int(f"{selection_name}_use_{variable}", 0, 1))
            if not use:
                continue
        new_cuts[variable] = {}
        for property in cut_config:
            if property in properties_to_modify:
                new_cuts[variable][property] = suggest(property, properties_to_modify[property])
            else:
                new_cuts[variable][property] = cut_config[property]

    return new_cuts


def get_default_params(selection_name, selection_config, variables_to_modify):
    params = {}
    for variable, cut_config in selection_config["cuts"].items():
        if variable not in variables_to_modify:
            continue

        properties_to_modify = variables_to_modify[variable]["properties"]
        variable_type = variables_to_modify[variable]["type"]
        if "use" in properties_to_modify:
            params[f"{selection_name}_use_{variable}"] = True
        for property in cut_config:
            if property in properties_to_modify:
                params[f"{selection_name}_{variable}_{property}"] = cut_config[property]

    return params


def create_skymap(filenames, treename, nranks, sample_name, config, workdir, selection_configs, nside):
    def _make_args():
        for rank in range(nranks):
            yield (filenames, treename, rank, nranks, config, sample_name, workdir, selection_configs, nside)

    skymap = None
    with mp.Pool(nranks) as pool:
        for part in pool.imap_unordered(create_skymap_task, _make_args()):
            if skymap is None:
                skymap = part
            else:
                skymap = skymap + part
    return skymap

def create_skymap_task(arg):
    filenames, treename, rank, nranks, config, sample_name, workdir, selection_configs, nside = arg
    for selection_name, new_cuts in selection_configs.items():
        config["selections"][selection_name]["cuts"].update(new_cuts)
    sample = Sample.load(config, sample_name, workdir, fill_selection_hists=False)

    selection_branches = [variable for selection in sample.selections.values() for variable in selection.cuts]
    estimator_branches = list(sample.estimators)
    branches = sorted(set(selection_branches + estimator_branches))

    skymap = Histogram(make_healpy_binning(nside), labels=("Healpy Index",))

    for events in read_tree(filenames, treename, branches=branches, rank=rank, nranks=nranks, verbose=False):
        events = sample.apply(events, derived_branches=[], is_mc=False)
        skymap.fill(hp.ang2pix(nside, theta=events.GalacticLongitude, phi=events.GalacticLatitude, lonlat=True))

    return skymap


class SelectionTrial:
    def __init__(self, filenames, treename, config, workdir, sample_name, variables_to_modify, signal_position, signal_window_size, background_region, background_window_size, signal_pixels, background_pixels, nside, nprocesses, resultdir, plotdir, outputprefix):
        self.filenames = filenames
        self.treename = treename
        self.config = config
        self.workdir = workdir
        self.sample_name = sample_name
        self.variables_to_modify = variables_to_modify
        self.signal_position = signal_position
        self.signal_window_size = signal_window_size
        self.background_region = background_region
        self.background_window_size = background_window_size
        self.signal_pixels = signal_pixels
        self.background_pixels = background_pixels
        self.nside = nside
        self.nprocesses = nprocesses
        self.resultdir = resultdir
        self.plotdir = plotdir
        self.outputprefix = outputprefix

        self.pool = None

    @contextlib.contextmanager
    def setup(self):
        input_queue = mp.Queue()
        feedback_queue = mp.Queue()
        sample = Sample.load(self.config, self.sample_name, self.workdir, fill_selection_hists=False)
        selection_branches = [variable for selection in sample.selections.values() for variable in selection.cuts]
        estimator_branches = list(sample.estimators)
        branches_to_load = sorted(set(selection_branches + estimator_branches + list(self.variables_to_modify)))
        kwargs = dict(selection_branches=branches_to_load, signal_position=self.signal_position, signal_window_size=self.signal_window_size, background_region=self.background_region, background_window_size=self.background_window_size, nside=self.nside, treename=self.treename)
        for rank in range(self.nprocesses):
            input_queue.put((self.filenames[rank::self.nprocesses], rank + 1))
        total_signal_region_events = 0
        total_background_region_events = 0

        healpy_binning = make_healpy_binning(self.nside)
        signal_histogram = Histogram(healpy_binning, labels=("Healpy Index",))
        background_histogram = Histogram(healpy_binning, labels=("Healpy Index",))

        with mp.Pool(self.nprocesses, initializer=initialize_process, initargs=(input_queue, feedback_queue, kwargs)) as pool:
            for _ in range(self.nprocesses):
                #sig, bkg = feedback_queue.get()
                sig, bkg, (signal_latitude, signal_longitude, background_latitude, background_longitude) = feedback_queue.get()
                total_signal_region_events += sig
                total_background_region_events += bkg
                signal_histogram.fill(hp.ang2pix(self.nside, theta=signal_longitude, phi=signal_latitude, lonlat=True))
                background_histogram.fill(hp.ang2pix(self.nside, theta=background_longitude, phi=background_latitude, lonlat=True))

            plot_skymap(signal_histogram, self.resultdir, self.plotdir, f"{self.outputprefix}_signal", "Signal", normalization="events", save_fits=False, rotate_center=self.signal_position[::-1])
            plot_skymap(background_histogram, self.resultdir, self.plotdir, f"{self.outputprefix}_background", "Background", normalization="events", save_fits=False, rotate_center=self.signal_position[::-1])

            print(f"Loaded {total_signal_region_events} in signal and {total_background_region_events} events in background region", flush=True)
            self.pool = pool
            yield

    def get_default_params(self):
        params = {}
        for selection_name in self.config["samples"][self.sample_name]["selections"]:
            params.update(get_default_params(selection_name, self.config["selections"][selection_name], self.variables_to_modify))
        print("default", params, flush=True)
        return params

    def perform_trial(self, trial):
        new_selection_configs = {}
        for selection_name in self.config["samples"][self.sample_name]["selections"]:
            new_selection_configs[selection_name] = modify_selection_config(trial, selection_name, self.config["selections"][selection_name], self.variables_to_modify)

        kwargs = dict(config=self.config, workdir=self.workdir, sample=self.sample_name, selections=new_selection_configs)

        total_signal = 0
        total_background = 0
        checksum = 0
        for signal_events, background_events, check_value in self.pool.imap_unordered(process_files, (kwargs for _ in range(self.nprocesses))):
            total_signal += signal_events
            total_background += background_events
            checksum += check_value
        assert checksum == sum(range(self.nprocesses + 1))
        excess, significance, signal_to_background_ratio = calculate_excess_and_significance(total_signal, total_background, self.signal_pixels, self.background_pixels)
        bg_prediction = total_signal - excess
        trial.set_user_attr("excess", excess)
        trial.set_user_attr("bg_prediction", bg_prediction)
        trial.set_user_attr("sb_ratio", signal_to_background_ratio)
        trial.set_user_attr("signal", total_signal)
        trial.set_user_attr("background", total_background)
        return significance

    def plot_trial_skymap(self, trial, resultdir, plotdir, outputprefix, title):
        new_selection_configs = {}
        for selection_name in self.config["samples"][self.sample_name]["selections"]:
            new_selection_configs[selection_name] = modify_selection_config(trial, selection_name, self.config["selections"][selection_name], self.variables_to_modify)
        skymap = create_skymap(self.filenames, self.treename, self.nprocesses, self.sample_name, self.config, self.workdir, new_selection_configs, self.nside)

        bg_per_pixel = trial.user_attrs["bg_prediction"] / self.signal_pixels
        excess_per_pixel = trial.user_attrs["excess"] * 2 / self.signal_pixels + bg_per_pixel
        signal_latitude, signal_longitude = self.signal_position
        excess_label = f"{trial.user_attrs['excess']:.0f} signal + {trial.user_attrs['bg_prediction']:.0f} bg = {trial.value:.1f}σ"
        title = f"{title}, {excess_label}"

        plot_skymap(skymap, resultdir, plotdir, f"{outputprefix}_all", title, normalization="events", save_fits=True, vmin=bg_per_pixel, vmax=excess_per_pixel, rotate_center=self.signal_position[::-1])
        plot_skymap(skymap, resultdir, plotdir, f"{outputprefix}_zoom", title, normalization="events", save_fits=True, vmin=bg_per_pixel, vmax=excess_per_pixel, latitude_range=(-5, +5), longitude_range=(-5, +5), rotate_center=self.signal_position[::-1])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("trees", nargs="+", help="Path to ROOT files to load.")
    parser.add_argument("--treename", default="GammaTree", help="Name of the tree in the ROOT files.")
    parser.add_argument("--config", nargs=2, required=True, help="Path to config file and workdir.")
    parser.add_argument("--sample", required=True, help="Name of the sample to optimize")
    parser.add_argument("--variable", dest="variables", action="append", nargs="+", required=True, help="Cut variable to vary, its type (bool, int, float), then properties (use, min, max, …) and their limits (min_value, max_value, only for non-bool properties).")
    parser.add_argument("--signal-position", type=float, metavar=("latitude", "longitude"), nargs=2, required=True, help="Source position in galactic coordinates in degrees")
    parser.add_argument("--window-size", type=float, nargs=2, default=(0.5, 5), metavar=("signal", "background"), help="Window sizes in degrees.")
    parser.add_argument("--background-region", required=True, help="Path to the background region file (for counting the pixels in the region)")
    parser.add_argument("--nside", type=int, default=256, help="nside for counting the signal and background region pixels.")
    parser.add_argument("--ntrials", type=int, default=1000, help="Number of trials to run.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--resultdir", default="results", help="Directory to store result files in.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plot files in.")
    parser.add_argument("--outputprefix", default="CutOptimization", help="Prefix for result files.")
    parser.add_argument("--title", default="Optimization", help="Title for plots.")
    parser.add_argument("--reload", action="store_true", help="Load previously run study.")

    args = parser.parse_args()

    config_filename, workdir = args.config
    config = get_config(config_filename)

    os.makedirs(args.plotdir, exist_ok=True)
    os.makedirs(args.resultdir, exist_ok=True)

    latitude, longitude = args.signal_position
    signal_ra, signal_dec = convert_galactic_to_equatorial_coordinates(latitude, longitude)
    if longitude < 180:
        longitude = -longitude
    elif longitude >= 180:
        longitude = 360 - longitude
    signal_window_size, background_window_size = args.window_size
    signal_region = [("Disc", (longitude, latitude, signal_window_size))]
    background_region = get_regions(args.background_region)
    pixels_in_signal_region = count_pixels_in_region(args.nside, signal_region, False, False)
    pixels_in_background_region = count_pixels_in_region(args.nside, background_region, True, True, dec_range=(signal_dec - background_window_size, signal_dec + background_window_size))
    print("Weights:", pixels_in_signal_region, pixels_in_background_region)

    config["samples"][args.sample]["estimators"]["Energy"] = "Energy"

    variables_to_modify = {}
    for variable_name, variable_type, *property_args in args.variables:
        variables_to_modify[variable_name] = {"type": variable_type, "properties": {}}
        while property_args:
            property = property_args[0]
            if property == "use" or variable_type == "bool":
                variables_to_modify[variable_name]["properties"][property] = ()
                property_args = property_args[1:]
            elif property in ("min", "max"):
                min_value, max_value = map(float, property_args[1:3])
                variables_to_modify[variable_name]["properties"][property] = (min_value, max_value)
                property_args = property_args[3:]
            else:
                raise ValueError(f"Unknown property args {property_args}")

    print(variables_to_modify)

    trial = SelectionTrial(args.trees, args.treename, config, workdir, args.sample, variables_to_modify, args.signal_position, signal_window_size, args.background_region, background_window_size, pixels_in_signal_region, pixels_in_background_region, args.nside, args.nprocesses, args.resultdir, args.plotdir, args.outputprefix)

    if not args.reload:
        with trial.setup():
            study = optuna.create_study(sampler=TPESampler(), direction="maximize", study_name=args.outputprefix)
            study.set_metric_names(["Significance"])
            study.enqueue_trial(trial.get_default_params())
            study.optimize(trial.perform_trial, n_trials=args.ntrials)
    else:
        with open(os.path.join(args.resultdir, f"{args.outputprefix}_study.pck"), "rb") as pickle_file:
            study = pickle.load(pickle_file)

    best_trial = study.best_trial

    print(study.trials[0])
    print(best_trial)
    trial.plot_trial_skymap(study.trials[0], args.resultdir, args.plotdir, f"{args.outputprefix}_default", f"{args.title} (Default)")
    trial.plot_trial_skymap(best_trial, args.resultdir, args.plotdir, f"{args.outputprefix}_best", f"{args.title} (Best)")

    if not args.reload:
        with open(os.path.join(args.resultdir, f"{args.outputprefix}_study.pck"), "wb") as pickle_file:
            pickle.dump(study, pickle_file)



if __name__ == "__main__":
    main()
