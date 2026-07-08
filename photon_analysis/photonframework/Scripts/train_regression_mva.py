#!/usr/bin/env python3

from fnmatch import filter as fnmatch
from glob import glob
import json
import multiprocessing as mp
import multiprocessing.pool as mp_pool
import os

import numpy as np
from numpy.lib import recfunctions
import matplotlib.pyplot as plt
from matplotlib.contour import ContourSet
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogLocator, LogFormatter
import xgboost as xgb
from iminuit import Minuit
from iminuit.cost import LeastSquares, BinnedNLL, ExtendedBinnedNLL, ExtendedUnbinnedNLL
import uncertainties
from scipy.stats import gaussian_kde, norm, chi2

from tools.binnings import Binning, Binnings, make_lin_binning, reduce_bins
from tools.confidence import calculate_contour, calculate_confidence_interval
from tools.config import get_config
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d
from tools.statistics import calculate_correlation, calculate_likelihood, calculate_efficiency_and_rejection, calculate_signal_and_background_efficiency, calculate_cut_value_for_efficiency, smooth_additive
from tools.utilities import plot_steps, shaded_steps, rec_to_float, filter_branches, float_or_int, plot_feature_importance, save_figure
from train_mva import calculate_rejection, Dataset, predict_bdt, prediction_hist


def train_estimator(variables, train_dataset, predict_dataset, target_variable, prefix, title, label, resultdir, plotdir, rig_title="", max_depth=4, ntrees=20, eta=1.0, binnings=None, nprocesses=os.cpu_count(), target_is_bool=False):

    os.makedirs(plotdir, exist_ok=True)

    train_events = filter_branches(train_dataset.events, variables)
    predict_events = filter_branches(predict_dataset.events, variables)

    train_labels = train_dataset.events[target_variable]
    predict_labels = predict_dataset.events[target_variable]

    print(train_events)
    for var in variables:
        print(var, np.any(np.isnan(train_events[var])), np.all(np.isfinite(train_events[var])))

    dtrain = xgb.DMatrix(rec_to_float(train_events), label=train_labels, feature_names=variables)
    dpredict = xgb.DMatrix(rec_to_float(predict_events), label=predict_labels, feature_names=variables)

    evallist = [(dpredict, "eval"), (dtrain, "train")]

    loss = "reg:squarederror"
    if target_is_bool:
        loss = "binary:logistic"
    train_params = {"max_depth": max_depth, "eta": eta, "objective": loss, "nthread": 4, "eval_metric": "rmse"}

    print("Starting training", flush=True)

    bdt = xgb.train(train_params, dtrain, num_boost_round=ntrees, evals=evallist, early_stopping_rounds=3)

    def _predict(dataset):
        return predict_bdt(bdt, dataset, variables)

    train_predictions = _predict(train_dataset)
    predict_predictions = _predict(predict_dataset)

    min_score = min(np.min(train_predictions), np.min(predict_predictions))
    max_score = max(np.max(train_predictions), np.max(predict_predictions))
    score_range = max_score - min_score
    min_score -= score_range / 8
    max_score += score_range / 8

    target_binning =  binnings.variable_binnings[target_variable]
    if target_is_bool:
        mva_nbins = 1000
        mva_binning = make_lin_binning(min_score, max_score, mva_nbins)
    else:
        mva_binning = target_binning

    bdt_model_path = os.path.abspath(os.path.join(resultdir, f"{prefix}_model.json"))
    result_data = {"path": bdt_model_path, "variables": variables, "title": title, "rig_title": rig_title, "label": label, "min_score": min_score, "max_score": max_score}

    plot_feature_importance(bdt, variables, f"{title} Ranking{rig_title}", plotdir, prefix)


    mva_target_figure = plt.figure(figsize=(12, 6.15))
    mva_target_figure.suptitle(f"{title}{rig_title}")
    mva_target_plot = mva_target_figure.subplots(1, 1)
    mva_target_hist_predict = WeightedHistogram.fill_direct((target_binning, mva_binning), predict_labels, predict_predictions, weights=predict_dataset.events.TotalWeight, labels=(f"{target_variable}", f"MVA {label}"))
    plot_histogram_2d(mva_target_plot, mva_target_hist_predict, log=True)
    save_figure(mva_target_figure, plotdir, f"{prefix}_target_vs_mva")

    for variable in variables:
        target_variable_figure = plt.figure(figsize=(12, 6.15))
        target_variable_figure.suptitle(f"{title}{rig_title}")
        target_variable_plot = target_variable_figure.subplots(1, 1)
        target_variable_hist_predict = WeightedHistogram.fill_direct((target_binning, binnings.variable_binnings[variable]), predict_labels, predict_dataset.events[variable], weights=predict_dataset.events.TotalWeight, labels=(f"{target_variable}", f"{variable}"))
        plot_histogram_2d(target_variable_plot, target_variable_hist_predict, log=True)
        save_figure(target_variable_figure, plotdir, f"{prefix}_target_vs_{variable}")

        mva_variable_figure = plt.figure(figsize=(12, 6.15))
        mva_variable_figure.suptitle(f"{title}{rig_title}")
        mva_variable_plot = mva_variable_figure.subplots(1, 1)
        mva_variable_hist_predict = WeightedHistogram.fill_direct((mva_binning, binnings.variable_binnings[variable]), predict_predictions, predict_dataset.events[variable], weights=predict_dataset.events.TotalWeight, labels=(f"MVA {label}", f"{variable}"))
        plot_histogram_2d(mva_variable_plot, mva_variable_hist_predict, log=True)
        save_figure(mva_variable_figure, plotdir, f"{prefix}_mva_vs_{variable}")

    if target_is_bool:
        true_selection = predict_labels == 1
        true_events = predict_dataset.events[true_selection]
        true_predictions = predict_predictions[true_selection]
        false_selection = ~true_selection
        false_events = predict_dataset.events[false_selection]
        false_predictions = predict_predictions[false_selection]

        split_figure = plt.figure(figsize=(12, 6.15))
        split_plot = split_figure.subplots(1, 1)
        true_hist = WeightedHistogram.fill_direct((mva_binning,), true_predictions, weights=true_events.TotalWeight, labels=(f"MVA {label}",))
        false_hist = WeightedHistogram.fill_direct((mva_binning,), false_predictions, weights=false_events.TotalWeight, labels=(f"MVA {label}",))
        plot_histogram_1d(split_plot, true_hist, log=True, label=f"{target_variable} is True")
        plot_histogram_1d(split_plot, false_hist, log=True, label=f"{target_variable} is False")
        split_plot.legend()
        save_figure(split_figure, plotdir, f"{prefix}_bool")

        result_data["efficiency_rejection"] = calculate_rejection(true_hist, false_hist, plotdir, f"{prefix}", title=f"MVA {label}", label=label, rig_title=rig_title)

    bdt.save_model(bdt_model_path)
    with open(os.path.join(resultdir, f"{prefix}.json"), "w") as model_file:
        json.dump(result_data, model_file)

    return bdt, result_data


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-dataset", required=True, nargs=9, help="Label and path to train histograms, events and selections.")
    parser.add_argument("--predict-dataset", required=True, nargs=9, help="Label and path to predict histograms, events and selections.")
    parser.add_argument("--variables", required=True, nargs="+", help="Variables to use in MVA.")
    parser.add_argument("--target-variable", required=True, help="Variable to regress to.")
    parser.add_argument("--resultdir", default="results", help="Directory to store result files in.")
    parser.add_argument("--config", nargs=2, required=True, help="Path to config file and workdir")
    parser.add_argument("--outputprefix", default="mva", help="Outputprefix for results and plots.")
    parser.add_argument("--title", default="MVA", help="MVA title for plots")
    parser.add_argument("--bin", type=int, help="Rigidity bin of this data.")
    parser.add_argument("--binning", default="rigidity_search", help="Rigidity binning by which training is split.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--mva-parameters", nargs=4, action="append", dest="mva_parameters", help="MVA parameter set (name, depth, number of trees, eta)")
    parser.add_argument("--bool", action="store_true", help="Target variable is boolean.")
    parser.add_argument("--mc-weighting-file", required=True, help="Path to weight file (same as used for histograms, for weighting triggers)")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel")

    args = parser.parse_args()

    config_file, workdir = args.config
    config = get_config(config_file)
    binnings = Binnings((config, workdir))

    variables = args.variables
    branches = sorted(set(variables + ["TotalWeight", "RunNumber", "EventNumber", args.target_variable]))

    title = args.title
    rig_title = ""
    if args.bin is not None:
        binning = binnings.special_binnings[args.binning]
        rig_min, rig_max = binning.edges[args.bin], binning.edges[args.bin + 1]
        rig_title = f", {rig_min:.2f} < |R| / GV <= {rig_max:.2f}"
        print(f"Rigidity range {rig_min:.2f} to {rig_max:.2f}")

    mc_weighting = load_mc_weighting(args.mc_weighting_file)
    rigidity_binning = binnings.special_binnings["rigidity_binning"]

    train_dataset = Dataset.load(args.train_dataset, variables, branches, rigidity_binning, mc_weighting)
    predict_dataset = Dataset.load(args.predict_dataset, variables, branches, rigidity_binning, mc_weighting)

    plotdir = args.plotdir
    os.makedirs(plotdir, exist_ok=True)
    os.makedirs(args.resultdir, exist_ok=True)

    print("\n".join(["variables:"] + [f"{index:>3}: {variable}" for index, variable in enumerate(variables)]))

    if args.mva_parameters is not None:
        train_parameter_sets = {name: dict(ntrees=int(ntrees), max_depth=int(depth), eta=float(eta)) for name, depth, ntrees, eta in args.mva_parameters}

        for train_param_name, train_params in train_parameter_sets.items():
            print("BDT", train_param_name, flush=True)

            bdt_plotdir = os.path.join(plotdir, train_param_name)
            bdt, train_results = train_estimator(variables, train_dataset, predict_dataset, args.target_variable, prefix=f"{args.outputprefix}_{train_param_name}", title=f"{title} {train_param_name}", label=f"{train_param_name}", rig_title=rig_title, resultdir=args.resultdir, plotdir=bdt_plotdir, binnings=binnings, nprocesses=args.nprocesses, target_is_bool=args.bool, **train_params)


if __name__ == "__main__":
    main()
