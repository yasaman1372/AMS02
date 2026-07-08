#!/usr/bin/env python3

import json
import os

import numpy as np

from uncertainties import ufloat

from tools.confidence import calculate_toy_mc_limit_cut_supremum, calculate_combined_confidence_limit_toy_mc_cut


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="inputs", action="append", nargs=2, help="Filename of cut limit result file and cut efficiency.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--outputprefix", default="CutLimit", help="Prefix for result files.")

    args = parser.parse_args()

    iss_background_events_above_cut = 0
    mc_background_events_above_cut_unweighted = 0
    mc_background_events_above_cut = 0
    iss_positive_events = 0

    samples = []

    for filename, efficiency in args.inputs:
        with open(filename) as json_file:
            limit_data = json.load(json_file)
            iss_background_events_above_cut += limit_data[efficiency]["negative_events_above_cut"]
            unweighted_mc_events = limit_data[efficiency]["mc_negative_unweighted_events_above_cut"]
            weighted_mc_events = limit_data[efficiency]["normalized_mc_background_events_above_cut"]
            mc_weight = ufloat(limit_data[efficiency]["mc_to_iss_factor"], limit_data[efficiency]["mc_to_iss_factor_uncertainty"])
            mc_background_events_above_cut_unweighted += unweighted_mc_events
            mc_background_events_above_cut += unweighted_mc_events * mc_weight
            iss_positive_events += limit_data[efficiency]["positive_events_above_cut"]
            samples.append((iss_background_events_above_cut, iss_positive_events, unweighted_mc_events, mc_weight.nominal_value, mc_weight.std_dev))


    average_weight = mc_background_events_above_cut / mc_background_events_above_cut_unweighted

    background_prediction = mc_background_events_above_cut_unweighted
    if iss_background_events_above_cut < mc_background_events_above_cut:
        background_prediction = iss_background_events_above_cut / average_weight.nominal_value

    print(f"{iss_background_events_above_cut:.2f} / {background_prediction*average_weight.nominal_value:.2f}")

    #upper_limit_combined = calculate_combined_confidence_limit_toy_mc_cut(confidence_level=0.95, samples=samples, verbose=True)
    #print(upper_limit_combined)

    upper_limit = calculate_toy_mc_limit_cut_supremum(
        confidence_level=0.95,
        events_above_cut=iss_background_events_above_cut,
        mc_events_above_cut=background_prediction,
        mc_average_weight=average_weight.nominal_value,
        mc_average_weight_error=average_weight.std_dev)

    print(f"CL = {upper_limit:.2f} / {iss_positive_events:.0f} = {upper_limit/iss_positive_events:.4g}")

    result_data = dict(
        negative_events_above_cut=iss_background_events_above_cut,
        positive_events_above_cut=iss_positive_events,
        confidence_limit_events_toy_sup=upper_limit,
        confidence_limit_ratio_toy_sup=upper_limit / iss_positive_events)

    os.makedirs(args.resultdir, exist_ok=True)
    with open(os.path.join(args.resultdir, f"{args.outputprefix}.json"), "w") as result_file:
        json.dump(result_data, result_file)



if __name__ == "__main__":
    main()
