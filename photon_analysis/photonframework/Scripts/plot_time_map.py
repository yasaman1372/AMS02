#!/usr/bin/env python3

from datetime import datetime, timedelta, timezone

import numpy as np
import matplotlib.pyplot as plt

from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d, load_histograms_from_files
from tools.utilities import save_figure

def make_yearly_ticks(begin_time, end_time):
    major_ticks = []
    major_labels = []
    begin = datetime.fromtimestamp(begin_time, tz=timezone.utc)
    end = datetime.fromtimestamp(end_time, tz=timezone.utc)
    dt = datetime(begin.year, 1, 1, tzinfo=timezone.utc)
    while dt < end:
        major_ticks.append(np.datetime64(int(dt.timestamp()), "s"))
        major_labels.append(dt.strftime("%Y"))
        dt = datetime(dt.year + 1, 1, 1, tzinfo=timezone.utc)
    return major_ticks, [], major_labels

def make_monthly_ticks(begin_time, end_time):
    major_ticks = []
    major_labels = []
    begin = datetime.fromtimestamp(begin_time, tz=timezone.utc)
    end = datetime.fromtimestamp(end_time, tz=timezone.utc)
    dt = datetime(begin.year, begin.month, 1, tzinfo=timezone.utc)
    while dt < end:
        major_ticks.append(np.datetime64(int(dt.timestamp()), "s"))
        major_labels.append(dt.strftime("%b %Y"))
        dt = datetime(dt.year + (dt.month // 12), (dt.month % 12) + 1, 1, tzinfo=timezone.utc)
    return major_ticks, [], major_labels

def make_time_ticks(begin_time, end_time):
    total_duration = end_time - begin_time

    if total_duration > 3 * 365 * 86400:
        return make_yearly_ticks(begin_time, end_time)
    if total_duration > 60 * 86400:
        return make_monthly_ticks(begin_time, end_time)

    if total_duration < 60:
        major_step = 10
        minor_step = 1
        major_format = "%M:%S"
    elif total_duration < 600:
        major_step = 60
        minor_step = 10
        major_format = "%M:%S"
    elif total_duration < 3600:
        major_step = 600
        minor_step = 60
        major_format = "%H:%M:%S"
    elif total_duration < 86400:
        major_step = 3600
        minor_step = 600
        major_format = "%H:%M:%S"
    elif total_duration < 3 * 86400:
        major_step = 3 * 3600
        minor_step = 1800
        major_format = "%d.%m.%Y %H:%M"
    elif total_duration < 30 * 86400:
        major_step = 86400
        minor_step = 7200
        major_format = "%d.%m.%Y %H:%M"
    elif total_duration < 365 * 86400:
        major_step = 86400 * 30
        minor_step = 86400
        major_format = "%d.%m.%Y"
    else:
        major_step = 86400 * 365
        minor_step = 86400 * 30
        major_format = "%d.%m.%Y"

    major_ticks = np.arange(begin_time, end_time + major_step, major_step)
    minor_ticks = np.arange(begin_time, end_time + minor_step, minor_step)

    major_labels = [datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(major_format) for timestamp in major_ticks]
    return np.array(major_ticks, "datetime64[s]"), np.array(minor_ticks, "datetime64[s]"), major_labels


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", nargs="+", help="Files to read histograms from.")
    parser.add_argument("--hist-name", default="hist_Time", help="Name of the histogram in the files.")
    parser.add_argument("--plotdir", default="plots", help="Directory to save plots in.")
    parser.add_argument("--outputprefix", default="Time", help="Prefix for plots and result files.")
    parser.add_argument("--title", help="Title to print on each plot.")
    parser.add_argument("--ylabel", help="Label to print on the y axis.")
    parser.add_argument("--threshold", type=float, help="Threshold above which datetimes should be printed.")

    args = parser.parse_args()

    hists = load_histograms_from_files(args.input_files, keys=[args.hist_name])
    time_hist = hists[args.hist_name]

    if args.threshold:
        sel = time_hist.values > args.threshold
        if np.any(sel):
            sel_times = time_hist.binnings[0].bin_centers[sel]
            for sel_time in sel_times:
                print(datetime.fromtimestamp(sel_time).strftime("%d.%m.%Y %H:%M:%S"))

    figure = plt.figure(figsize=(8, 4.2))
    plot = figure.subplots(1, 1)

    plot_histogram_1d(plot, time_hist, style="iss", markersize=3, linewidth=1)
    major_ticks, minor_ticks, major_labels = make_time_ticks(*plot.get_xlim())
    plot.set_xticks(major_ticks, labels=major_labels)
    plot.set_xticks(minor_ticks, minor=True)
    if args.ylabel is not None:
        plot.set_ylabel(args.ylabel)
    if args.title is not None:
        plot.set_title(args.title)

    figure.subplots_adjust(left=0.1, right=0.95, bottom=0.125, top=0.9)
    save_figure(figure, args.plotdir, f"{args.outputprefix}_time")


if __name__ == "__main__":
    main()
