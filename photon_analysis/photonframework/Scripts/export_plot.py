#!/usr/bin/env python3

import os
import pickle
import re

import numpy as np
import matplotlib
matplotlib.rc("text", usetex=False)
matplotlib.rc("pdf", fonttype=42)
matplotlib.rc("ps", fonttype=42)
#matplotlib.rc("font", weight="bold")
#matplotlib.rc("axes", labelweight="bold", titleweight="bold")
#matplotlib.rc("figure", labelweight="bold", titleweight="bold")
import matplotlib.pyplot as plt

plt._get_backend_mod()

MATH_PATTERN = r"\$([^$]+)\$"
MATH_BF_REPL = r"$\\mathbf{\1}$"

def _boldify_math(text):
    return re.subn(MATH_PATTERN, MATH_BF_REPL, text)[0]


def make_bold(figure):
    if figure._suptitle:
        figure.suptitle(_boldify_math(figure._suptitle.get_text()), fontweight="bold", fontname="Arial")
    for plot in figure.get_axes():
        if plot.get_title():
            plot.set_title(_boldify_math(plot.get_title()), fontweight="bold", fontname="Arial")
        if plot.get_xlabel():
            plot.set_xlabel(_boldify_math(plot.get_xlabel()), fontweight="bold", fontname="Arial")
        if plot.get_ylabel():
            plot.set_ylabel(_boldify_math(plot.get_ylabel()), fontweight="bold", fontname="Arial")


def hide_legend_background(figure):
    for plot in figure.get_axes():
        legend = plot.get_legend()
        if legend is not None:
            legend.set(frame_on=False)


def adjust_exponent_notation(figure):
    for plot in figure.get_axes():
        try:
            plot.ticklabel_format(useMathText=True)
        except AttributeError:
            pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfile", help="Path to pickled plot file to export.")
    parser.add_argument("outputfile", help="Path to store plot at.")
    parser.add_argument("--size", type=float, nargs=2, default=(12, 6.15), help="Width and height in inches (default 12x6.15)")
    parser.add_argument("--margin", type=float, nargs=4, default=(0.1, 0.9, 0.1, 0.9), help="Margin left, right, bottom, top")
    parser.add_argument("--dpi", type=int, default=300, help="DPI to export with (default 300).")
    parser.add_argument("--textwidth", type=float, default=5.56691, help="Actual size of the document textwidth in inches.")
    parser.add_argument("--textwidth-fraction", type=float, default=1, help="Fraction of textwidth this plot takes in the target document.")
    parser.add_argument("--hide-legend-background", action="store_true", help="Remove frame around legend.")

    args = parser.parse_args()

    width, height = args.size
    margin_l, margin_r, margin_b, margin_t = args.margin

    textwidth = args.textwidth * args.textwidth_fraction
    dpi = args.dpi * (textwidth / width)

    with open(args.inputfile, "rb") as figure_file:
        figure = pickle.load(figure_file)
    make_bold(figure)
    adjust_exponent_notation(figure)
    if args.hide_legend_background:
        hide_legend_background(figure)
    figure.set_size_inches(args.size)
    figure.subplots_adjust(left=margin_l, right=margin_r, bottom=margin_b, top=margin_t)
    figure.savefig(args.outputfile, dpi=args.dpi)
    plt.close(figure)


if __name__ == "__main__":
    main()
