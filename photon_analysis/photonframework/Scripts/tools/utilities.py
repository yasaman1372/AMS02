
from collections import defaultdict
from glob import glob
import os
import pickle
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np
from numpy.lib import recfunctions
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.optimize import curve_fit
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, PowerNorm, LinearSegmentedColormap
from iminuit import Minuit
from iminuit.cost import LeastSquares

from tools.constants import MC_PARTICLE_IDS, MC_PARTICLE_CHARGES, MC_PARTICLE_MASSES, RICH_RADIATOR_BETA, RICH_RADIATOR_RESOLUTION
from tools.conversions import calc_rig, calc_beta
from tools.statistics import fermi_function, inverse_fermi_function, lafferty_whyatt

def load_flux_from_xml(path, gamma=2.7, energy_key="rigidity", flux_key="flux"):
    root = ET.parse(path).getroot()
    header = root.find("HEADER")
    mission = header.findtext("MISSION")
    particle = header.findtext("PARTICLE")
    year = header.findtext("YEAR")
    title = f"{mission} {particle} {year}"
    rigidities = []
    flux = []
    flux_error_low = []
    flux_error_high = []
    for d in root.findall("DATA"):
        rmin = float(d.findtext(f"{energy_key}_min"))
        rmax = float(d.findtext(f"{energy_key}_max"))
        rigidities.append(lafferty_whyatt(np.array([rmin, rmax]), gamma=gamma)[0])
        flux.append(float(d.findtext(flux_key)))
        flux_error_low.append((float(d.findtext(f"{flux_key}_statistical_error_low"))**2 + float(d.findtext(f"{flux_key}_systematical_error_low"))**2)**0.5)
        flux_error_high.append((float(d.findtext(f"{flux_key}_statistical_error_high"))**2 + float(d.findtext(f"{flux_key}_systematical_error_high"))**2)**0.5)
    rigidities, flux, flux_error_low, flux_error_high = map(np.array, (rigidities, flux, flux_error_low, flux_error_high))
    array = np.core.records.fromarrays((rigidities, flux, flux_error_low, flux_error_high, flux, flux_error_low, flux_error_high), names=(energy_key, flux_key, f"{flux_key}_error_low", f"{flux_key}_error_high", f"raw_{flux_key}", f"raw_{flux_key}_error_low", f"raw_{flux_key}_error_high"))
    return array, title


def load_flux(path):
    flux_data = np.genfromtxt(path)
    rigidity = flux_data[:,0]
    flux = flux_data[:,3]
    flux_err_low = flux_data[:,4]
    flux_err_high = flux_data[:,5]
    title = None
    with open(path) as flux_file:
        for line in flux_file:
            if not line.startswith("#"):
                break
            if line.startswith("# GraphTitle:"):
                title = line.split(":", 1)[1].split(";")[0].strip()
    array = np.core.records.fromarrays((rigidity, flux, flux_err_low, flux_err_high, flux, flux_err_low, flux_err_high), names=("rigidity", "flux", "flux_error_low", "flux_error_high", "raw_flux", "raw_flux_error_low", "raw_flux_error_high"))
    return array, title


def power_law(rigidity, phi, c, rigidity_scale, gamma):
    effective_rigidity = rigidity + phi
    return (rigidity / effective_rigidity)**2 * (c * (effective_rigidity / rigidity_scale)**gamma)


def fit_flux(flux_array, flux_title, create_plot=False):
    flux_error = (flux_array.flux_error_low + flux_array.flux_error_high) / 2
    guess = dict(phi=1, c=10, rigidity_scale=10, gamma=-2.7)
    loss = LeastSquares(flux_array.rigidity, flux_array.flux, flux_error, model=power_law)
    m = Minuit(loss, **guess)
    m.migrad()

    fit_param_dict = dict(zip(m.parameters, m.values))

    def _parametrized_power_law(rigidity):
        return power_law(rigidity, **fit_param_dict)

    if create_plot:
        dense_rigidity = np.logspace(np.log10(flux_array.rigidity[0]), np.log10(flux_array.rigidity[-1]), 100)
        flux_figure = plt.figure(figsize=(12, 6.15))
        flux_figure.suptitle(flux_title)
        flux_plot = flux_figure.subplots(1, 1)
        flux_plot.errorbar(flux_array.rigidity, flux_array.flux, yerr=np.stack((flux_array.flux_error_low, flux_array.flux_error_high)), fmt=".", label="Flux")
        flux_plot.plot(dense_rigidity, power_law(dense_rigidity, **guess), "-", label="Guess")
        flux_plot.plot(dense_rigidity, power_law(dense_rigidity, **fit_param_dict), "-", label="Fit")
        flux_plot.set_xscale("log")
        flux_plot.set_yscale("log")
        flux_plot.set_xlabel("Rigidity / GV")
        flux_plot.set_ylabel("Flux")
        flux_plot.legend()
        flux_figure.savefig("flux-fit.png", dpi=250)
        plt.close(flux_figure)

    return _parametrized_power_law, fit_param_dict


def load_mc_trigger_density(mc_triggers_filename):
    with open(mc_triggers_filename) as mc_triggers_file:
        components = []
        for line in mc_triggers_file:
            rig_min, rig_max, triggers = map(float, line.split(" "))
            components.append((abs(rig_min), abs(rig_max), triggers))

    def _trigger_density(rigidity):
        cumulative_density = 0
        return np.sum([
            ((rig_min <= rigidity) & (rig_max >= rigidity)) * triggers / (np.log(rig_max) - np.log(rig_min)) / rigidity
            for rig_min, rig_max, triggers in components
        ], axis=0)

    return _trigger_density


def load_mc_trigger_count(mc_triggers_filename, binning):
    from tools.histograms import Histogram
    mc_triggers_hist = Histogram(binning)
    with open(mc_triggers_filename) as mc_triggers_file:
        for line in mc_triggers_file:
            rig_min, rig_max, triggers = map(float, line.split(" "))
            rig_min, rig_max = map(abs, (rig_min, rig_max))
            min_index = np.digitize(rig_min, binning.edges) - 1
            max_index = np.digitize(rig_max, binning.edges)
            rig_in_range = binning.edges[min_index:max_index + 1]
            for index, (bin_min, bin_max) in enumerate(zip(rig_in_range[:-1], rig_in_range[1:])):
                fraction = (np.log10(min(bin_max, rig_max)) - np.log10(max(bin_min, rig_min))) / (np.log10(rig_max) - np.log10(rig_min))
                mc_triggers_hist.values[min_index + index] += fraction * triggers
    return mc_triggers_hist


def load_weighted_mc_trigger_count(mc_triggers_filename, binning, mc_weighting):
    from tools.histograms import Histogram
    from tools.binnings import Binning, increase_bin_number
    fine_binning = increase_bin_number(binning, 10)
    mc_triggers_hist = Histogram(fine_binning)
    with open(mc_triggers_filename) as mc_triggers_file:
        for line in mc_triggers_file:
            rig_min, rig_max, triggers = map(float, line.split(" "))
            rig_min, rig_max = map(abs, (rig_min, rig_max))
            min_index = np.digitize(rig_min, fine_binning.edges) - 1
            max_index = np.digitize(rig_max, fine_binning.edges)
            rig_in_range = fine_binning.edges[min_index:max_index + 1]
            for index, (bin_min, bin_max) in enumerate(zip(rig_in_range[:-1], rig_in_range[1:])):
                eff_min = max(bin_min, rig_min)
                eff_max = min(bin_max, rig_max)
                fraction = (np.log10(eff_max) - np.log10(eff_min)) / (np.log10(rig_max) - np.log10(rig_min))
                weight = mc_weighting.get_weights(None, np.array((eff_min + eff_max) / 2)).item()
                mc_triggers_hist.values[min_index + index] += fraction * triggers * weight
    return mc_triggers_hist.rebin(binning)



def plot_steps(plot, edges, values, **kwargs):
    x = np.concatenate(([edges[0]], edges))
    y = np.concatenate(([0], values, [0]))
    return plot.step(x, y, where="post", **kwargs)

def shaded_steps(plot, edges, values_low=None, values_high=None, values=None, errors=None, **kwargs):
    if values is not None and errors is not None:
        values_low = values - errors
        values_high = values + errors
    if values_low is None and values_high is None:
        raise ValueError("Either values_low and values_high or values and errors is required!")
    x = np.concatenate(([edges[0]], edges))
    y_low = np.concatenate(([0], values_low, [0]))
    y_high = np.concatenate(([0], values_high, [0]))
    return plot.fill_between(x, y_low, y_high, step="post", **kwargs)


def round_up(value, digits=1, log=False):
    if log:
        return 10**(round_up(np.log10(value), digits=digits))
    if value == 0:
        return 0
    if value < 0:
        return -round_down(-value)
    factor = 10**digits
    return np.ceil(value * factor) / factor

def round_down(value, digits=1, log=False):
    if log:
        return 10**(round_down(np.log10(value), digits=digits))
    if value == 0:
        return 0
    if value < 0:
        return -round_up(-value)
    factor = 10**digits
    return np.floor(value * factor) / factor


def superscript(value):
    digits = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻", ".": "·"}
    s = str(value)
    return "".join((digits.get(c, c) for c in s))

def format_order_of_magnitude(value, digits=2, use_tex=False):
    if value == 0:
        return "0"
    elif abs(value) >= 1 and abs(value) < 1000:
        if value == int(value):
            return str(int(value))
        if abs(value) < 10:
            return f"{value:.2f}"
        if abs(value) < 100:
            return f"{value:.1f}"
        return f"{value:.0f}"
    elif abs(value) < 1 and abs(value) >= 0.1:
        return f"{value:.2f}"
    elif abs(value) < 0.1 and abs(value) >= 0.01:
        return f"{value:.3f}"
    sign = np.sign(value)
    value = value * sign
    magnitude = int(np.floor(np.log10(value)))
    mantisse = value / 10**magnitude
    sign_str = "" if sign > 0 else "-"
    if use_tex:
        return f"\\ensuremath{{{sign_str}{mantisse:.{digits}f}\\times 10^{{{magnitude}}}}}"
    return f"{sign_str}{mantisse:.{digits}f}×10{superscript(str(magnitude))}"


def format_human(value):
    order_of_magnitude = int(np.log10(value))
    prefix = np.round(value / 10**order_of_magnitude)
    rounded_value = prefix * 10**order_of_magnitude
    if order_of_magnitude < 3:
        return f"{rounded_value:.0f}"
    if order_of_magnitude < 6:
        return f"{rounded_value / int(1e3):.0f} Thousand"
    if order_of_magnitude < 9:
        return f"{rounded_value / int(1e6):.0f} Million"
    if order_of_magnitude < 12:
        return f"{rounded_value / int(1e9):.0f} Billion"
    return f"{rounded_value:.0g}"


def set_plot_lim_x(plot, values, log=False, override=False):
    xmin_old, xmax_old = plot.get_xlim()
    xmax_new = xmax_old
    if np.any(np.isfinite(values)):
        xmax_new = round_up(np.max(values[np.isfinite(values)]))
    if override:
        xmax = xmax_new
    else:
        xmax = max(xmax_old, xmax_new)
    xmin_new = 0
    if log:
        xmin_new = xmin_old
        if np.any(np.isfinite(values) & (values > 0)):
            xmin_new = round_down(np.min(values[np.isfinite(values) & (values > 0)]))
    if override:
        xmin = xmin_new
    else:
        xmin = min(xmin_old, xmin_new)
    if not log and xmin < 0 and xmin_new == 0:
        xmin = 0
    plot.set_xlim(left=xmin, right=xmax)

def set_plot_lim_y(plot, values, log=False, override=False):
    ymin_old, ymax_old = plot.get_ylim()
    ymax_new = ymax_old
    if np.any(np.isfinite(values)):
        ymax_new = round_up(np.max(values[np.isfinite(values)]))
    if override:
        ymax = ymax_new
    else:
        ymax = max(ymax_old, ymax_new)
    ymin_new = 0
    if log:
        ymin_new = ymin_old
        if np.any(np.isfinite(values) & (values > 0)):
            ymin_new = round_down(np.min(values[np.isfinite(values) & (values > 0)]))
    if override:
        ymin = ymin_new
    else:
        ymin = min(ymin_old, ymin_new)
    if not log and ymin < 0 and ymin_new == 0:
        ymin = 0
    plot.set_ylim(bottom=ymin, top=ymax)

def set_plot_lim(plot, values, log=False, axis="y", override=False):
    if axis == "x":
        set_plot_lim_x(plot, values, log, override=override)
    elif axis == "y":
        set_plot_lim_y(plot, values, log, override=override)


def set_energy_ticks_old(plot, axis="x"):
    axis = plot.xaxis if axis == "x" else plot.yaxis
    xmin, xmax = axis.get_view_interval()
    xticks = [t for t in (0.1, 1, 10, 100, 1000, 10000, 100000) if xmin <= t <= xmax]
    labels = [str(t) for t in xticks]
    axis.set_ticks(xticks)
    axis.set_ticklabels(labels)
    def _label(tick):
        if tick / 10**(int(np.log10(tick))) > 5:
            return ""
        if (tick - int(tick)) / tick < 1e-10:
            return str(int(tick))
        if tick < 1:
            return f"{tick:.1f}"
        return str(tick)
    if len(xticks) <= 2:
        minor_xticks = [t * f for t in (1, 10, 100, 1000) for f in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) if xmin <= t * f <= xmax]
        minor_labels = [_label(tick) for tick in minor_xticks]
        axis.set_ticks(minor_xticks, minor=True)
        axis.set_ticklabels(minor_labels, minor=True)



def get_rigidity_ticks(xmin, xmax, max_labels=11, weight_label_width=True):
    min_magnitude = np.floor(np.log10(xmin))
    max_magnitude = np.ceil(np.log10(xmax))

    xticks = list(10.0**np.arange(min_magnitude, max_magnitude + 1))
    xticks = [t for t in xticks if xmin <= t <= xmax]

    def _label(tick):
        if (tick - int(tick)) / tick < 1e-10:
            return str(int(tick))
        if tick > 1000:
            magnitude = np.floor(np.log10(tick))
            prefix = tick / 10**magnitude
            digits = 1
            if prefix == 1:
                return f"$10^{{{magnitude:.0f}}}$"
            if (prefix - np.round(prefix)) / prefix < 1e-10:
                digits = 0
            return f"${prefix:.{digits}f}\\times 10^{{{magnitude:.0f}}}$"
        if tick >= 1:
            return str(tick)
        if tick > 0.1:
            return f"{tick:.1f}"
        if tick > 0.01:
            return f"{tick:.2f}"
        magnitude = np.floor(np.log10(tick))
        prefix = tick / 10**magnitude
        digits = 1
        if prefix == 1:
            return f"$10^{{{magnitude:.0f}}}$"
        if (prefix - np.round(prefix)) / prefix < 1e-10:
            digits = 0
        return f"${prefix:.{digits}f}\\times 10^{{{magnitude:.0f}}}$"

    major_xticks = xticks
    major_labels = [_label(t) for t in xticks]
    minor_xticks = []
    minor_labels = []

    if len(xticks) <= 2:
        minor_xticks = [t * f for t in 10**np.arange(min_magnitude, max_magnitude + 1) for f in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) if xmin <= t * f <= xmax]
        minor_labels = [_label(tick) for tick in minor_xticks]
        label_dict = dict(zip(minor_xticks, minor_labels))
        label_length_dict = {tick: len(label) for tick, label in zip(minor_xticks, minor_labels)} | {tick: len(label) for tick, label in zip(major_xticks, major_labels)}
        safe_ticks = set(major_xticks)
        all_ticks = sorted(major_xticks + minor_xticks)
        if all_ticks[0] not in safe_ticks and (len(all_ticks) <= 1 or all_ticks[1] not in safe_ticks):
            safe_ticks.add(all_ticks[0])
        if all_ticks[-1] not in safe_ticks and (len(all_ticks) <= 1 or all_ticks[-2] not in safe_ticks):
            safe_ticks.add(all_ticks[-1])
        while label_dict and len(all_ticks) > len(safe_ticks) and len(label_dict) > max_labels - len(major_xticks):
            tick_values = np.log(np.array(all_ticks))
            label_lengths = np.array([label_length_dict[tick] for tick in all_ticks])
            deltas = (tick_values[1:] - tick_values[:-1]) / ((label_lengths[1:] + label_lengths[:-1]) if weight_label_width else 1)
            min_delta = np.minimum(deltas[1:], deltas[:-1])
            candidates = sorted(zip(all_ticks[1:-1], min_delta), key=lambda t: t[1])
            best = next((c for c, _ in candidates if c not in safe_ticks))
            label_dict.pop(best)
            all_ticks.remove(best)

        minor_labels = [label_dict.get(tick, "") for tick in minor_xticks]

    return major_xticks, major_labels, minor_xticks, minor_labels


def set_energy_ticks(plot, axis="x", max_labels=11):
    plot_axis = plot.xaxis if axis == "x" else plot.yaxis
    xmin, xmax = plot_axis.get_view_interval()

    major_xticks, major_labels, minor_xticks, minor_labels = get_rigidity_ticks(xmin, xmax, max_labels=max_labels, weight_label_width=axis == "x")

    plot_axis.set_ticks(major_xticks)
    plot_axis.set_ticklabels(major_labels)
    if minor_xticks:
        plot_axis.set_ticks(minor_xticks, minor=True)
        plot_axis.set_ticklabels(minor_labels, minor=True)



def plot_2d(plot, values, edges_x, edges_y, min_value=None, max_value=None, cmap=None, log=False, sqrt=False, colorbar=True, colorbar_ax=None, colorbar_width=0.05, colorbar_label=None, colorbar_orientation=None, scale=None, mask_zeros=True, mask_below=None, **kwargs):
    values_z = values.transpose()
    if scale is not None:
        values_z = values_z * scale
    mask = False
    if mask_below is not None:
        mask = values_z <= mask_below
    elif mask_zeros:
        mask = values_z == 0
    values_z = np.ma.masked_where(mask, values_z)
    if min_value is not None and max_value is not None and max_value < min_value:
        max_value = min_value
    if "edgecolor" not in kwargs and "linewidth" not in kwargs:
        kwargs["edgecolor"] = "face"
        kwargs["linewidth"] = 0.1
    if log:
        norm = LogNorm(min_value, max_value)
    elif sqrt:
        norm = PowerNorm(0.5, min_value, max_value)
    else:
        norm = Normalize(min_value, max_value)
    mesh = plot.pcolormesh(edges_x, edges_y, values_z, cmap=cmap, norm=norm, antialiased=True, rasterized=True, **kwargs)
    if colorbar:
        cbar = plt.colorbar(mesh, ax=plot, cax=colorbar_ax, fraction=colorbar_width, label=colorbar_label, orientation=colorbar_orientation)
        return cbar

def hist_most_probable_value(histogram, axis=1, with_overflow=False):
    values = histogram.values
    centers = histogram.binnings[axis].bin_centers
    offset = 0
    if not with_overflow:
        values = values[tuple([slice(1, -1) for axis in range(histogram.dimensions)])]
        offset = 1
        centers = centers[1:-1]
    indices = np.argmax(values, axis=axis) + offset
    return centers[indices - 1]


def transform_overflow_edges(edges):
    underflow = 2 * edges[1] - edges[2]
    overflow = 2 * edges[-2] - edges[-3]
    return np.array([underflow] + list(edges[1:-1]) + [overflow])


def make_colormap_passed():
    red = ((0.0, 0.2, 0.2), (1.0, 0.0, 0.0))
    green = ((0.0, 1.0, 1.0), (1.0, 0.2, 0.2))
    blue = ((0.0, 0.2, 0.2), (1.0, 0.0, 0.0))
    return LinearSegmentedColormap("passed", {"red": red, "green": green, "blue": blue})

def make_colormap_failed():
    red = ((0.0, 1.0, 1.0), (1.0, 0.2, 0.2))
    green = ((0.0, 0.2, 0.2), (1.0, 0.0, 0.0))
    blue = ((0.0, 0.2, 0.2), (1.0, 0.0, 0.0))
    return LinearSegmentedColormap("failed", {"red": red, "green": green, "blue": blue})

def make_downwards_colormap_rgb(name, weight_red, weight_green, weight_blue, base=0.6):
    r0 = base + (1 - base) * weight_red
    r1 = base * weight_red
    g0 = base + (1 - base) * weight_green
    g1 = base * weight_green
    b0 = base + (1 - base) * weight_blue
    b1 = base * weight_blue
    red = ((0.0, r0, r0), (1.0, r1, r1))
    green = ((0.0, g0, g0), (1.0, g1, g1))
    blue = ((0.0, b0, b0), (1.0, b1, b1))
    return LinearSegmentedColormap(name, dict(red=red, green=green, blue=blue))


PARTICLE_COLORMAPS = {
    "Default": make_downwards_colormap_rgb("Default", 0.5, 0.5, 0.5),
    "Positron": make_downwards_colormap_rgb("Positron", 1, 0.1, 0),
    "Electron": make_downwards_colormap_rgb("Electron", 1, 0.2, 0),
    "Proton": make_downwards_colormap_rgb("Proton", 0, 0, 1),
    "Antiproton": make_downwards_colormap_rgb("Antiproton", 0, 0.1, 1),
    "Alpha": make_downwards_colormap_rgb("He4", 0, 1, 0),
    "He3": make_downwards_colormap_rgb("He3", 0.25, 1, 0),
    "Li6": make_downwards_colormap_rgb("Li6", 1, 0.75, 0),
    "Li7": make_downwards_colormap_rgb("Li7", 0.75, 1, 0),
    "C12": make_downwards_colormap_rgb("C12", 0, 1, 0.8),
    "N14": make_downwards_colormap_rgb("N14", 0, 0.8, 1),
}


def translate_parameter_name(name):
    return name.replace("mu", "\\mu").replace("sigma", "\\sigma").replace("delta", "\\Delta").replace("tau", "\\tau")


def rec_to_float(recarray):
    result = np.zeros((len(recarray), (len(recarray.dtype.names))), dtype=np.float32)
    for index, column in enumerate(recarray.dtype.names):
        result[:,index] = recarray[column]
    return result


def float_or_int(value):
    if float(value).is_integer():
        return int(value)
    return float(value)


def filter_branches(array, branches):
    if isinstance(array, ak.Array):
        return np.core.records.fromarrays([ak.to_numpy(array[branch]) for branch in branches], names=branches)
    return recfunctions.require_fields(array, [(name, array.dtype[name]) for name in branches])


def title_to_name(title):
    return re.sub("_+", "_", "".join(c if c.isalnum() else "_" for c in title))


BR_LENGTH = 27 * 86400
BR_2394 = 1230768000

def start_time_of_bartels_rotation(br_number):
    return BR_2394 + BR_LENGTH * (br_number - 2394)


def sort_by_dependencies(elements, dependencies):
    result = []
    done = set()
    remaining = set(elements)
    while remaining:
        size = len(remaining)
        to_remove = []
        for element in sorted(remaining):
            if all(dep in done or dep not in elements for dep in dependencies[element]):
                result.append(element)
                done.add(element)
        remaining = remaining - done
        if len(remaining) == size:
            raise ValueError(f"Cannot resolve order of remaining elements: {remaining!r}")
    return result


def resolve_derived_branches(required_branches, dependencies, derivation_functions):
    derived_branches = set()
    while True:
        required_after, newly_derived = resolve_derived_branches_step(required_branches, dependencies, derivation_functions)
        derived_branches = derived_branches | newly_derived
        if required_after == required_branches:
            return required_branches, sort_by_dependencies(derived_branches, dependencies)
        required_branches = required_after


def resolve_derived_branches_step(required_branches, dependencies, derivation_functions):
    required_branches = set(required_branches)
    derived_branches = {branch for branch in required_branches if branch in derivation_functions}
    dependency_branches = {dependency for branch in derived_branches for dependency in dependencies[branch]}
    return (required_branches - derived_branches) | dependency_branches, derived_branches


def recursive_dependencies(variable, dependencies):
    vars = dependencies.get(variable, [])
    dependency_vars = []
    for var in vars:
        dependency_vars.extend(recursive_dependencies(var, dependencies))
    return set(vars) | set(dependency_vars)


def decompose_graph(nodes, edges):
    nodes_in_graph = set()
    edges_in_graph = set()
    nodes_to_check = []
    while nodes:
        node = nodes[0]
        nodes_in_graph.add(node)
        nodes_to_check.append(node)
        while nodes_to_check:
            node = nodes_to_check.pop(0)
            for edge in edges:
                if edge[0] == node:
                    other = edge[1]
                elif edge[1] == node:
                    other = edge[0]
                else:
                    continue
                edges_in_graph.add(edge)
                if other not in nodes_in_graph:
                    nodes_in_graph.add(other)
                    nodes_to_check.append(other)
        yield nodes_in_graph, edges_in_graph
        nodes = [node for node in nodes if node not in nodes_in_graph]
        edges = [edge for edge in edges if edge not in edges_in_graph]
        nodes_in_graph = set()
        edges_in_graph = set()


class Palette:
    def __init__(self, colors, index=None):
        self.colors = colors
        if index is None:
            index = 0
        self.index = index

    def get_color(self):
        color = self.colors[self.index % len(self.colors)]
        self.index += 1
        return color

    def reset(self):
        self.index = 0

def make_tab_palette():
    return Palette(("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"))


def check_binning_compatibility(points_1, points_2, precision=1e-7):
    low_end = max(points_1[0], points_2[0])
    high_end = min(points_1[-1], points_2[-1])
    points_1_in = points_1[(points_1 >= low_end) & (points_1 <= high_end)]
    points_2_in = points_2[(points_2 >= low_end) & (points_2 <= high_end)]
    if len(points_1_in) != len(points_2_in):
        return False
    if np.any(np.abs(points_1 - points_2) > precision): 
        return False
    return True

def get_compatible_binpoints(points_1, points_2, precision=1e-7):
    low_end = max(points_1[0], points_2[0])
    high_end = min(points_1[-1], points_2[-1])
    sel_1 = (points_1 >= low_end) & (points_1 <= high_end)
    sel_2 = (points_2 >= low_end) & (points_2 <= high_end)
    assert np.all(np.abs(points_1[sel_1] - points_2[sel_2]) < precision)
    return points_1[sel_1], sel_1, sel_2


def local_power_law(x, a, gamma):
    return a * x**gamma

def interpolate_power_law(x, y, y_error, target_rigs, window_size=3):
    result = []
    for rig in target_rigs:
        closest = np.argmin(np.abs(x - rig))
        first_bin = max(0, closest - window_size)
        last_bin = min(len(x), closest + window_size)
        fit_params, fit_param_cov = curve_fit(local_power_law, x[first_bin:last_bin], y[first_bin:last_bin], sigma=y_error[first_bin:last_bin], absolute_sigma=True)
        result.append(local_power_law(rig, *fit_params))
    return np.array(result)
    

def compare_fluxes(fluxes, plotdir, prefix, title, energy_estimator, max_diff=0.15):
    assert len(fluxes) > 1
    ref_flux = fluxes[0]
    comp_fluxes = fluxes[1:]

    ref_flux_rigidities, ref_flux_values, ref_flux_errors, ref_flux_label = ref_flux
    
    figure = plt.figure(figsize=(12, 6.15))
    figure.suptitle(title)
    plot = figure.subplots(1, 1)
    plot.set_xlabel(f"{energy_estimator} / GV")
    plot.set_ylabel(f"Ratio to {ref_flux_label}")
    plot.set_ylim(1 - max_diff, 1 + max_diff)
    plot.set_xscale("log")
    plot.plot(ref_flux_rigidities, np.ones_like(ref_flux_rigidities), "-", color="gray")
    for comp_flux_rigidities, comp_flux_values, comp_flux_errors, comp_flux_label in comp_fluxes:
        if check_binning_compatibility(ref_flux_rigidities, comp_flux_rigidities):
            rig_points, ref_sel, comp_sel = get_compatible_binpoints(ref_flux_rigidities, comp_flux_rigidities)
            ratio = comp_flux_values[comp_sel] / ref_flux_values[ref_sel]
            ratio_error = ratio * np.sqrt((comp_flux_errors[comp_sel] / comp_flux_values[comp_sel])**2 + (ref_flux_errors[ref_sel] / ref_flux_values[ref_sel])**2)
            plot.errorbar(rig_points, ratio, ratio_error, fmt=".", label=f"{comp_flux_label}") 
        else:
            interpolated_flux_values = interpolate_power_law(ref_flux_rigidities, ref_flux_values, ref_flux_errors, comp_flux_rigidities)
            ratio = comp_flux_values / interpolated_flux_values
            ratio_error = ratio * np.sqrt((comp_flux_errors / comp_flux_values)**2)
            plot.errorbar(comp_flux_rigidities, ratio, ratio_error, fmt=".", label=f"{comp_flux_label}")
    plot.legend()
    set_energy_ticks(plot)
    save_figure(figure, plotdir, f"{prefix}_comparison")


def rigidity_from_radiator_and_particle(radiator_name, particle_name):
    beta = RICH_RADIATOR_BETA[radiator_name]
    particle_id = MC_PARTICLE_IDS[particle_name]
    charge = MC_PARTICLE_CHARGES[particle_id]
    mass = MC_PARTICLE_MASSES[particle_id]
    return calc_rig(beta, mass, charge)

def rigidity_from_beta_resolution(radiator_name, particle_name_1, particle_name_2, precision=1e-4):
    particle_id_1 = MC_PARTICLE_IDS[particle_name_1]
    particle_id_2 = MC_PARTICLE_IDS[particle_name_2]
    charge_1 = abs(MC_PARTICLE_CHARGES[particle_id_1])
    charge_2 = abs(MC_PARTICLE_CHARGES[particle_id_2])
    mass_1 = MC_PARTICLE_MASSES[particle_id_1] + 1e-7
    mass_2 = MC_PARTICLE_MASSES[particle_id_2] + 1e-7
    if abs(charge_1 / mass_1 - charge_2 / mass_2) < 1e-2:
        return 0
    beta_resolution = RICH_RADIATOR_RESOLUTION[radiator_name]
    rig = 1
    step = 1
    dir = 1
    for _ in range(100):
        delta_beta = abs(calc_beta(rig, mass_1, charge_1) - calc_beta(rig, mass_2, charge_2))
        if abs((delta_beta - beta_resolution) / delta_beta) <= precision:
            #print(f"{particle_name_1} {particle_name_2} {radiator_name} converged to {rig:.2f}")
            return rig
        if delta_beta > beta_resolution:
            if dir < 0:
                step /= 2
            dir = 1
        else:
            if dir > 0:
                step /= 2
            dir = -1
            if step >= rig:
                step /= 2
        rig += dir * step
    raise ValueError("Max iterations reached.")


PATTERNS = {
    r"R\((?P<radiator_name>NaF|AGL),(?P<particle_name>[A-Za-z0-9]+)\)": rigidity_from_radiator_and_particle,
    r"RDeltaBeta\((?P<radiator_name>NaF|AGL),(?P<particle_name_1>[A-Za-z0-9]+),(?P<particle_name_2>[A-Za-z0-9]+)\)": rigidity_from_beta_resolution,
}

def parse_rigidity_value(raw_value):
    if isinstance(raw_value, int) or isinstance(raw_value, float):
        return raw_value
    elif isinstance(raw_value, str):
        for pattern, function in PATTERNS.items():
            match = re.fullmatch(pattern, raw_value)
            if match is not None:
                return function(**match.groupdict())
        raise ValueError(f"Cannot parse rigidity expression {raw_value!r}!")
    raise ValueError(f"Cannot understand rigidity value {raw_value!r}!")


def parse_bool(bool_str):
    return bool_str.lower() in ("true", "yes", "y")


def save_figure(figure, plotdir, prefix, dpi=250, save_png=True, save_pdf=False, save_pickle=False, close_figure=True, transparent=False):
    if save_png:
        figure.savefig(os.path.join(plotdir, f"{prefix}.png"), dpi=dpi, transparent=transparent)
    if save_pdf:
        figure.savefig(os.path.join(plotdir, f"{prefix}.pdf"), dpi=dpi, transparent=transparent)
    if save_pickle:
        with open(os.path.join(plotdir, f"{prefix}.pck"), "wb") as pickle_file:
            pickle.dump(figure, pickle_file)
    if close_figure:
        plt.close(figure)


def plot_feature_importance(bdt, variables, title, plotdir, prefix, labelling=None):
    ncuts = bdt.get_score(importance_type="weight")
    gain = bdt.get_score(importance_type="gain")
    ncuts = {var: ncuts.get(var, 0) for var in variables}
    gain = {var: gain.get(var, 0) for var in variables}
    ncuts_arr = np.array([ncuts[var] for var in variables])
    gain_arr = np.array([gain[var] for var in variables])
    total_gain_arr = ncuts_arr * gain_arr
    total_gain = {var: tg for var, tg in zip(variables, total_gain_arr)}
    indices = np.arange(len(variables))

    def _make_labels(vars):
        if labelling is None:
            return vars
        return [labelling.get_label(var) for var in vars]

    gain_figure = plt.figure(figsize=(16, 8.2))
    gain_figure.suptitle(title)
    gain_plot = gain_figure.subplots(1, 1, gridspec_kw=dict(left=0.33))
    gain_plot.set_xlabel("Avg. Gain")
    variables_by_gain = sorted(variables, key=lambda v: gain[v])
    gain_plot.barh(indices, [gain[var] for var in variables_by_gain], tick_label=_make_labels(variables_by_gain))
    save_figure(gain_figure, plotdir, f"{prefix}_gain")

    ncuts_figure = plt.figure(figsize=(16, 8.2))
    ncuts_figure.suptitle(title)
    ncuts_plot = ncuts_figure.subplots(1, 1, gridspec_kw=dict(left=0.33))
    ncuts_plot.set_xlabel("Occurance")
    variables_by_ncuts = sorted(variables, key=lambda v: ncuts[v])
    ncuts_plot.barh(indices, [ncuts[var] for var in variables_by_ncuts], tick_label=_make_labels(variables_by_ncuts))
    save_figure(ncuts_figure, plotdir, f"{prefix}_ncuts")

    total_gain_figure = plt.figure(figsize=(16, 8.2))
    total_gain_figure.suptitle(title)
    total_gain_plot = total_gain_figure.subplots(1, 1)
    total_gain_plot.set_xlabel("Total Gain")
    variables_by_total_gain = sorted(variables, key=lambda v: total_gain[v])
    total_gain_plot.barh(indices, [total_gain[var] for var in variables_by_total_gain], tick_label=_make_labels(variables_by_total_gain))
    total_gain_figure.subplots_adjust(left=0.2, right=0.95)
    save_figure(total_gain_figure, plotdir, f"{prefix}_total_gain")

    ncuts_gain_figure = plt.figure(figsize=(16, 8.2))
    ncuts_gain_figure.suptitle(title)
    ncuts_gain_plot_gain = ncuts_gain_figure.subplots(1, 1, gridspec_kw=dict(left=0.33))
    ncuts_gain_plot_ncuts = ncuts_gain_plot_gain.twiny()
    ncuts_gain_plot_gain.set_xlabel("Avg. Gain")
    ncuts_gain_plot_ncuts.set_xlabel("Occurance")
    ncuts_gain_plot_gain.barh(indices, [gain[var] for var in variables_by_gain], tick_label=_make_labels(variables_by_gain), height=0.4, label="Avg. Gain", color="tab:blue")
    ncuts_gain_plot_ncuts.barh(indices + 0.5, [ncuts[var] for var in variables_by_gain], tick_label=_make_labels(variables_by_gain), height=0.4, label="Occurance", color="tab:orange")
    ncuts_gain_figure.legend()
    save_figure(ncuts_gain_figure, plotdir, f"{prefix}_ncuts_gain")

    ncuts_gain_2d_figure = plt.figure(figsize=(16, 8.2))
    ncuts_gain_2d_figure.suptitle(title)
    ncuts_gain_2d_plot = ncuts_gain_2d_figure.subplots(1, 1)
    ncuts_gain_2d_plot.set_xlabel("Occurance")
    ncuts_gain_2d_plot.set_ylabel("Avg. Gain")
    ncuts_gain_2d_plot.set_yscale("log")
    ncuts_gain_2d_plot.plot(ncuts_arr, gain_arr, ".")
    for variable, ncuts, gain in zip(variables, ncuts_arr, gain_arr):
        ncuts_gain_2d_plot.text(ncuts, gain, variable, ha="center")
    save_figure(ncuts_gain_2d_figure, plotdir, f"{prefix}_ncuts_gain_2d")
    return total_gain


def sort_contour_points(line_segments):
    connections = defaultdict(lambda: [])
    for first, second in line_segments:
        connections[first].append(second)
        connections[second].append(first)
    assert all(len(points) % 2 == 0 for points in connections.values())

    while connections:
        line = []
        start_point = list(connections)[0]
        current_point = start_point
        line.append(current_point)
        while len(line) == 1 or current_point != start_point:
            next_point = connections[current_point][0]
            connections[current_point].remove(next_point)
            connections[next_point].remove(current_point)
            current_point = next_point
            line.append(current_point)
        point_array = np.array(line)
        yield point_array[:,0], point_array[:,1]
        connections = {key: value for key, value in connections.items() if value}


def create_histogram_contour(histogram, target_efficiency=0.9):
    assert histogram.dimensions == 2
    assert 0 <= target_efficiency <= 1

    total_events = histogram.values.sum()
    sorted_counts = np.sort(histogram.values.flatten())[::-1]
    cumulative = np.cumsum(sorted_counts) / total_events
    cutoff_index = np.argmin(np.abs(cumulative - target_efficiency))
    cutoff_value = sorted_counts[cutoff_index]

    edges_x = transform_overflow_edges(histogram.binnings[1].edges)
    edges_y = transform_overflow_edges(histogram.binnings[0].edges)
    indices_x, indices_y = np.meshgrid(np.arange(len(histogram.binnings[1]) + 2), np.arange(len(histogram.binnings[0]) + 2))
    values = np.zeros((len(edges_y) + 1, len(edges_x) + 1))
    values[1:-1,1:-1] = histogram.values
    selection_2d = values >= cutoff_value

    vertical_edge_left_indices_x = indices_x[:,:-1]
    vertical_edge_right_indices_x = indices_x[:,1:]
    vertical_edge_indices_y = indices_y[:,:-1]
    vertical_edge = selection_2d[vertical_edge_indices_y,vertical_edge_left_indices_x] != selection_2d[vertical_edge_indices_y,vertical_edge_right_indices_x]
    vertical_edge_x = vertical_edge_right_indices_x[vertical_edge]
    vertical_edge_y = vertical_edge_indices_y[vertical_edge]
    vertical_coords_x = edges_x[vertical_edge_x - 1]
    vertical_coords_y_start = edges_y[vertical_edge_y - 1]
    vertical_coords_y_stop = edges_y[vertical_edge_y]

    horizontal_edge_left_indices_y = indices_y[:-1,:]
    horizontal_edge_right_indices_y = indices_y[1:,:]
    horizontal_edge_indices_x = indices_x[:-1,:]
    horizontal_edge = selection_2d[horizontal_edge_left_indices_y,horizontal_edge_indices_x] != selection_2d[horizontal_edge_right_indices_y,horizontal_edge_indices_x]
    horizontal_edge_x = horizontal_edge_indices_x[horizontal_edge]
    horizontal_edge_y = horizontal_edge_right_indices_y[horizontal_edge]
    horizontal_coords_x_start = edges_x[horizontal_edge_x - 1]
    horizontal_coords_x_stop = edges_x[horizontal_edge_x]
    horizontal_coords_y = edges_y[horizontal_edge_y - 1]

    for coords_x, coords_y in sort_contour_points([((x, y_start), (x, y_stop)) for x, y_start, y_stop in zip(vertical_coords_x, vertical_coords_y_start, vertical_coords_y_stop)] + [((x_start, y), (x_stop, y)) for x_start, x_stop, y in zip(horizontal_coords_x_start, horizontal_coords_x_stop, horizontal_coords_y)]):
        yield coords_y, coords_x


def load_histograms(pattern, variables):
    from .histograms import WeightedHistogram
    histograms = {}
    histograms_per_rig = {}
    filenames = list(glob(pattern))
    if not filenames:
        raise FileNotFoundError(f"No file matches {pattern!r}")
    for filename in filenames:
        with np.load(filename) as hist_file:
            for variable in variables:
                hist = WeightedHistogram.from_file(hist_file, f"hist_{variable}")
                hist_per_rig = WeightedHistogram.from_file(hist_file, f"hist_per_rig_{variable}")
                if variable not in histograms:
                    histograms[variable] = hist
                    histograms_per_rig[variable] = hist_per_rig
                else:
                    histograms[variable] += hist
                    histograms_per_rig[variable] += hist_per_rig
    return histograms, histograms_per_rig


def make_kde_template(kde, binning):
    kde_template = np.zeros(len(binning.edges) - 1)
    for index, (lower_edge, upper_edge) in enumerate(zip(binning.edges[1:-2], binning.edges[2:-1])):
        kde_template[index + 1] = kde.integrate_box_1d(lower_edge, upper_edge)
    return kde_template


def read_magnetic_field(path):
    header_format = "I" * 15
    with open(path, "rb") as file:
        data = file.read()
        header = struct.unpack_from(header_format, buffer=data, offset=0)
        nx, ny, nz = header[12:]
        header_size = 15 * 4
        format_x = "f" * nx
        size_x = nx * 4
        x = np.array(struct.unpack_from(format_x, buffer=data, offset=header_size), dtype=np.float32)
        format_y = "f" * ny
        size_y = ny * 4
        y = np.array(struct.unpack_from(format_y, buffer=data, offset=header_size + size_x), dtype=np.float32)
        format_z = "f" * nz
        size_z = nz * 4
        z = np.array(struct.unpack_from(format_z, buffer=data, offset=header_size + size_x + size_y), dtype=np.float32)
        coord_size = (nx + ny + nz) * 4
        size_b = nx * ny * nz * 4
        format_b = "f" * (nx * ny * nz)
        bx = np.array(struct.unpack_from(format_b, buffer=data, offset=header_size + coord_size), dtype=np.float32)
        by = np.array(struct.unpack_from(format_b, buffer=data, offset=header_size + coord_size + size_b), dtype=np.float32)
        bz = np.array(struct.unpack_from(format_b, buffer=data, offset=header_size + coord_size + size_b * 2), dtype=np.float32)
        bx, by, bz = map(lambda b: b.reshape((nx, ny, nz), order="F"), (bx, by, bz))
        size_skip = (nx + ny + nz) * 4
        format_bd = "f" * (nx * ny * nz * 2)
        size_bd = size_b * 2
        bdx = np.array(struct.unpack_from(format_bd, buffer=data, offset=header_size + coord_size + size_b * 3 + size_skip), dtype=np.float32)
        bdy = np.array(struct.unpack_from(format_bd, buffer=data, offset=header_size + coord_size + size_b * 3 + size_skip + size_bd), dtype=np.float32)
        bdz = np.array(struct.unpack_from(format_bd, buffer=data, offset=header_size + coord_size + size_b * 3 + size_skip + size_bd * 2), dtype=np.float32)

        return x, y, z, bx, by, bz

def load_magnetic_field():
    import lzma
    with lzma.open(os.path.join(os.environ["PHOTONFRAMEWORK"], "data", "magnetic_field.npz.xz"), "rb") as lzma_file:
        with np.load(lzma_file) as np_file:
            return np_file["x"], np_file["y"], np_file["z"], np_file["bx"], np_file["by"], np_file["bz"]

if __name__ == "__main__":
    x, y, z, bx, by, bz = read_magnetic_field(os.path.join(os.environ["AMSDataDir"], "v6.00", "MagneticFieldMapPermanent_NEW_FULL.bin"))
    np.savez("magnetic_field.npz", x=x, y=y, z=z, bx=bx, by=by, bz=bz)
    

def distribute_weights(fractions, batch_size):
    sizes = fractions * batch_size
    seats = np.floor(sizes)
    remainders = np.ma.MaskedArray(sizes - seats)
    while seats.sum() < batch_size:
        index = np.argmax(remainders)
        seats[index] += 1
        remainders[index] = np.ma.masked
    return seats


def merge_run_and_event_number(run_numbers, event_numbers):
    return run_numbers.astype(np.uint64) * 10_000_000 + event_numbers.astype(np.uint64)


def glob_re(pattern):
    base_dir_parts = []
    pattern_parts = []
    had_wildcard = False
    for dirname in pattern.split("/"):
        if "*" in dirname:
            had_wildcard = True
        if had_wildcard:
            pattern_parts.append(dirname)
        else:
            base_dir_parts.append(dirname)
    basedir = "/".join(base_dir_parts)
    patterns = []
    for pattern_str in pattern_parts:
        patterns.append(re.compile(pattern_str.replace("*", ".*")))

    def _walk_dir(dir, patterns):
        current_pattern = patterns[0]
        final_layer = len(patterns) == 1
        if os.path.isdir(dir):
            for entry in os.listdir(dir):
                if current_pattern.fullmatch(entry):
                    if final_layer:
                        yield os.path.join(dir, entry)
                    else:
                        yield from _walk_dir(os.path.join(dir, entry), patterns[1:])

    return sorted(list(_walk_dir(basedir, patterns)))
