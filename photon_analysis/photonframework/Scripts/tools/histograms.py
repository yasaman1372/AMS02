#!/usr/bin/env python3

from datetime import datetime, timezone

import numpy as np
import awkward as ak

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

from .binnings import Binning
from .statistics import lafferty_whyatt, row_mean_and_std, approximate_upper_poisson_error, approximate_lower_poisson_error
from .utilities import plot_steps, shaded_steps, set_plot_lim, plot_2d, transform_overflow_edges


def _np(array, dtype=None):
    if array is None:
        return None
    array = ak.to_numpy(array)
    if array.dtype == np.bool_:
        array = array.astype(np.uint8)
    if dtype is not None:
        array = array.astype(dtype)
    return array


def rebin_indices(old_edges, new_edges):
    #assert set(new_edges) <= set(old_edges) or (len(new_edges) == len(old_edges) and np.all(np.abs(new_edges[1:-1] - old_edges[1:-1]) < 1e-10))
    old_centers = (old_edges[1:] + old_edges[:-1]) / 2
    indices = np.digitize(old_centers, new_edges) - 1
    if old_edges[-1] == np.inf and indices[-1] == len(new_edges) - 1:
        indices[-1] -= 1
    return indices


def iterate_indices(indices):
    if len(indices) == 1:
        for old_index, new_index in enumerate(indices[0]):
            yield (old_index,), (new_index,)
    else:
        for old_index_tuple, new_index_tuple in iterate_indices(indices[:-1]):
            for old_index, new_index in enumerate(indices[-1]):
                yield (*old_index_tuple, old_index), (*new_index_tuple, new_index)


class Histogram:
    def __init__(self, *binnings, values=None, labels=None, dtype=np.float64):
        self.binnings = binnings
        self.bin_edges = [binning.edges for binning in self.binnings]
        self.dimensions = len(self.binnings)
        assert self.dimensions > 0
        if values is None:
            values = np.zeros([len(edges) - 1 for edges in self.bin_edges], dtype=dtype)
        assert len(values.shape) == self.dimensions
        assert all(shape == len(binning) for (shape, binning) in zip(values.shape, self.binnings))
        self.values = values
        self.labels = labels

    def rebin(self, *new_binnings, method="sum"):
        assert len(self.binnings) == len(new_binnings)
        indices = [rebin_indices(old_binning.edges, new_binning.edges) for (old_binning, new_binning) in zip(self.binnings, new_binnings)]
        shape = [len(binning.edges) - 1 for binning in new_binnings]
        new_values = np.zeros(shape, dtype=self.values.dtype)
        new_entries = np.zeros(shape, dtype=self.values.dtype)
        if self.dimensions == 1:
            for value, index in zip(self.values, indices[0]):
                new_values[index] += value
                new_entries[index] += 1
        elif self.dimensions == 2:
            for old_index_x, new_index_x in enumerate(indices[0]):
                for old_index_y, new_index_y in enumerate(indices[1]):
                    new_values[new_index_x,new_index_y] += self.values[old_index_x,old_index_y]
                    new_entries[new_index_x,new_index_y] += 1
        else:
            for old_index_tuple, new_index_tuple in iterate_indices(indices):
                new_values[new_index_tuple] += self.values[old_index_tuple]
                new_entries[new_index_tuple] += 1
        if method == "sum":
            pass
        elif method == "mean":
            new_values /= new_entries
        else:
            raise NotImplementedError
        return Histogram(*new_binnings, values=new_values, labels=self.labels)

    def serialize(self):
        return [v for binning in self.binnings for v in (binning.edges, binning.log)], self.values, self.labels

    @staticmethod
    def deserialize(t):
        binning_data, values, labels = t
        binnings = [Binning(edges, log) for (edges, log) in zip(binning_data[::2], binning_data[1::2])]
        return Histogram(*binnings, values=values, labels=labels)

    def add_to_file(self, file_dict, name):
        file_dict[f"{name}_dimensions"] = len(self.binnings)
        for index, binning in enumerate(self.binnings):
            binning.add_to_file(file_dict, f"{name}_binning_{index}")
        file_dict[f"{name}_values"] = self.values
        if self.labels is not None:
            file_dict[f"{name}_labels"] = True
            for index, label in enumerate(self.labels):
                file_dict[f"{name}_label_{index}"] = label
        else:
            file_dict[f"{name}_labels"] = False

    @staticmethod
    def from_file(file_dict, name):
        binning_log = [bool(file_dict[f"{name}_binning_{index}_log"]) for index in range(file_dict[f"{name}_dimensions"])]
        binnings = [Binning.from_file(file_dict, f"{name}_binning_{index}") for index in range(file_dict[f"{name}_dimensions"])]
        dimensions = len(binnings)
        values = file_dict[f"{name}_values"]
        labels = None
        if f"{name}_labels" in file_dict and file_dict[f"{name}_labels"]:
            labels = [file_dict[f"{name}_label_{index}"].item() for index in range(dimensions)]
        return Histogram(*binnings, values=values, labels=labels)


    @staticmethod
    def fill_direct(binnings, values, labels=None):
        hist = Histogram(*binnings, labels=labels)
        hist.fill(values)
        return hist

    def fill(self, *values):
        assert len(values) == self.dimensions
        if len(values[0]) == 0:
            return
        if self.dimensions == 1:
            hist, _ = np.histogram(_np(values[0]), bins=self.bin_edges[0])
            self.values += hist
        elif self.dimensions == 2:
            hist, *_ = np.histogram2d(*map(_np, values), bins=self.bin_edges)
            self.values += hist
        else:
            values = np.vstack([_np(value) for value in values]).transpose()
            hist, _ = np.histogramdd(values, bins=self.bin_edges)
            self.values += hist

    def get(self, *values):
        assert len(values) == self.dimensions
        coordinates = tuple([binning.get_indices(_np(value)) for value, binning in zip(values, self.binnings)])
        return self.values[coordinates]

    def get_errors(self):
        return np.sqrt(self.values)

    def add(self, other):
        self.values += other.values

    def __iadd__(self, other):
        self.add(other)
        return self

    def __add__(self, other):
        assert len(self.binnings) == len(other.binnings) and all(b1 == b2 for (b1, b2) in zip(self.binnings, other.binnings))
        return Histogram(*self.binnings, values=self.values + other.values, labels=self.labels)

    def __mul__(self, other):
        values = self.values * other
        squared_values = self.values * other**2
        return WeightedHistogram(*self.binnings, values=values, squared_values=squared_values, labels=self.labels)

    def __eq__(self, other):
        return len(self.binnings) == len(other.binnings) and all(ours == others for ours, others in zip(self.binnings, other.binnings)) and np.all(self.values == other.values)

    def project(self, min_index, max_index=None, axis=0):
        if max_index is None:
            max_index = min_index + 1
        binnings = [b for i, b in enumerate(self.binnings) if i != axis]
        labels = None
        if self.labels is not None:
            labels = [l for i, l in enumerate(self.labels) if i != axis]
        projection_binning = self.binnings[axis]
        min_value = projection_binning.edges[min_index]
        max_value = projection_binning.edges[max_index]
        value_slices = tuple([slice(None) for _ in range(axis)] + [slice(min_index, max_index)])
        values = self.values[value_slices].sum(axis=axis)
        return Histogram(*binnings, values=values, labels=labels)

    def project_axis(self, axis=0):
        return self.project(0, len(self.binnings[axis]), axis=axis)

    def project_by_value(self, min_value, max_value=None, axis=0, return_bin_edges = False):
        binning = self.binnings[axis]
        min_bin = binning.get_indices([min_value])[0]
        max_bin = binning.get_indices([max_value])[0] + 1 if max_value is not None else min_bin + 1
        if return_bin_edges:
            v_min = binning.edges[min_bin]
            v_max = binning.edges[max_bin]
            return self.project(min_bin, max_bin, axis=axis), v_min, v_max
        return self.project(min_bin, max_bin, axis=axis)

    def project_all(self, axis=0, include_overflow=False):
        binning = self.binnings[axis]
        bin_range = range(0, len(binning)) if include_overflow else range(1, len(binning) - 1)
        for bin_index in bin_range:
            v_min = binning.edges[bin_index]
            v_max = binning.edges[bin_index + 1]
            yield self.project(bin_index, axis=axis), v_min, v_max

    def to_uhi_proxy(self):
        axes = []
        labels = self.labels or ["xaxis", "yaxis", "zaxis"]
        for binning, label in zip(self.binnings, labels):
            axes.append(BinningUHIProxy(transform_overflow_edges(binning.edges), label))
        return WeightedHistogramUHIProxy(axes=axes, values=self.values, variances=self.values**2)




class WeightedHistogram:
    def __init__(self, *binnings, values=None, squared_values=None, labels=None, dtype=np.float64):
        self.binnings = binnings
        self.bin_edges = [binning.edges for binning in self.binnings]
        self.dimensions = len(self.binnings)
        assert self.dimensions > 0
        if values is None:
            shape = [len(edges) - 1 for edges in self.bin_edges]
            values = np.zeros(shape, dtype=dtype)
            squared_values = np.zeros(shape, dtype=dtype)
        assert values is not None and squared_values is not None
        assert len(values.shape) == self.dimensions and squared_values.shape == values.shape
        self.values = values
        self.squared_values = squared_values
        if labels is not None:
            assert len(labels) == self.dimensions
        self.labels = labels

    def fill_direct(binnings, *values, weights, labels=None):
        hist = WeightedHistogram(*binnings, labels=labels)
        hist.fill(*values, weights=weights)
        return hist

    def rebin(self, *new_binnings, method="sum"):
        assert len(self.binnings) == len(new_binnings)
        indices = [rebin_indices(old_binning.edges, new_binning.edges) for (old_binning, new_binning) in zip(self.binnings, new_binnings)]
        shape = [len(binning.edges) - 1 for binning in new_binnings]
        new_values = np.zeros(shape, dtype=self.values.dtype)
        new_squared_values = np.zeros(shape, dtype=self.values.dtype)
        new_scale = np.zeros(shape, dtype=self.values.dtype)
        if self.dimensions == 1:
            for value, squared_value, index in zip(self.values, self.squared_values, indices[0]):
                if method == "sum":
                    new_values[index] += value
                    new_squared_values[index] += squared_value
                elif method == "mean":
                    if squared_value > 0:
                        new_values[index] += value / squared_value
                        new_squared_values[index] = 1
                        new_scale[index] += 1 / squared_value
        elif self.dimensions == 2:
            for old_index_x, new_index_x in enumerate(indices[0]):
                for old_index_y, new_index_y in enumerate(indices[1]):
                    if method == "sum":
                        new_values[new_index_x,new_index_y] += self.values[old_index_x,old_index_y]
                        new_squared_values[new_index_x,new_index_y] += self.squared_values[old_index_x,old_index_y]
                    elif method == "mean":
                        if self.squared_values[old_index_x,old_index_y] > 0:
                            new_values[new_index_x,new_index_y] += self.values[old_index_x,old_index_y] / self.squared_values[old_index_x,old_index_y]
                            new_squared_values[new_index_x,new_index_y] = 1
                            new_scale[new_index_x,new_index_y] += 1 / self.squared_values[old_index_x,old_index_y]
        else:
            for old_index_tuple, new_index_tuple in iterate_indices(indices):
                if method == "sum":
                    new_values[new_index_tuple] += self.values[old_index_tuple]
                    new_squared_values[new_index_tuple] += self.squared_values[old_index_tuple]
                elif method == "mean":
                    if self.squared_values[old_index_tuple] > 0:
                        new_values[new_index_tuple] += self.values[old_index_tuple] / self.squared_values[old_index_tuple]
                        new_squared_values[new_index_tuple] = 1
                        new_scale[new_index_tuple] += 1 / self.squared_values[old_index_tuple]
        if method == "sum":
            pass
        elif method == "mean":
            new_values = new_values / new_scale
            new_squared_values = new_squared_values / new_scale
            new_values[new_scale == 0] = 0
            new_squared_values[new_scale == 0] = 0
        else:
            raise NotImplementedError
        return WeightedHistogram(*new_binnings, values=new_values, squared_values=new_squared_values, labels=self.labels)

    def serialize(self):
        return [v for binning in self.binnings for v in (binning.edges, binning.log)], self.values, self.squared_values, self.labels

    @staticmethod
    def deserialize(t):
        binning_data, values, squared_values, labels = t
        binnings = [Binning(edges, log) for (edges, log) in zip(binning_data[::2], binning_data[1::2])]
        return WeightedHistogram(*binnings, values=values, squared_values=squared_values, labels=labels)

    def add_to_file(self, file_dict, name):
        file_dict[f"{name}_dimensions"] = len(self.binnings)
        for index, binning in enumerate(self.binnings):
            binning.add_to_file(file_dict, f"{name}_binning_{index}")
        file_dict[f"{name}_values"] = self.values
        file_dict[f"{name}_squared_values"] = self.squared_values
        if self.labels is not None:
            file_dict[f"{name}_labels"] = True
            for index, label in enumerate(self.labels):
                file_dict[f"{name}_label_{index}"] = label
        else:
            file_dict[f"{name}_labels"] = False

    @staticmethod
    def from_file(file_dict, name):
        binning_log = [bool(file_dict[f"{name}_binning_{index}_log"]) for index in range(file_dict[f"{name}_dimensions"])]
        binnings = [Binning.from_file(file_dict, f"{name}_binning_{index}") for index in range(file_dict[f"{name}_dimensions"])]
        dimensions = len(binnings)
        values = file_dict[f"{name}_values"]
        squared_values = file_dict[f"{name}_squared_values"]
        labels = None
        if f"{name}_labels" in file_dict and file_dict[f"{name}_labels"]:
            labels = [file_dict[f"{name}_label_{index}"].item() for index in range(dimensions)]
        return WeightedHistogram(*binnings, values=values, squared_values=squared_values, labels=labels)

    @staticmethod
    def from_thist(thist, transform=None):
        if transform is None:
            transform = [lambda x: x for _ in thist.axes]
        binnings = [Binning(t(axis.edges())) for axis, t in zip(thist.axes, transform)]
        values = np.pad(thist.values(), 1)
        squared_values = np.pad(thist.errors()**2, 1)
        return WeightedHistogram(*binnings, values=values, squared_values=squared_values)

    def fill(self, *values, weights):
        assert len(values) == self.dimensions
        if self.dimensions == 1:
            v = _np(values[0])
            weighted_hist, _ = np.histogram(v, bins=self.bin_edges[0], weights=_np(weights, dtype=self.values.dtype))
            square_weighted_hist, _ = np.histogram(v, bins=self.bin_edges[0], weights=_np(weights**2, dtype=self.values.dtype))
            self.values += weighted_hist
            self.squared_values += square_weighted_hist
        elif self.dimensions == 2:
            values = [_np(v) for v in values]
            weighted_hist, *_ = np.histogram2d(*values, bins=self.bin_edges, weights=_np(weights, dtype=self.values.dtype))
            square_weighted_hist, *_ = np.histogram2d(*values, bins=self.bin_edges, weights=_np(weights**2, dtype=self.values.dtype))
            self.values += weighted_hist
            self.squared_values += square_weighted_hist
        else:
            values = np.vstack([_np(v) for v in values]).transpose()
            weighted_hist, _ = np.histogramdd(values, bins=self.bin_edges, weights=_np(weights, dtype=self.values.dtype))
            square_weighted_hist, _ = np.histogramdd(values, bins=self.bin_edges, weights=_np(weights**2, dtype=self.values.dtype))
            self.values += weighted_hist
            self.squared_values += square_weighted_hist

    def get_errors(self):
        return np.sqrt(self.squared_values)

    def get_approximate_poisson_errors(self):
        return approximate_lower_poisson_error(self.values), approximate_upper_poisson_error(self.values)

    def projection_mean_and_std(self, axis=1):
        assert self.dimensions == 2
        return row_mean_and_std(self.binnings[axis].bin_centers[1:-1], self.values[1:-1,1:-1], axis=axis)

    def add(self, other):
        self.values += other.values
        self.squared_values += other.squared_values

    def get(self, *values):
        assert len(values) == self.dimensions
        coordinates = tuple([binning.get_indices(_np(value)) for value, binning in zip(values, self.binnings)])
        return self.values[coordinates]

    def __iadd__(self, other):
        self.add(other)
        return self

    def __add__(self, other):
        assert len(self.binnings) == len(other.binnings) and all(b1 == b2 for (b1, b2) in zip(self.binnings, other.binnings))
        return WeightedHistogram(*self.binnings, values=self.values + other.values, squared_values=self.squared_values + other.squared_values, labels=self.labels)

    def __mul__(self, other):
        values = self.values * other
        squared_values = self.squared_values * other**2
        return WeightedHistogram(*self.binnings, values=values, squared_values=squared_values, labels=self.labels)

    def __truediv__(self, other):
        return self * (1 / other)

    def project(self, min_index, max_index=None, axis=0):
        if max_index is None:
            max_index = min_index + 1
        binnings = [b for i, b in enumerate(self.binnings) if i != axis]
        labels = None
        if self.labels is not None:
            labels = [l for i, l in enumerate(self.labels) if i != axis]
        projection_binning = self.binnings[axis]
        min_value = projection_binning.edges[min_index]
        max_value = projection_binning.edges[max_index]
        value_slices = tuple([slice(None) for _ in range(axis)] + [slice(min_index, max_index)])
        values = self.values[value_slices].sum(axis=axis)
        squared_values = self.squared_values[value_slices].sum(axis=axis)
        return WeightedHistogram(*binnings, values=values, squared_values=squared_values, labels=labels)

    def project_axis(self, axis=0):
        return self.project(0, len(self.binnings[axis]), axis=axis)

    def project_by_value(self, min_value, max_value=None, axis=0, return_bin_edges = False):
        binning = self.binnings[axis]
        min_bin = binning.get_indices([min_value])[0]
        max_bin = binning.get_indices([max_value])[0] if max_value is not None else min_bin + 1
        if return_bin_edges:
            v_min = binning.edges[min_bin]
            v_max = binning.edges[max_bin]
            return self.project(min_bin, max_bin, axis=axis), v_min, v_max
        return self.project(min_bin, max_bin, axis=axis)

    def project_all(self, axis=0, include_overflow=False):
        binning = self.binnings[axis]
        bin_range = range(0, len(binning)) if include_overflow else range(1, len(binning) - 1)
        for bin_index in bin_range:
            v_min = binning.edges[bin_index]
            v_max = binning.edges[bin_index + 1]
            yield self.project(bin_index, axis=axis), v_min, v_max

    # for uproot writing (UHI compatibility)
    def to_uhi_proxy(self):
        axes = []
        labels = self.labels or ["xaxis", "yaxis", "zaxis"]
        for binning, label in zip(self.binnings, labels):
            axes.append(BinningUHIProxy(transform_overflow_edges(binning.edges), label))
        return WeightedHistogramUHIProxy(axes=axes, values=self.values, variances=self.squared_values)


class BinningUHIProxy:
    def __init__(self, edges, label):
        self.edges = edges
        self.name = label

    def __len__(self):
        return len(self.edges) - 1


class WeightedHistogramUHIProxy:
    def __init__(self, axes, values, variances):
        self.axes = axes
        self._values = values
        self._variances = variances
        self.kind = "COUNT"
        import boost_histogram
        self.storage_type = boost_histogram.storage.Weight

    def values(self, flow=None):
        if flow:
            raise TypeError
        return self._values

    def variances(self, flow=None):
        if flow:
            raise TypeError
        return self._variances

    def counts(self):
        # dummy?
        pass


def plot_histogram_1d(plot, histogram, style="mc", color=None, label=None, label_y=None, scale=None, gamma=None, log=False, shade_errors=False, show_overflow=False, adjust_limits=None, adjust_limits_x=None, adjust_limits_y=None, flip_axes=False, override_limits=False, use_approximate_poisson_errors=False, draw_zeros=True, draw_density=False, marker=".", datetime_axis=None, **kwargs):
    assert histogram.dimensions == 1
    values_y = histogram.values
    if use_approximate_poisson_errors:
        errors_y_low, errors_y_high = histogram.get_approximate_poisson_errors()
    else:
        errors_y = histogram.get_errors()
        errors_y_low = errors_y
        errors_y_high = errors_y
    if not show_overflow:
        values_y = values_y[1:-1]
        errors_y_low = errors_y_low[1:-1]
        errors_y_high = errors_y_high[1:-1]
    if scale is not None:
        values_y = values_y * scale
        errors_y_low = errors_y_low * scale
        errors_y_high = errors_y_high * scale

    binning = histogram.binnings[0]
    bin_edges = transform_overflow_edges(binning.edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 if not binning.log else (bin_edges[:-1] * bin_edges[1:])**0.5
    if gamma is not None:
        bin_centers = lafferty_whyatt(binning.edges, gamma)
    if not show_overflow:
        bin_edges = bin_edges[1:-1]
        bin_centers = bin_centers[1:-1]
    if datetime_axis or binning.is_datetime:
        bin_centers = np.array(bin_centers, dtype="datetime64[s]")
        bin_edges = np.array(bin_edges, dtype="datetime64[s]")

    if draw_density:
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        values_y = values_y / bin_widths
        errors_y_low = errors_y_low / bin_widths
        errors_y_high = errors_y_high / bin_widths

    if binning.log:
        if flip_axes:
            plot.set_yscale("log")
        else:
            plot.set_xscale("log")
    if log:
        if flip_axes:
            plot.set_xscale("log")
        else:
            plot.set_yscale("log")
    if histogram.labels is not None:
        x_label = histogram.labels[0]
        if label_y is not None:
            y_label = label_y
        else:
            y_label = "Events"
            if draw_density:
                y_label = "Event Density"
        if flip_axes:
            plot.set_xlabel(y_label)
            plot.set_ylabel(x_label)
        else:
            plot.set_xlabel(x_label)
            plot.set_ylabel(y_label)

    result = []
    if style == "iss":
        if not draw_zeros:
            nonzero = values_y > 0
            bin_centers = bin_centers[nonzero]
            values_y = values_y[nonzero]
            errors_y_low = errors_y_low[nonzero]
            errors_y_high = errors_y_high[nonzero]
        if flip_axes:
            result = plot.errorbar(y=bin_centers, x=values_y, xerr=(errors_y_low, errors_y_high), marker=marker, linestyle="", color=color, label=label, **kwargs)
        else:
            result = plot.errorbar(x=bin_centers, y=values_y, yerr=(errors_y_low, errors_y_high), marker=marker, linestyle="", color=color, label=label, **kwargs)
    elif style == "curve":
        if not draw_zeros:
            nonzero = values_y > 0
            bin_centers = bin_centers[nonzero]
            values_y = values_y[nonzero]
            #errors_y_low = errors_y_low[nonzero]
            #errors_y_high = errors_y_high[nonzero]
        if flip_axes:
            result = plot.plot( values_y, bin_centers, marker=marker, linestyle="-", color=color, label=label, **kwargs)
        else:
            result = plot.plot(bin_centers, values_y, marker=marker, linestyle="-", color=color, label=label, **kwargs)
    
    elif style == "mc":
        result = plot_steps(plot, bin_edges, values_y, color=color, label=label, **kwargs)
        if flip_axes:
            for line in result:
                xdata, ydata = line.get_data()
                line.set_data(ydata, xdata)
                line.set_drawstyle("steps-pre")
        if shade_errors:
            shaded_steps(plot, bin_edges, values_y - errors_y_low, values_y + errors_y_high, color=color, alpha=0.5)
    else:
        raise ValueError(f"Unknown draw style {style}")
    if adjust_limits is not None:
        if adjust_limits_x is None:
            adjust_limits_x = adjust_limits
        if adjust_limits_y is None:
            adjust_limits_y = adjust_limits
    else:
        if adjust_limits_x is None:
            adjust_limits_x = False
        if adjust_limits_y is None:
            adjust_limits_y = True
    if adjust_limits:
        if adjust_limits_y:
            axis = "x" if flip_axes else "y"
            set_plot_lim(plot, values_y + errors_y_high, log=log, axis=axis, override=override_limits)
            set_plot_lim(plot, values_y - errors_y_low, log=log, axis=axis, override=override_limits)
        if adjust_limits_x:
            axis = "y" if flip_axes else "x"
            set_plot_lim(plot, bin_edges[0], log=log, axis=axis, override=override_limits)
            set_plot_lim(plot, bin_edges[-1], log=log, axis=axis, override=override_limits)

    return result


def plot_histogram_2d(plot, histogram, scale=None, transpose=False, show_overflow=True, show_overflow_x=None, show_overflow_y=None, draw_density=False, label=None, **kwargs):
    assert histogram.dimensions == 2
    values = histogram.values
    if show_overflow_x is None:
        show_overflow_x = show_overflow
    if show_overflow_y is None:
        show_overflow_y = show_overflow
    bin_widths_x = histogram.binnings[0].bin_widths
    bin_widths_y = histogram.binnings[1].bin_widths
    if not show_overflow_x:
        values = values[1:-1,:]
        bin_widths_x = bin_widths_x[1:-1]
    if not show_overflow_y:
        values = values[:,1:-1]
        bin_widths_y = bin_widths_y[1:-1]
    if draw_density:
        values = values / (bin_widths_x[:,None] * bin_widths_y[None,:])
        if label is None:
            label = "Event Density"
    if transpose:
        values = values.transpose()
    if scale is not None:
        if isinstance(scale, np.ndarray) and not show_overflow_x:
            scale = scale[1:-1]
        if label is None:
            label = "Normalized Events"
    if label is None:
        label = "Events"
    if histogram.labels is not None:
        label_x, label_y = histogram.labels
        if transpose:
            label_x, label_y = label_y, label_x
        plot.set_xlabel(label_x)
        plot.set_ylabel(label_y)
    index_x, index_y = 0, 1
    if transpose:
        index_x, index_y = index_y, index_x
    if histogram.binnings[index_x].log:
        plot.set_xscale("log")
    if histogram.binnings[index_y].log:
        plot.set_yscale("log")
    edges_x = histogram.binnings[index_x].edges
    edges_y = histogram.binnings[index_y].edges
    if show_overflow_x:
        edges_x = transform_overflow_edges(edges_x)
    else:
        edges_x = edges_x[1:-1]
    if show_overflow_y:
        edges_y = transform_overflow_edges(edges_y)
    else:
        edges_y = edges_y[1:-1]
        # klappt nicht immer muss man noch schauen
    cbar = plot_2d(plot, values, edges_x, edges_y, scale=scale, colorbar_label=label, **kwargs)
    return cbar

def load_histogram(file, name):
    keys = list(file)
    assert f"{name}_dimensions" in keys
    if f"{name}_squared_values" in keys:
        return WeightedHistogram.from_file(file, name)
    return Histogram.from_file(file, name)

def list_histograms(file):
    return [key[:-len("_dimensions")] for key in file if key.endswith("_dimensions")]

def load_histograms(file, keys=None):
    if keys is None:
        keys = list_histograms(file)
    return {name: load_histogram(file, name) for name in keys}

def load_histograms_from_files(filenames, keys=None):
    histograms = None
    for filename in filenames:
        with np.load(filename) as np_file:
            hists = load_histograms(np_file, keys=keys)
        if histograms:
            for key in histograms:
                histograms[key].add(hists[key])
        else:
            histograms = hists
    return histograms


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")

    args = parser.parse_args()

    with np.load(args.filename) as hist_file:
        print("\n".join(list_histograms(hist_file)))
