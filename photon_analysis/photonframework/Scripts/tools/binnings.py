#!/usr/bin/env python3

from datetime import datetime, timedelta

import numpy as np
import awkward as ak
import healpy

from tools.constants import MC_PARTICLE_IDS, MC_PARTICLE_CHARGE_ARRAY
from tools.utilities import parse_rigidity_value

def add_overflow(edges):
    if edges[0] == -np.inf and edges[-1] == np.inf:
        return edges
    return np.concatenate(([-np.inf], edges, [np.inf]))


class Binning:
    def __init__(self, edges, log=False, is_datetime=False):
        self.edges = add_overflow(edges)
        self.bin_centers = (self.edges[1:] + self.edges[:-1]) / 2
        self.bin_widths = (self.edges[1:] - self.edges[:-1])
        self.log = log
        self.is_datetime = is_datetime

    def add_to_file(self, file_dict, name):
        file_dict[f"{name}_edges"] = self.edges
        file_dict[f"{name}_log"] = self.log
        file_dict[f"{name}_is_datetime"] = self.is_datetime

    @staticmethod
    def from_file(file_dict, name):
        return Binning(edges=file_dict[f"{name}_edges"], log=file_dict[f"{name}_log"], is_datetime=file_dict.get(f"{name}_is_datetime", False))

    def __eq__(self, other):
        return len(self.edges) == len(other.edges) and np.all(self.edges == other.edges) and self.log == other.log

    def __ne__(self, other):
        return len(self.edges) != len(other.edges) or np.any(self.edges != other.edges) or self.log != other.log

    def __len__(self):
        return len(self.edges) - 1

    def get_indices(self, values, with_overflow=True):
        return np.clip(np.digitize(ak.to_numpy(values), self.edges) - 1, 0 if with_overflow else 1, len(self.edges) - (2 if with_overflow else 3))

    def reduce_range(self, min_value=-np.inf, max_value=np.inf):
        new_edges = self.edges[(self.edges >= min_value) & (self.edges <= max_value)]
        return Binning(new_edges, log=self.log)


def get_rebin_factor(binning, min_factor=2):
    tried = set()
    for factor in range(2, len(binning) // 2):
        if any(factor % old_factor == 0 for old_factor in tried):
            continue
        tried.add(factor)
        if (len(binning) - 2) % factor == 0 and factor >= min_factor:
            return factor
    return 1


def reduce_bins(binning, factor, accept_unequal=False):
    bins = len(binning) - 2
    if not accept_unequal:
        assert bins % factor == 0
    new_edges = binning.edges[1:-1:factor]
    if not accept_unequal:
        assert new_edges[-1] == binning.edges[-2]
    else:
        if new_edges[-1] != binning.edges[-2]:
            new_edges = np.concatenate((new_edges, [binning.edges[-2]]))
    return Binning(new_edges, log=binning.log)


def increase_bin_number(binning, factor):
    assert int(factor) == factor and factor > 1
    edges = binning.edges[1:-1]
    def _make_edges(min, max, n):
        if binning.log:
            edges = np.logspace(np.log10(min), np.log10(max), n, endpoint=False)
        else:
            edges = np.linspace(min, max, n, endpoint=False)
        edges[0] = min
        return edges
    new_edges = [_make_edges(low, high, factor) for (low, high) in zip(edges[:-1], edges[1:])] + [np.array([edges[-1]])]
    new_edges = np.concatenate(new_edges)
    return Binning(new_edges, log=binning.log)
    

def combine_binnings(binnings):
    log = any(binning.log for binning in binnings)
    edges = np.array(sorted({edge for binning in binnings for edge in binning.edges if np.isfinite(edge)}))
    distanced = np.concatenate(([True], (edges[1:] - edges[:-1]) > 1e-10))
    return Binning(edges[distanced], log=log)


def make_lin_binning(start, stop, nbins):
    return Binning(np.linspace(start, stop, nbins + 1))


def make_log_binning(start, stop, nbins):
    return Binning(np.logspace(np.log10(start), np.log10(stop), nbins + 1), log=True)


def make_int_binning(n, nmin=0):
    return Binning(np.arange(nmin, n + 1) - 0.5)


def make_bool_binning():
    return make_int_binning(2)


def make_lin_binning_with_known_edge(start, stop, nbins_min, known_edge):
    assert stop >= known_edge >= start
    bin_width_max = (stop - start) / nbins_min
    bins_before_edge = int(np.ceil((known_edge - start) / bin_width_max))
    bin_width = (known_edge - start) / bins_before_edge
    bins_after_edge = int(np.ceil((stop - known_edge) / bin_width))
    nbins = bins_before_edge + bins_after_edge
    new_stop = start + nbins * bin_width
    assert new_stop >= stop
    assert nbins >= nbins_min
    return make_lin_binning(start, new_stop, nbins)


def make_log_binning_with_known_edge(start, stop, nbins_min, known_edge):
    binning = make_lin_binning_with_known_edge(np.log10(start), np.log10(stop), nbins_min, np.log10(known_edge))
    edges = 10**binning.edges
    return Binning(edges, log=True)


def make_binning_from_config(config, name):
    return Binning([parse_rigidity_value(edge) for edge in config["analysis"]["binnings"][name]], log=True)


def make_energy_binning(rig_min, rig_max, log_rig_resolution):
    log_rig_min = np.log10(rig_min)
    log_rig_max = np.log10(rig_max)
    nbins = int(np.ceil((log_rig_max - log_rig_min) / log_rig_resolution))
    return Binning(np.logspace(log_rig_min, log_rig_max, nbins + 1), log=True)

def make_energy_binning_from_config(config):
    rig_min = config["analysis"]["energy"]["min"]
    rig_max = config["analysis"]["energy"]["max"]
    log_rig_resolution = config["analysis"]["energy"].get("log_resolution", 0.033)
    return make_energy_binning(rig_min, rig_max, log_rig_resolution)


def make_mc_edge_binning(config):
    return Binning(np.array(sorted({p / np.abs(dataset["charge"]) for dataset in config["datasets"].values() if dataset["acqt_datatype"] == "MC" for acqt in dataset["acqt_datasets"] for p in (acqt["pmin"], acqt["pmax"])})), log=True)

def make_mc_dataset_edge_binning(config, dataset_name):
    dataset = config["datasets"][dataset_name]
    charge = np.abs(dataset["charge"])
    edges = sorted(set([d[key] / charge for d in dataset["acqt_datasets"] for key in ["pmin", "pmax"]]))
    return Binning(np.array(edges), log=True)

def make_mc_dataset_energy_binning(config, dataset_name):
    dataset_config = config["datasets"][dataset_name]
    charge = dataset_config["charge"]
    log_rig_resolution = config["analysis"]["energy"].get("log_resolution", 0.033)
    pmin = min(acqt_dataset["pmin"] for acqt_dataset in dataset_config["acqt_datasets"])
    pmax = max(acqt_dataset["pmax"] for acqt_dataset in dataset_config["acqt_datasets"])
    rmin = np.abs(pmin / charge)
    rmax = np.abs(pmax / charge)
    return make_energy_binning(rmin, rmax, log_rig_resolution)

def make_signed_binning(positive_binning):
    negative_edges = -positive_binning.edges[::-1]
    all_edges = np.concatenate((negative_edges[1:-1], positive_binning.edges[1:-1]))
    return Binning(all_edges)


def make_mass_binning(config):
    mass = config["analysis"]["charge"] * 2
    return make_lin_binning(0, 2 * mass, 50 * mass)


def make_year_binning_from_config(config):
    min_year = datetime.strptime(config["analysis"].get("time", {}).get("min", "2011-01-01"), "%Y-%m-%d").year
    max_year = datetime.strptime(config["analysis"].get("time", {}).get("max", "2022-01-01"), "%Y-%m-%d").year
    return make_year_binning(min_year, max_year)

def make_year_binning(min_year, max_year):
    years = np.arange(min_year, max_year + 1)
    timestamps = np.array([datetime(year, 1, 1).timestamp() for year in years])
    return Binning(timestamps, is_datetime=True)

def make_month_binning_from_config(config):
    min_date = datetime.strptime(config["analysis"].get("time", {}).get("min", "2011-01-01"), "%Y-%m-%d")
    max_date = datetime.strptime(config["analysis"].get("time", {}).get("max", "2022-01-01"), "%Y-%m-%d")
    return make_month_binning(min_date, max_date)

def make_month_binning(min_date, max_date):
    current = datetime(year=min_date.year, month=min_date.month, day=1)
    timestamps = [current.timestamp()]
    while current < max_date:
        current = datetime(year=current.year + current.month // 12, month=(current.month % 12) + 1, day=1)
        timestamps.append(current.timestamp())
    return Binning(np.array(timestamps), is_datetime=True)

def make_week_binning_from_config(config):
    min_date = datetime.strptime(config["analysis"].get("time", {}).get("min", "2011-01-01"), "%Y-%m-%d")
    max_date = datetime.strptime(config["analysis"].get("time", {}).get("max", "2022-01-01"), "%Y-%m-%d")
    return make_week_binning(min_date, max_date)

def make_week_binning(min_date, max_date):
    current = datetime(year=min_date.year, month=min_date.month, day=min_date.day, tzinfo=min_date.tzinfo) - timedelta(days=min_date.weekday())
    timestamps = [current.timestamp()]
    while current < max_date:
        current = current + timedelta(days=7)
        timestamps.append(current.timestamp())
    return Binning(np.array(timestamps), is_datetime=True)

def make_day_binning_from_config(config):
    min_date = datetime.strptime(config["analysis"].get("time", {}).get("min", "2011-01-01"), "%Y-%m-%d")
    max_date = datetime.strptime(config["analysis"].get("time", {}).get("max", "2022-01-01"), "%Y-%m-%d")
    return make_day_binning(min_date, max_date)

def make_day_binning(min_date, max_date):
    current = datetime(year=min_date.year, month=min_date.month, day=min_date.day, tzinfo=min_date.tzinfo) - timedelta(days=min_date.weekday())
    timestamps = [current.timestamp()]
    while current < max_date:
        current = current + timedelta(days=1)
        timestamps.append(current.timestamp())
    return Binning(np.array(timestamps), is_datetime=True)

def make_flux_energy_binning():
    flux_energies = np.array([0.04385869406930723, 0.05700124121455603, 
    0.07408203935268977, 0.09628121138618302, 0.12513251183404786, 
    0.1626292948796985, 0.2113622364437464, 0.27469832558487545, 
    0.357013491855332, 0.4639949409788287, 0.6030340874097455, 
    0.7837372317270523, 1.0185892658797924, 1.3238162620898277, 
    1.720506542212375, 2.2360676829294905, 2.9061201221660187, 
    3.776958197165874, 5.596095315918129, 9.452405326071256, 
    15.966126630150379, 26.968500691236194, 45.552690792253664, 
    76.94338154619712, 129.965625757731, 219.52588434727338, 
    370.8027689435848, 626.3256556968166, 1057.931223387742])
    #flux_energy_bin_edges = list(flux_energies)
    #flux_energy_bin_edges.append(10**(np.log10(flux_energies[-1])+(np.log10(flux_energies[-1])-np.log10(flux_energies[-2]))))
    flux_energy_binning = Binning(np.array(flux_energies), log=True)
    return flux_energy_binning


def make_healpy_binning(nside=128, cache={}):
    if nside in cache:
        return cache[nside]
    binning = make_int_binning(healpy.nside2npix(nside))
    cache[nside] = binning
    return binning


def make_hierarchical_binning(min_value, max_value, positive=False, density=10):
    factor = max_value / 10**int(np.log10(max_value))
    return combine_binnings([
        (make_lin_binning(0, factor * 10**n, density) if positive else make_lin_binning(-factor * 10**n, factor * 10**n, 2 * density))
        for n in range(int(np.log10(min_value)) + 1, int(np.log10(max_value)) + 1)
    ])


class Binnings:
    def __init__(self, named_binnings, variable_binnings):
        self.named_binnings = named_binnings
        self.variable_binnings = variable_binnings

    def __getitem__(self, key):
        if key in self.variable_binnings:
            entry = self.variable_binnings[key]
            if isinstance(entry, Binning):
                return entry
            key = entry
        return self.named_binnings[key]


    def register_binning(self, name, binning):
        assert isinstance(binning, Binning)
        self.named_binnings[name] = binning

    def register_variable(self, variable_name, binning_key):
        if not isinstance(binning_key, Binning) and not binning_key in self.named_binnings:
            raise ValueError(f"Unknown binning key: {binning_key!r}, cannot register.")
        self.variable_binnings[variable_name] = binning_key

    
    @staticmethod
    def from_config(config):
        binnings = Binnings({}, {})
        binnings.initialize(config)
        return binnings

    def initialize(self, config):
        self.register_base_binnings(config)
        self.register_base_variables()

    def register_base_binnings(self, config):
        self.register_binning("energy", make_energy_binning_from_config(config))
        self.register_binning("rigidity", make_signed_binning(self["energy"]))
        self.register_binning("time", make_year_binning_from_config(config))
        self.register_binning("year", make_year_binning_from_config(config))
        self.register_binning("month", make_month_binning_from_config(config))
        self.register_binning("week", make_week_binning_from_config(config))
        self.register_binning("day", make_day_binning_from_config(config))
        self.register_binning(bool, make_bool_binning())
        self.register_binning("longitude", make_lin_binning(-180, 180, 360))
        self.register_binning("latitude", make_lin_binning(-90, 90, 180))
        self.register_binning("charge", make_lin_binning(0, 2.5, 250))
        self.register_binning("trigger_flags", make_int_binning(128))
        self.register_binning("bdt", make_lin_binning(-1, 1, 100))
        self.register_binning("logit", make_lin_binning(-10, 10, 100))
        self.register_binning("direction", make_lin_binning(-1, 1, 100))
        self.register_binning("dxydz", make_lin_binning(-0.5, 0.5, 100))
        self.register_binning("theta", make_lin_binning(-np.pi / 2, np.pi / 2, 180))
        self.register_binning("phi", make_lin_binning(0, 2 * np.pi, 180))
        self.register_binning("coordinate", make_lin_binning(-200, 200, 200))
        self.register_binning("radius", make_lin_binning(0, 200, 100))
        self.register_binning("mc_particle_id", make_int_binning(max(MC_PARTICLE_IDS.values()) + 1))
        self.register_binning("beta", make_lin_binning(0, 2, 200))
        self.register_binning("flux_energy", make_flux_energy_binning())



    def register_base_variables(self):
        self.register_variable("Time", "time")
        self.register_variable("UTCTime", "time")
        self.register_variable("RunNumber", "time")
        self.register_variable("EventNumber", make_lin_binning(0, 1e8, 100))
        self.register_variable("McMomentum", "energy")
        self.register_variable("McPrimaryX", "direction")
        self.register_variable("McPrimaryY", "direction")
        self.register_variable("McPrimaryZ", "direction")
        self.register_variable("McPrimaryFinalMomentum", "energy")
        self.register_variable("McPrimaryFinalX", "direction")
        self.register_variable("McPrimaryFinalY", "direction")
        self.register_variable("McPrimaryFinalZ", "direction")
        self.register_variable("TrdNActiveLayers", make_int_binning(21))
        self.register_variable("NTrdHits", make_int_binning(150))
        self.register_variable("AccNClusters", make_int_binning(11))
        self.register_variable("AccNClustersTrigger", make_int_binning(11))
        self.register_variable("TrkNTracks", make_int_binning(10))
        self.register_variable("NEcalShowers", make_int_binning(4))
        self.register_variable("EcalEnergy", "energy")
        self.register_variable("EcalBdt", "bdt")
        self.register_variable("EcalIntegralLikelihood", make_lin_binning(0, 20, 100))
        self.register_variable("EcalReweightedLikelihood", make_lin_binning(0, 10, 100))
        self.register_variable("NEcal2DShowers", make_int_binning(5))
