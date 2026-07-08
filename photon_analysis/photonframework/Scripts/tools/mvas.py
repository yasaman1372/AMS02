#!/usr/bin/env python3

import json
import os

import awkward as ak
import numpy as np
from scipy.interpolate import interp1d

import xgboost as xgb

from tools.utilities import rec_to_float, filter_branches


def generate_mva_parameter_combinations(mva_config, config):
    mva_params = mva_config.get("mva_parameters", config["analysis"]["mva_parameters"])
    if isinstance(mva_params, dict):
        for depth in mva_params["depth"]:
            for ntrees in mva_params["ntrees"]:
                for eta in mva_params["eta"]:
                    yield (depth, ntrees, eta)
    elif isinstance(mva_params, list):
        for param_dict in mva_params:
            yield (param_dict["depth"], param_dict["ntrees"], param_dict["eta"])
    else:
        raise ValueError(f"Cannot parse mva parameters from {mva_params!r}")
 


class SingleMVA:
    def __init__(self, variables, bdt=None, load_bdt=None, score_to_efficiency=None):
        self.variables = variables
        self.bdt = bdt
        if bdt is None:
            if load_bdt is None:
                raise ValueError("Either bdt or load_bdt is required!")
            self.load_bdt = load_bdt
        self.score_to_efficiency = score_to_efficiency

    def predict(self, chunk, verbose=False):
        if self.bdt is None:
            self.bdt = self.load_bdt()
        d = filter_branches(chunk, self.variables)
        for column_name in d.dtype.names:
            if np.any(np.isnan(d[column_name])):
                print("NaN in:", column_name)
            if np.any(np.invert(np.isfinite(d[column_name]))):
                print("inf in:", column_name)
        input_data = rec_to_float(d)
        result = self.bdt.predict(xgb.DMatrix(input_data, feature_names=self.variables))
        if verbose:
            for event, input_values, result_value in zip(chunk, input_data, result):
                input_value_str = " ".join(f"{v:.6f}" for v in input_values)
                print(f"{event.RunNumber} {event.EventNumber}: {result_value:.4f} <= {input_value_str}")
        return result

    def predict_as_efficiency(self, chunk):
        return self.score_to_efficiency(self.predict(chunk))

    @staticmethod
    def load(filenames):
        filename = None
        for f in filenames:
            if os.path.exists(f):
                filename = f
                break
        if filename is None:
            return None
        with open(filename, "r") as json_file:
            bdt_config = json.load(json_file)
            bdt_path = bdt_config["path"]
            bdt_path_candidates = [
                bdt_config["path"],
                os.path.join(os.path.dirname(filename), os.path.basename(bdt_config["path"])),
            ]
            def _load_bdt():
                bdt = xgb.Booster()
                loaded = False
                for bdt_path in bdt_path_candidates:
                    if os.path.exists(bdt_path):
                        bdt.load_model(bdt_path)
                        loaded = True
                        break
                if not loaded:
                    raise FileNotFoundError(bdt_path_candidates)
                return bdt
            score_to_efficiency = None
            efficiency_config = bdt_config.get("efficiency", bdt_config.get("efficiency_rejection", None))
            if "efficiency" in bdt_config:
                efficiency_config = bdt_config["efficiency"]
                if "signal_efficiency" in efficiency_config and "cut_values" in efficiency_config:
                    efficiency = np.array(efficiency_config["signal_efficiency"])
                    cut_values = np.array(efficiency_config["cut_values"])
                    score_to_efficiency = interp1d(cut_values, efficiency)
            elif "efficiency_rejection" in bdt_config:
                efficiency_config = bdt_config["efficiency_rejection"]
                if "efficiency" in efficiency_config and "cut_values" in efficiency_config:
                    efficiency = np.array(efficiency_config["efficiency"])
                    cut_values = np.array(efficiency_config["cut_values"])
                    score_to_efficiency = interp1d(cut_values, efficiency)
            return SingleMVA(bdt_config["variables"], load_bdt=_load_bdt, score_to_efficiency=score_to_efficiency)


class MVA:
    def __init__(self, energy_estimator, rigidity_binning, mvas):
        self.energy_estimator = energy_estimator
        self.rigidity_binning = rigidity_binning
        self.mvas = mvas

    def predict(self, chunk, verbose=False):
        bins = self.rigidity_binning.get_indices(np.abs(chunk[self.energy_estimator]), with_overflow=False)
        bin_range = range(1, len(self.rigidity_binning) - 1)
        result = np.zeros(len(chunk), dtype=np.float32)
        for rig_bin in bin_range:
            mask = bins == rig_bin
            if not np.any(mask):
                continue
            result[mask] = self.mvas[rig_bin].predict(chunk[mask], verbose=verbose)
        return result

    def predict_as_efficiency(self, chunk):
        bins = self.rigidity_binning.get_indices(chunk[self.energy_estimator], with_overflow=False)
        bin_range = range(1, len(self.rigidity_binning) - 1)
        return np.sum([self.mvas[bin].predict_as_efficiency(chunk) * (bins == bin) for bin in bin_range], axis=0)

    @staticmethod
    def load_all(mva_config, mva_name, energy_estimator, config, workdir, binnings):
        directory = os.path.join(workdir, "mvas", mva_name, "train", "results")
        rigidity_binning = binnings.special_binnings[mva_config.get("rigidity_binning", "rigidity_search")]
        variable_name = mva_config["creates"]
        mva_variables = mva_config["variables"]
        for depth, ntrees, eta in generate_mva_parameter_combinations(mva_config, config):
            param_name = f"{depth}x{ntrees}x{eta:.2f}"
            mva_loader = MVALoader(directory=directory, mva_name=mva_name, param_name=param_name, energy_estimator=energy_estimator, rigidity_binning=rigidity_binning)
            yield f"{variable_name}{param_name}", mva_loader.load, mva_variables
            for rig_bin in range(1, len(rigidity_binning) - 1):
                single_mva_loader = SingleMVALoader(directory=directory, mva_name=mva_name, param_name=param_name, rigidity_bin=rig_bin)
                yield f"{variable_name}{param_name}r{rig_bin}", single_mva_loader.load, mva_variables
                

    @staticmethod
    def load_binnings(mva_config, mva_name, config, workdir, special_binnings):
        directory = os.path.join(workdir, "mvas", mva_name, "train", "results")
        rigidity_binning = special_binnings[mva_config.get("rigidity_binning", "rigidity_search")]
        variable_name = mva_config["creates"]
        for depth, ntrees, eta in generate_mva_parameter_combinations(mva_config, config):
            param_name = f"{depth}x{ntrees}x{eta:.2f}"
            min_score = np.inf
            max_score = -np.inf
            mva_exists = True
            for bin in range(1, len(rigidity_binning) - 1):
                filename = os.path.join(directory, f"{mva_name}_r{bin}_{param_name}.json")
                if not os.path.exists(filename):
                    mva_exists = False
                    continue
                with open(filename, "r") as mva_file:
                    mva_data = json.load(mva_file)
                    if "min_score" not in mva_data or "max_score" not in mva_data:
                        continue
                    min_score = min(mva_data["min_score"], min_score)
                    max_score = max(mva_data["max_score"], max_score)
            if mva_exists:
                yield f"{variable_name}{param_name}", min_score, max_score


class MVALoader:
    def __init__(self, directory, mva_name, param_name, energy_estimator, rigidity_binning):
        self.directory = directory
        self.mva_name = mva_name
        self.param_name = param_name
        self.energy_estimator = energy_estimator
        self.rigidity_binning = rigidity_binning

    def load(self):
        mvas = {
            rig_bin: SingleMVA.load([
                    os.path.join(self.directory, f"{self.mva_name}_r{rig_bin}_{self.param_name}.json"),
                    os.path.join(self.directory, self.param_name, f"{self.mva_name}_r{rig_bin}_all.json")
            ])
            for rig_bin in range(1, len(self.rigidity_binning) - 1)
        }
        return MVA(self.energy_estimator, self.rigidity_binning, mvas)

class SingleMVALoader:
    def __init__(self, directory, mva_name, param_name, rigidity_bin):
        self.directory = directory
        self.mva_name = mva_name
        self.param_name = param_name
        self.rigidity_bin = rigidity_bin

    def load(self):
        return SingleMVA.load([
            os.path.join(self.directory, f"{self.mva_name}_r{self.rigidity_bin}_{self.param_name}.json"),
            os.path.join(self.directory, self.param_name, f"{self.mva_name}_r{self.rigidity_bin}_all.json")
        ])
