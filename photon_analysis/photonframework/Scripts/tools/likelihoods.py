#!/usr/bin/env python3

from collections import defaultdict
import json
import os

import awkward as ak
import numpy as np
from scipy.interpolate import interp1d

from tools.statistics import gaussian, landau, novosibirsk


def ecal_energy_deposited_signal_pdf(ecal_energy, rigidity, deposition_width, mip_position, mip_width, mip_fraction):
    return (1 - mip_fraction) * gaussian(ecal_energy, 0, rigidity * deposition_width) + mip_fraction * landau(ecal_energy, mip_position, mip_width)

def ecal_energy_deposited_background_pdf(ecal_energy, rigidity, decay_param, mip_position, mip_width, mip_fraction, gaussian_position, gaussian_width, gaussian_fraction):
    return (1 - gaussian_fraction) * (1 - mip_fraction) * np.exp(-ecal_energy / decay_param) / decay_param + (1 - gaussian_fraction) * mip_fraction * landau(ecal_energy, mip_position, mip_width) + gaussian_fraction * gaussian(ecal_energy, gaussian_position, gaussian_width)

def trd_amplitude_per_pathlength_signal_pdf(trd_dedx, rigidity, mu, sigma, k, delta_sigma, delta_k, delta_sigma2, delta_k2, fraction, fraction2):
    return (1 - fraction2) * (fraction * novosibirsk(trd_dedx, mu, sigma, k) + (1 - fraction) * novosibirsk(trd_dedx, mu, sigma + delta_sigma, k + delta_k)) + fraction2 * novosibirsk(trd_dedx, mu, sigma + delta_sigma + delta_sigma2, k + delta_k + delta_k2)

def trd_amplitude_per_pathlength_background_pdf(trd_dedx, rigidity, mu, sigma, k, delta_sigma, delta_k, fraction):
    return fraction * novosibirsk(trd_dedx, mu, sigma, k) + (1 - fraction) * novosibirsk(trd_dedx, mu, sigma + delta_sigma, k + delta_k)


PDFS = {
    "ecal_energy_deposited": dict(signal=ecal_energy_deposited_signal_pdf, background=ecal_energy_deposited_background_pdf),
    "trd_amplitude_per_pathlength": dict(signal=trd_amplitude_per_pathlength_signal_pdf, background=trd_amplitude_per_pathlength_background_pdf),
}

INPUT_VARIABLES = {
    "ecal": "EcalEnergyDeposited",
    "trd": "TrdHitAmplitudePerPathlength",
}


class ParametrizedPdf:
    def __init__(self, pdf, parametrizations, rigidity_range):
        self.pdf = pdf
        self.parametrizations = parametrizations
        self.rig_min, self.rig_max = rigidity_range

    @staticmethod
    def load(filename, key):
        with open(filename, "r") as file:
            pdf_config = json.load(file)
            pdf_shape = PDFS[pdf_config["shape"]][key]
            rigidities = []
            parameters = defaultdict(lambda: [])
            for rigidity, parametrization in pdf_config[key]:
                rigidities.append(rigidity)
                for param_name, param_value in parametrization["parameters"].items():
                    parameters[param_name].append(param_value)
            rigidities = np.array(rigidities)
            parameters = {key: np.array(values) for key, values in parameters.items()}
            for values in parameters.values():
                assert values.shape == rigidities.shape
            rig_min = np.min(rigidities)
            rig_max = np.max(rigidities)
            parametrizations = {key: interp1d(rigidities, values, kind="cubic") for key, values in parameters.items()}
            return ParametrizedPdf(pdf_shape, parametrizations, (rig_min, rig_max))

    def evaluate(self, rigidities, values_x):
        rigidities = np.clip(rigidities.to_numpy(), self.rig_min, self.rig_max)
        parameters = {param: self.parametrizations[param](rigidities) for param in self.parametrizations}
        return self.pdf(values_x, rigidities, **parameters)


class Likelihood:
    def __init__(self, input_variable, energy_estimator, signal_pdf, background_pdf):
        self.input_variable = input_variable
        self.energy_estimator = energy_estimator
        self.signal_pdf = signal_pdf
        self.background_pdf = background_pdf
        self._dependencies = [self.input_variable, self.energy_estimator]
        self._dtype = np.float32


    @staticmethod
    def load(config, workdir, likelihood_varname, energy_estimator):
        param_filename = None
        input_variable = None
        if "likelihoods" not in config:
            return None
        for likelihood_name, likelihood_config in config["likelihoods"].items():
            for likelihood_type, created_varname in likelihood_config["creates"].items():
                if likelihood_varname == created_varname:
                    input_variable = INPUT_VARIABLES[likelihood_type]
                    param_filename = os.path.join(workdir, "likelihoods", likelihood_name, "parametrization", "results", f"{likelihood_varname}_parameters.json")
                    break
        if param_filename is None or not os.path.isfile(param_filename):
            return None
        signal_pdf = ParametrizedPdf.load(param_filename, "signal")
        background_pdf = ParametrizedPdf.load(param_filename, "background")
        return Likelihood(input_variable, energy_estimator, signal_pdf, background_pdf)
        

    def __call__(self, chunk):
        values_x = chunk[self.input_variable]
        rigidities = np.abs(chunk[self.energy_estimator])
        signal_likelihoods = self.signal_pdf.evaluate(rigidities, values_x)
        background_likelihoods = self.background_pdf.evaluate(rigidities, values_x)
        if signal_likelihoods.ndim == 2:
            signal_likelihoods = np.nan_to_num(ak.prod(signal_likelihoods, axis=1)**(1 / np.maximum(1, ak.num(signal_likelihoods))), nan=0.5)
            background_likelihoods = np.nan_to_num(ak.prod(background_likelihoods, axis=1)**(1 / np.maximum(1, ak.num(background_likelihoods))), nan=0.5)
        assert signal_likelihoods.ndim == 1
        return signal_likelihoods / (signal_likelihoods + background_likelihoods)
