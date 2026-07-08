
import json
import os

import awkward as ak
import numpy as np

def linear_correction(mc_values, a, b):
    return a * mc_values + b

def proportional_correction(mc_values, a):
    return a * mc_values

def shift_correction(mc_values, b):
    return mc_values + b


CORRECTION_FUNCTIONS = {
    "linear": linear_correction,
    "proportional": proportional_correction,
    "shift": shift_correction,
}

CORRECTION_GUESSES = {
    "linear": dict(a=1, b=0),
    "proportional": dict(a=1),
    "shift": dict(b=0),
}


class Corrections:
    def __init__(self, variables, correction_functions, correction_parameters):
        self.variables = variables
        self.correction_functions = correction_functions
        self.correction_parameters = correction_parameters

    @staticmethod
    def load(filename): 
        variables = []
        corr_functions = {}
        corr_parameters = {}
        with open(filename) as corr_file:
            for key, (model, parameters) in json.load(corr_file).items():
                variables.append(key)
                corr_functions[key] = CORRECTION_FUNCTIONS[model]
                corr_parameters[key] = parameters
        return Corrections(variables, corr_functions, corr_parameters)

    def apply(self, chunk):
        for variable in self.variables:
            if variable not in chunk.fields:
                continue
            if np.all(chunk.McParticleID == 0):
                continue
            func = self.correction_functions[variable]
            params = self.correction_parameters[variable]
            chunk = ak.with_field(chunk, func(chunk[variable], **params), variable)
        return chunk
