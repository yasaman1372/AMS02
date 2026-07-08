
import numpy as np
import awkward as ak

from tools.variables import returns, depends
from tools.likelihoods import Likelihood


def load_variables(config, workdir, energy_estimator, binnings):
    if "likelihoods" in config:
        for likelihood_config in config["likelihoods"].values():
            for likelihood_varname in likelihood_config["creates"].values():
                likelihood = Likelihood.load(config, workdir, likelihood_varname, energy_estimator)
                # dependencies are missing?!
                # this is clearly not finished
                if likelihood is not None:
                    yield likelihood_varname, likelihood
