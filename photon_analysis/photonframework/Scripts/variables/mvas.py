
import numpy as np
import awkward as ak

from tools.binnings import Binnings
from tools.mvas import MVA
from tools.variables import returns, depends


class LazyMVA:
    def __init__(self, load_function):
        self.load_function = load_function
        self.mva = None

    @returns(np.float32)
    def predict(self, *args, **kwargs):
        if self.mva is None:
            self.mva = self.load_function()
        return self.mva.predict(*args, **kwargs)

    @returns(np.float32)
    def predict_as_efficiency(self, *args, **kwargs):
        if self.mva is None:
            self.mva = self.load_function()
        return self.mva.predict_as_efficiency(*args, **kwargs)


def load_variables(config, workdir, energy_estimator, binnings):
    binnings = Binnings((config, workdir))
    for mva_name, mva_config in list(config.get("mvas", {}).items()) + list(config.get("regression_mvas", {}).items()):
        if "creates" in mva_config:
            for mva_var_name, load_mva, mva_dependencies in MVA.load_all(mva_config, mva_name, energy_estimator, config, workdir, binnings):
                lazy_mva = LazyMVA(load_mva)
                annotate_dependencies = depends(mva_dependencies)
                yield mva_var_name, annotate_dependencies(lazy_mva.predict)
                yield f"{mva_var_name}SignalEfficiency", annotate_dependencies(lazy_mva.predict_as_efficiency)
