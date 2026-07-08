
import numpy as np
import awkward as ak

from tools.binnings import Binnings
from tools.variables import returns, depends
from tools.selection import Selection

class LazySelection:
    def __init__(self, selection_config, energy_estimator, binnings, config, workdir, labelling):
        self.selection = None
        self.selection_config = selection_config
        self.energy_estimator = energy_estimator
        self.binnings = binnings
        self.config = config
        self.workdir = workdir
        self.labelling = labelling

    def load(self):
        self.selection = Selection.load(self.selection_config, energy_estimator=self.energy_estimator, binnings=self.binnings, config=self.config, workdir=self.workdir, labelling=self.labelling)

    @returns(bool)
    def apply(self, chunk):
        if self.selection is None:
            self.load()
        if len(chunk) == 0:
            return np.zeros(0, dtype=bool)
        return self.selection.apply(chunk)


def load_variables(config, workdir, energy_estimator, binnings):
    for selection_name, selection_config in config["selections"].items():
        var_name = f"PassesSelection{selection_name}"
        selection_function = LazySelection(selection_config, energy_estimator=energy_estimator, binnings=binnings, config=config, workdir=workdir, labelling=None).apply
        dependencies = list(selection_config["cuts"]) + ["McMomentum", energy_estimator]
        yield var_name, selection_function, dependencies
