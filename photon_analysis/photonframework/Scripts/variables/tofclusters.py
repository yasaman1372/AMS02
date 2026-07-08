
import numpy as np
import awkward as ak

from tools.binnings import make_int_binning, make_lin_binning, make_log_binning
from tools.variables import depends, returns, annotate_binning as binning, make_sum


def make_tof_clusters_in_layer(layer_index):
    @returns(np.int16)
    @depends(["TofClustersInLayer"])
    @binning("n_tof_clusters")
    def _tof_clusters_in_layer(events):
        return events.TofClustersInLayer[:,layer_index]
    return _tof_clusters_in_layer


@returns(np.float32)
@depends(["TofClusterLayers", "TofClusterEnergies"])
@binning(make_lin_binning(0, 10, 100))
def mean_lower_tof_cluster_energy(events):
    in_layer_two = events.TofClusterLayers == 2
    in_layer_three = events.TofClusterLayers == 3
    mean_energy_in_layer_two = ak.sum(events.TofClusterEnergies[in_layer_two], axis=1) / np.maximum(ak.sum(in_layer_two, axis=1), 1)
    mean_energy_in_layer_three = ak.sum(events.TofClusterEnergies[in_layer_three], axis=1) / np.maximum(ak.sum(in_layer_three, axis=1), 1)
    return mean_energy_in_layer_two + mean_energy_in_layer_three


def tof_energy_in_layer(layer):
    @returns(np.float32)
    @depends(["TofClusterLayers", "TofClusterEnergies"])
    @binning(make_lin_binning(0, 20, 100))
    def _tof_energy_in_layer(events):
        return ak.sum(events.TofClusterEnergies[events.TofClusterLayers == layer], axis=1)
    return _tof_energy_in_layer


def load_variables(config, workdir, energy_estimator, binnings):
    binnings.register_binning("n_tof_clusters", make_int_binning(9))

    yield ("NTofClustersL0", make_tof_clusters_in_layer(0))
    yield ("NTofClustersL1", make_tof_clusters_in_layer(1))
    yield ("NTofClustersL2", make_tof_clusters_in_layer(2))
    yield ("NTofClustersL3", make_tof_clusters_in_layer(3))

    yield ("NLowerTofClusters", make_sum("NTofClustersL2", "NTofClustersL3", np.int16, "n_tof_clusters"))
    yield ("NUpperTofClusters", make_sum("NTofClustersL0", "NTofClustersL1", np.int16, "n_tof_clusters"))

    yield ("MeanLowerTofClusterEnergy", mean_lower_tof_cluster_energy)

    yield ("TofEnergyL0", tof_energy_in_layer(0))
    yield ("TofEnergyL1", tof_energy_in_layer(1))
    yield ("TofEnergyL2", tof_energy_in_layer(2))
    yield ("TofEnergyL3", tof_energy_in_layer(3))
