
import numpy as np
import awkward as ak

from tools.binnings import make_int_binning, make_lin_binning, make_log_binning
from tools.variables import returns, depends, annotate_binning as binning
from tools.constants import MC_PARTICLE_IDS, MC_PROCESS_IDS

@returns(np.int16)
@depends(["McSecondaryMomenta"])
@binning(make_int_binning(10))
def n_mc_secondaries(events):
    return ak.to_numpy(ak.num(events.McSecondaryMomenta)).astype(np.int16)


@returns(np.int16)
@depends(["McSecondaryMomenta"])
@binning("mc_particle_id")
def select_highest_momentum_mc_secondary_pid(events):
    result = -np.ones(len(events), dtype=np.int16)
    mask = n_mc_secondaries(events) > 0
    result[mask] = ak.argmax(events.McSecondaryMomenta[mask], axis=1)
    return result

def select_mc_secondary_pid(process_id, particle_id):
    @returns(np.int16)
    @depends(["McSecondaryParticleIDs", "McSecondaryCreatingProcesses"])
    @binning("mc_particle_id")
    def _selected_mc_secondary_pid(events):
        result = -np.ones(len(events), dtype=np.int16)
        for index in range(np.max(ak.num(events.McSecondaryParticleIDs))):
            mask = ak.to_numpy(ak.num(events.McSecondaryParticleIDs) > index)
            sel = np.zeros(len(mask), dtype=bool)
            mcSecondaryCreatingId = events.McSecondaryCreatingProcesses[mask]
            mcSecondaryParticleID = events.McSecondaryParticleIDs[mask]
            sel[mask] = (mcSecondaryCreatingId[:,index] == process_id) & (mcSecondaryParticleID[:,index] == particle_id)
            mask[np.invert(sel)] = False            
            result[mask] = index
        return result
    return _selected_mc_secondary_pid


def mc_secondary_pid_momentum(index_branch):
    @returns(np.float32)
    @depends([index_branch, "McSecondaryMomenta"])
    @binning("energy")
    def _mc_secondary_pid_momentum(events):
        result = np.zeros(len(events), dtype=np.float32)
        sel = events[index_branch] >= 0
        result[sel] = events.McSecondaryMomenta[np.arange(len(events))[sel], events[index_branch][sel]]
        return result
    return _mc_secondary_pid_momentum


def mc_secondary_pid_coordinate(index_branch, coordinate_index):
    @returns(np.float32)
    @depends([index_branch, "McSecondaryPositions"])
    @binning("coordinate")
    def _mc_secondary_pid_coordinate(events):
        result = np.zeros(len(events), dtype=np.float32)
        sel = events[index_branch] >= 0
        result[sel] = events.McSecondaryPositions[np.arange(len(events))[sel], 3 * events[index_branch][sel] + coordinate_index]
        return result
    return _mc_secondary_pid_coordinate

def mc_secondary_pid_direction(index_branch, direction_index, n_axes):
    @returns(np.float32)
    @depends([index_branch, "McSecondaryDirections"])
    @binning("direction")
    def _mc_secondary_pid_direction(events):
        result = np.zeros(len(events), dtype=np.float32)
        sel = events[index_branch] >= 0
        result[sel] = events.McSecondaryDirections[np.arange(len(events))[sel], n_axes * events[index_branch][sel] + direction_index]
        return result
    return _mc_secondary_pid_direction


@returns(bool)
@depends(["TrkTrackBestPairIndex", "TrkTrackPairTrackIndices", "McSecondaryCreatingProcesses"])
@binning("bool")
def mc_secondary_creation_process_is_pair_production(events):
    result = np.zeros(len(events), dtype=bool)
    mask = (ak.num(events.TrkTrackPairTrackIndices) > 0)
    bpindex = events.TrkTrackBestPairIndex[mask]
    array_indices = np.arange(len(events))[mask]
    indices = events.TrkTrackPairTrackIndices
    id1 = indices[array_indices, 2 * bpindex]
    id2 = indices[array_indices, 2 * bpindex +1]
    result[mask] = (events.McSecondaryCreatingProcesses[id1] == MC_PROCESS_IDS["PairProduction"]) & (events.McSecondaryCreatingProcesses[id2] == MC_PROCESS_IDS["PairProduction"])
    return result


def load_variables(config, workdir, energy_estimator, binnings):
    # mc secondaries
    yield ("NMcSecondaryParticles", n_mc_secondaries)
    yield ("McSecondaryPairProductionElectronId", select_mc_secondary_pid(MC_PROCESS_IDS["PairProduction"], MC_PARTICLE_IDS["Electron"]))
    yield ("McSecondaryPairProductionPositronId", select_mc_secondary_pid(MC_PROCESS_IDS["PairProduction"], MC_PARTICLE_IDS["Positron"]))
    yield ("McHighestMomentumSecondaryId", select_highest_momentum_mc_secondary_pid)
    yield ("McSecondaryPairProductionElectronMomentum", mc_secondary_pid_momentum("McSecondaryPairProductionElectronId"))
    yield ("McSecondaryPairProductionPositronMomentum", mc_secondary_pid_momentum("McSecondaryPairProductionPositronId"))
    yield ("McHighestMomentumSecondaryMomentum", mc_secondary_pid_momentum("McHighestMomentumSecondaryId"))
    yield ("McSecondaryPairProductionElectronCoordX", mc_secondary_pid_coordinate("McSecondaryPairProductionElectronId", 0))
    yield ("McSecondaryPairProductionElectronCoordY", mc_secondary_pid_coordinate("McSecondaryPairProductionElectronId", 1))
    yield ("McSecondaryPairProductionElectronCoordZ", mc_secondary_pid_coordinate("McSecondaryPairProductionElectronId", 2))
    yield ("McSecondaryPairProductionElectronDirX", mc_secondary_pid_direction("McSecondaryPairProductionElectronId", 0, 3))
    yield ("McSecondaryPairProductionElectronDirY", mc_secondary_pid_direction("McSecondaryPairProductionElectronId", 1, 3))
    yield ("McSecondaryPairProductionElectronDirZ", mc_secondary_pid_direction("McSecondaryPairProductionElectronId", 2, 3))
    yield ("McSecondaryPairProductionPositronCoordX", mc_secondary_pid_coordinate("McSecondaryPairProductionPositronId", 0))
    yield ("McSecondaryPairProductionPositronCoordY", mc_secondary_pid_coordinate("McSecondaryPairProductionPositronId", 1))
    yield ("McSecondaryPairProductionPositronCoordZ", mc_secondary_pid_coordinate("McSecondaryPairProductionPositronId", 2))
    yield ("McSecondaryPairProductionPositronDirX", mc_secondary_pid_direction("McSecondaryPairProductionPositronId", 0, 3))
    yield ("McSecondaryPairProductionPositronDirY", mc_secondary_pid_direction("McSecondaryPairProductionPositronId", 1, 3))
    yield ("McSecondaryPairProductionPositronDirZ", mc_secondary_pid_direction("McSecondaryPairProductionPositronId", 2, 3))
    yield ("McHighestMomentumSecondaryCoordX", mc_secondary_pid_coordinate("McHighestMomentumSecondaryId", 0))
    yield ("McHighestMomentumSecondaryCoordY", mc_secondary_pid_coordinate("McHighestMomentumSecondaryId", 1))
    yield ("McHighestMomentumSecondaryCoordZ", mc_secondary_pid_coordinate("McHighestMomentumSecondaryId", 2))
    yield ("McHighestMomentumSecondaryDirX", mc_secondary_pid_direction("McHighestMomentumSecondaryId", 0, 3))
    yield ("McHighestMomentumSecondaryDirY", mc_secondary_pid_direction("McHighestMomentumSecondaryId", 1, 3))
    yield ("McHighestMomentumSecondaryDirZ", mc_secondary_pid_direction("McHighestMomentumSecondaryId", 2, 3))
    #yield ("McSecondaryProcessIsPairProduction", mc_secondary_creation_process_is_pair_production)

