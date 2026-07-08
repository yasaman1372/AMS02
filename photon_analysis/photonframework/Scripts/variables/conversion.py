
import numpy as np
import awkward as ak

from tools.binnings import make_int_binning, make_lin_binning, make_log_binning, make_bool_binning
from tools.constants import TRK_LAYER_POSITION_Z, TRD_Z_LOWER, TRD_Z_UPPER, MC_PARTICLE_IDS, TRD_CONTOUR_BOTTOM, TRD_CONTOUR_TOP, TOF_Layer_Bars_Contours, MC_PROCESS_IDS
from tools.variables import returns, depends, annotate_binning as binning

from matplotlib.path import Path

def make_distance_between_tracks(first_X_branch, second_X_branch, first_Y_branch, second_Y_branch):
    @returns(np.float32)
    @depends([first_X_branch, second_X_branch, first_Y_branch, second_Y_branch])
    @binning('radius')
    def _distance_between_tracks(events):
        mask = ak.num(events[first_X_branch]) > 0
        result = np.ones(len(events)) * 1000
        x1 = events[first_X_branch]
        x2 = events[second_X_branch]
        y1 = events[first_Y_branch]
        y2 = events[second_Y_branch]
        dx = x1[mask] - x2[mask]
        dy = y1[mask] - y2[mask]
        result[mask] = np.sqrt(dx**2 + dy**2)
        return result
    return _distance_between_tracks

def make_distance_xy(first_xy_branch, second_xy_branch):
    @returns(np.float32)
    @depends([first_xy_branch, second_xy_branch])
    @binning('radius')
    def _distance_xy(events):
        xy1 = events[first_xy_branch]
        xy2 = events[second_xy_branch]
        result = np.abs(xy1 - xy2)
        return result
    return _distance_xy

@returns(np.float32)
@depends(['TrkTrackElectronRigidities', 'TrkTrackBestPairIndex', 'TrkTrackPairTrackIndices'])
@binning('rigidity_asymmetry')
def trk_track_best_pair_rigidity_asymmetry(events):
    mask = (ak.num(events.TrkTrackElectronRigidities) > 0) & (ak.num(events.TrkTrackPairTrackIndices) > 0)
    result = np.ones(len(events)) * 1000
    bpindex = events.TrkTrackBestPairIndex[mask]
    array_indices = np.arange(len(events))[mask]
    indices = events.TrkTrackPairTrackIndices
    rigidities = events.TrkTrackElectronRigidities
    id1 = indices[array_indices, 2 * bpindex]
    id2 = indices[array_indices, 2 * bpindex +1]
    rig1 = rigidities[array_indices, id1]
    rig2 = rigidities[array_indices, id2]
    result[mask] = (rig1+rig2)/np.abs(rig1 - rig2)
    return result

def make_trk_track_best_pair_pid_rigidity(pid):
    @returns(np.float32)
    @depends(['TrkTrackElectronRigidities', 'TrkTrackBestPairIndex', 'TrkTrackPairTrackIndices'])
    @binning('energy')
    def _trk_track_best_pair_pid_rigidity(events):
        result = np.zeros(len(events))
        mask = (ak.num(events.TrkTrackElectronRigidities) > 0) & (ak.num(events.TrkTrackPairTrackIndices) > 0)
        rig1 = ak.to_numpy(events.TrkTrackBestPairFirstRigidity[mask])
        rig2 = ak.to_numpy(events.TrkTrackBestPairSecondRigidity[mask])
        Rig = np.array([rig1,rig2]).T
        if pid == 2:
            result[mask] = np.abs(np.max(Rig, axis = 1))
        elif pid == 3:
            result[mask] = np.abs(np.min(Rig, axis = 1))
        return result
    return _trk_track_best_pair_pid_rigidity

def make_tof_clusters_in_layer(layer):
    @returns(np.int16)
    @depends(["TofClustersInLayer"])
    @binning(make_int_binning(10))
    def _tof_clusters_in_layer(events):
        mask = ak.num(events.TofClustersInLayer) > 0
        result = np.zeros(len(events), dtype = np.int16)
        clusters = events.TofClustersInLayer[mask]
        result[mask] = clusters[:, layer]
        return result
    return _tof_clusters_in_layer

def make_tof_clusters_in_upper_lower_tof(tof_section):
    if tof_section == 'upper':
        layer1 = "TofClustersInLayer0"
        layer2 = "TofClustersInLayer1"
    elif tof_section == 'lower':
        layer1 = "TofClustersInLayer2"
        layer2 = "TofClustersInLayer3"
    @returns(np.int16)
    @depends([layer1, layer2])
    @binning(make_int_binning(10))
    def _tof_clusters_in_upper_lower_tof(events):
        result = np.zeros(len(events), dtype = np.int16)
        clusters1 = events[layer1]
        clusters2 = events[layer2]
        result = clusters1 + clusters2
        return result
    return _tof_clusters_in_upper_lower_tof

def make_trk_track_has_good_inner_track(layer_pattern_branch):
    @returns(bool)
    @depends([layer_pattern_branch])
    @binning(make_bool_binning())
    def _trk_track_has_good_inner_track(events):
        L = [1<<x for x in range(10)]
        pattern = events[layer_pattern_branch]
        has2 = (pattern & L[2]) != 0
        has3_4 = (pattern & (L[3] | L[4])) != 0
        has5_6 = (pattern & (L[5] | L[6])) != 0
        has7_8 = (pattern & (L[7] | L[8])) != 0

        result = has2 & has3_4 & has5_6 & has7_8
        return result
    return _trk_track_has_good_inner_track

def make_tof_energy_per_cluster(layer):
    @returns(np.float32)
    @depends(["TofClusterEnergies", "TofClusterLayers", "TofClustersInLayer"])
    @binning('energy')
    def _tof_energy_per_cluster(events):
        mask = (ak.num(events.TofClusterEnergies) > 0) & (ak.num(events.TofClusterLayers) > 0)
        result = np.zeros(len(events))
        cluster_layer_indices = events.TofClusterLayers[mask]
        energies = events.TofClusterEnergies[mask]
        array_index = np.arange(len(energies))
        clusters_in_layer = events.TofClustersInLayer[mask]
        layer_mask = (cluster_layer_indices == layer)
        energie_in_layer = ak.sum(energies[layer_mask], axis = 1)
        result[mask] = energie_in_layer/clusters_in_layer[array_index, layer]
        return result
    return _tof_energy_per_cluster

def make_tof_energy_in_layer(layer):
    @returns(np.float32)
    @depends(["TofClusterEnergies", "TofClusterLayers", "TofClustersInLayer"])
    @binning('energy')
    def _tof_energy_in_layer(events):
        mask = (ak.num(events.TofClusterEnergies) > 0) & (ak.num(events.TofClusterLayers) > 0)
        result = np.zeros(len(events))
        cluster_layer_indices = events.TofClusterLayers[mask]
        energies = events.TofClusterEnergies[mask]
        layer_mask = (cluster_layer_indices == layer)
        energie_in_layer = ak.sum(energies[layer_mask], axis = 1)
        result[mask] = energie_in_layer
        return result
    return _tof_energy_in_layer



def make_tof_energy_in_layer_for_one_cluster_cut(layer, default_value = 7):
    @returns(np.float32)
    @depends(["TofClusterEnergies", "TofClusterLayers", "TofClustersInLayer"])
    @binning('energy')
    def _tof_energy_in_layer_for_one_cluster_cut(events):
        array_index = np.arange(len(events))
        mask = events.TofClustersInLayer[array_index, layer] == 1
        result = np.ones(len(events)) * default_value
        energies = events.TofClusterEnergies[mask]
        cluster_layer_indices = events.TofClusterLayers[mask]
        layer_mask = (cluster_layer_indices == layer)
        energie_in_layer = ak.flatten(energies[layer_mask])
        result[mask] = energie_in_layer
        return result
    return _tof_energy_in_layer_for_one_cluster_cut


def make_tof_max_charge_in_layer(layer):
    @returns(bool)
    @depends(["TofClusterCharges", "TofClusterLayers", "TofClustersInLayer"])
    @binning(bool)
    def _tof_max_charge_in_layer(events):
        mask = (ak.num(events.TofClusterEnergies) > 0) & (ak.num(events.TofClusterLayers) > 0)
        result = np.ones(len(events)) * 1000
        cluster_layer_indices = events.TofClusterLayers[mask]
        charges = events.TofClusterCharges[mask]
        layer_mask = (cluster_layer_indices == layer)
        max_charge = ak.max(charges[layer_mask], axis = 1)
        result[mask] = ak.to_numpy(max_charge) 
        return result
    return _tof_max_charge_in_layer


def make_trk_track_best_pair_electron_positron_rigidity(ID):
    @returns(np.float32)
    @depends(['TrkTrackBestPairFirstRigidity', 'TrkTrackBestPairSecondRigidity'])
    @binning("rigidity")
    def _trk_track_best_pair_electron_positron_rigidity(events):
        mask = (events.TrkTrackBestPairFirstRigidity*events.TrkTrackBestPairSecondRigidity < 0) & (events.TrkTrackBestPairFirstRigidity > -10000) & (events.TrkTrackBestPairSecondRigidity > -10000)
        result = np.ones(len(events))*-10000
        Frig = ak.to_numpy(events.TrkTrackBestPairFirstRigidity[mask])
        Srig = ak.to_numpy(events.TrkTrackBestPairSecondRigidity[mask])
        Rig = np.array([Frig,Srig]).T
        if ID == MC_PARTICLE_IDS["Electron"]:
            result[mask] = np.max(Rig, axis = 1)
        if ID == MC_PARTICLE_IDS['Positron']:
            result[mask] = np.min(Rig, axis = 1)
        return result
    return _trk_track_best_pair_electron_positron_rigidity

def make_associated_tof_cluster_in_layer_coordinate(track_index, layer_index):
    @returns(np.float32)
    @depends(["TrkTrackAssociatedTofClusterIndices", "TrkTrackBestPairIndex", "TofClusterLayers", "TofClusterCoordinates", "TrkTrackPairTrackIndices"])
    @binning(make_lin_binning(-100, 100, 200))
    def _associated_tof_cluster_in_layer_coordinate(events):
        if layer_index == 0 or layer_index == 3:
            coord_offset = 1
        elif layer_index == 1 or layer_index == 2:
            coord_offset = 0
        result = np.ones(len(events)) * -1000
        bpindex = events.TrkTrackBestPairIndex
        array_indices = np.arange(len(bpindex))
        track_indices = events.TrkTrackPairTrackIndices[array_indices,bpindex * 2 + track_index]
        AssociatedTofClusterIndex = events.TrkTrackAssociatedTofClusterIndices
        AssociatedClusterInLayer = AssociatedTofClusterIndex[array_indices, track_indices*4+layer_index]
        mask = (AssociatedClusterInLayer >= 0)
        AssociatedClusterInLayerMasked = AssociatedClusterInLayer[mask]
        TofClusterCoordinate = events.TofClusterCoordinates[mask]
        array_indices = np.arange(len(TofClusterCoordinate))
        result[mask] = TofClusterCoordinate[array_indices, AssociatedClusterInLayerMasked*3+coord_offset]
        return result
    return _associated_tof_cluster_in_layer_coordinate


def make_trk_track_best_pair_pass_tof_bar_in_layer(layer_index, track_X_branch, track_Y_branch):
    @returns(np.float32)
    @depends([track_X_branch, track_Y_branch])
    @binning(make_lin_binning(-1.25,10.25,23))
    def _trk_track_best_pair_pass_tof_bar_in_layer(events):
        results = np.ones(len(events))*-1
        bar_contours = TOF_Layer_Bars_Contours[layer_index]
        bar_contours = [Path(bar) for bar in bar_contours]
        x = events[track_X_branch]
        y = events[track_Y_branch]
        points = np.array([x,y]).T
        inside = np.array([bar.contains_points(points) for bar in bar_contours])
        inside_bars = np.array([x*(i+1) for i,x in enumerate(inside)])
        hit_bar = (np.sum(inside_bars, axis=0)/np.sum(inside, axis=0)) -1
        results[np.sum(inside, axis = 0, dtype = bool)] = hit_bar[np.sum(inside, axis = 0, dtype = bool)]
        return results
    return _trk_track_best_pair_pass_tof_bar_in_layer

def make_trk_track_best_pair_associated_tof_bar_in_layer(track_index, layer_index):
    @returns(np.float32)
    @depends(["TrkTrackAssociatedTofClusterIndices", "TrkTrackBestPairIndex", "TofClusterLayers", "TofClusterCoordinates", "TrkTrackPairTrackIndices"])
    @binning(make_lin_binning(-1.25,10.25,23))
    def _trk_track_best_pair_associated_tof_bar_in_layer(events):
        result = np.ones(len(events)) * -10
        mask1 = (ak.num(events.TrkTrackPairTrackIndices) >= 2)
        bar_contours = TOF_Layer_Bars_Contours[layer_index]
        bar_contours = [Path(bar) for bar in bar_contours]
        bpindex = events.TrkTrackBestPairIndex[mask1]
        array_indices = np.arange(len(bpindex))
        track_indices = events.TrkTrackPairTrackIndices[mask1][array_indices,bpindex * 2 + track_index]
        AssociatedTofClusterIndex = events.TrkTrackAssociatedTofClusterIndices[mask1]
        AssociatedClusterInLayer = AssociatedTofClusterIndex[array_indices, track_indices*4+layer_index]
        mask = (AssociatedClusterInLayer >= 0)
        AssociatedClusterInLayerMasked = AssociatedClusterInLayer[mask]
        TofClusterCoordinates = events.TofClusterCoordinates[mask1][mask]
        TofClusterCoordinates = events.TofClusterCoordinates[mask1][mask]
        array_indices = np.arange(len(TofClusterCoordinates))
        x = TofClusterCoordinates[array_indices, AssociatedClusterInLayerMasked*3]
        y = TofClusterCoordinates[array_indices, AssociatedClusterInLayerMasked*3+1]
        Points = np.array([x,y]).T
        inside = np.array([bar.contains_points(Points) for bar in bar_contours])
        inside_bars = np.array([x*(i+1) for i,x in enumerate(inside)])
        hit_bar = (np.sum(inside_bars, axis=0)/np.sum(inside, axis=0)) -1
        hit_bar = np.nan_to_num(hit_bar, nan = -1)
        res_index = np.arange(len(events))
        result[res_index[mask1][mask]] = hit_bar
        return result
    return _trk_track_best_pair_associated_tof_bar_in_layer

def make_trk_track_best_pair_distance_associated_tof_bar_passed_bar_in_layer(associated_branch, passed_branch):
    @returns(np.float32)
    @depends([associated_branch, passed_branch])
    @binning(make_lin_binning(-1.25,10.25,23))
    def _trk_track_best_pair_distance_associated_tof_bar_passed_bar_in_layer(events):
        results = np.ones(len(events)) * 100
        Associated_bar = events[associated_branch]
        Passed_bar = events[passed_branch]
        mask = (Associated_bar >= 0) & (Passed_bar >= 0)
        Associated_bar = Associated_bar[mask]
        Passed_bar = Passed_bar[mask]
        distance = np.abs(Associated_bar - Passed_bar)
        results[mask] = distance
        return results
    return _trk_track_best_pair_distance_associated_tof_bar_passed_bar_in_layer

def make_trk_track_best_pair_matching_clusters_in_layer(layer_index):
    @returns(bool)
    @depends([f"TrkTrackBestPairFirstDistanceAssociatedTofBarPassedTofBarLayer{layer_index}", f"TrkTrackBestPairSecondDistanceAssociatedTofBarPassedTofBarLayer{layer_index}"])
    @binning(bool)
    def _trk_track_best_pair_matching_clusters_in_layer(events):
        results = np.zeros(len(events), dtype=bool)
        FirstDistance = events[f"TrkTrackBestPairFirstDistanceAssociatedTofBarPassedTofBarLayer{layer_index}"]
        SecondDistance = events[f"TrkTrackBestPairSecondDistanceAssociatedTofBarPassedTofBarLayer{layer_index}"]
        mask = (FirstDistance > -1) & (SecondDistance > -1)
        FirstDistance = FirstDistance[mask]
        SecondDistance = SecondDistance[mask]
        if layer_index <= 1:
            matches = (FirstDistance <= 0.5) & (SecondDistance <= 0.5)
        elif layer_index > 1:
            matches = (FirstDistance <= 0.5) | (SecondDistance <= 0.5)
        results[mask] = matches
        return results
    return _trk_track_best_pair_matching_clusters_in_layer

@returns(bool)
@depends(["TrkTrackBestPairFirstAssociatedTofClusterBarInLayer2", "TrkTrackBestPairSecondAssociatedTofClusterBarInLayer2"])
@binning(bool)
def trk_track_best_pair_matching_cluster_in_layer2_in_central_bars(events):
    FirstBar = events.TrkTrackBestPairFirstAssociatedTofClusterBarInLayer2
    SecondBar = events.TrkTrackBestPairSecondAssociatedTofClusterBarInLayer2
    FirstBar_inner = (FirstBar >= 1) & (FirstBar <= 9)
    SecondBar_innner = (SecondBar >= 1) & (SecondBar <= 9)
    results = FirstBar_inner | SecondBar_innner
    return results



def make_min_max(extremum, first_branch, second_branch):
    @returns(np.float32)
    @depends([first_branch, second_branch])
    @binning(make_lin_binning(-250,250,500))
    def _min_max(events):
        result = np.zeros(len(events))
        f = events[first_branch]
        s = events[second_branch]
        fs = np.array([f,s]).T
        if extremum == 'min':
            result = np.min(fs, axis = 1)
        elif extremum == 'max':
            result = np.max(fs, axis = 1)
        return result
    return _min_max

def make_min_max_abs(extremum, first_branch, second_branch):
    @returns(np.float32)
    @depends([first_branch, second_branch])
    @binning(make_lin_binning(-250,250,500))
    def _min_max(events):
        result = np.zeros(len(events))
        f = events[first_branch]
        s = events[second_branch]
        fs = np.array([f,s]).T
        array_indices = np.arange(len(fs))
        if extremum == 'min':
            result = fs[array_indices, np.argmin(np.abs(fs), axis = 1)]
        elif extremum == 'max':
            result = fs[array_indices, np.argmax(np.abs(fs), axis = 1)]
        return result
    return _min_max

def make_trk_track_best_pair_min_distane_is_in_upper_lower_trd(trd_edge):
    @returns(bool)
    @depends([f"TrkTrackBestPairMinDistanceExtrapolatedXAt{trd_edge}TRD", f"TrkTrackBestPairMinDistanceExtrapolatedYAt{trd_edge}TRD"])
    @binning(make_bool_binning())
    def _trk_track_best_pair_min_distane_is_in_upper_lower_trd(events):
        result = np.zeros(len(events), dtype = bool)
        if trd_edge == "Lower":
            Conture = Path(TRD_CONTOUR_BOTTOM)
        elif trd_edge == 'Upper':
            Conture = Path(TRD_CONTOUR_TOP)

        x = events[f"TrkTrackBestPairMinDistanceExtrapolatedXAt{trd_edge}TRD"]
        y = events[f"TrkTrackBestPairMinDistanceExtrapolatedYAt{trd_edge}TRD"]
        points = np.array([x,y]).T
        result = Conture.contains_points(points)
        return result
    return _trk_track_best_pair_min_distane_is_in_upper_lower_trd

def make_trk_track_best_pair_tracker_hit_charge_in_layer(track_index, layer_index):
    @returns(np.float32)
    @depends(["TrkTrackLayerCharges", "TrkTrackBestPairIndex", "TrkTrackPairTrackIndices"])
    @binning("charge")
    def _trk_track_best_pair_tracker_hit_charge_in_layer(events):
        result = np.ones(len(events), dtype=np.float32)*1000
        mask = (ak.num(events.TrkTrackLayerCharges) > 17) & (ak.num(events.TrkTrackPairTrackIndices) > 0)
        bpindex = events.TrkTrackBestPairIndex[mask]
        array_indices = np.arange(len(events))[mask]
        indices = events.TrkTrackPairTrackIndices
        track_indices = indices[array_indices, bpindex*2 + track_index]
        TrkTrackLayerCharges = events.TrkTrackLayerCharges
        Charge = TrkTrackLayerCharges[array_indices, track_indices*9+layer_index]
        result[mask] = Charge
        return result
    return _trk_track_best_pair_tracker_hit_charge_in_layer

def make_trk_hit_max_charge_in_layer(layer_index):
    @returns(np.float32)
    @depends(['TrkHitCharges', 'TrkHitLayers'])
    @binning("charge")
    def _trk_hit_max_charge_in_layer(events):
        result = np.ones(len(events))*0
        mask = ak.any(events.TrkHitLayers == layer_index, axis = 1)
        #array_index = np.arange(len(events))[mask]
        layer_indices = events.TrkHitLayers[mask]
        layer_indices_repeated = ak.unflatten(np.repeat(ak.flatten(layer_indices), 2), ak.num(layer_indices)*2)
        charges = events.TrkHitCharges[mask]
        max_charge_in_layer = ak.max(charges[layer_indices_repeated == layer_index], axis = 1)
        result[mask] = max_charge_in_layer
        return result
    return _trk_hit_max_charge_in_layer

def make_simple_trk_hit_max_charge_in_layer(layer_number):
    @returns(np.float32)
    @depends(["TrkMaxLayerCharges"])
    @binning("charge")
    def _trk_hit_max_charge_in_layer(events):
        return events.TrkMaxLayerCharges[:,layer_number - 1]
    return _trk_hit_max_charge_in_layer

def make_bremsstrahlung_corrected_energy(energy_branch, bremsstrahlung_brach):
    @returns(np.float32)
    @depends([energy_branch, bremsstrahlung_brach, "McSecondaryPairProductionElectronId", "McSecondaryPairProductionPositronId"])
    @binning("energy")
    def _bremsstrahlung_corrected_energy(events):
        result = np.zeros(len(events))
        energy = events[energy_branch]
        mask = (events.McSecondaryPairProductionElectronId >= 0) & (events.McSecondaryPairProductionPositronId >= 0)
        bremsstrahlung = events[bremsstrahlung_brach][mask]
        ElectronID = events.McSecondaryPairProductionElectronId[mask]
        PositronID = events.McSecondaryPairProductionElectronId[mask]
        array_index = np.arange(len(bremsstrahlung))    
        electron_bremsstrahlung = bremsstrahlung[array_index, ElectronID]
        positron_bremsstrahlung = bremsstrahlung[array_index, PositronID]
        bremsstrahlung_lost = electron_bremsstrahlung+positron_bremsstrahlung
        result[mask] = bremsstrahlung_lost
        return energy+result
    return _bremsstrahlung_corrected_energy

def make_bremsstrahlung_fit_corrected_energy(energy_branch, correction_factor):
    @returns(np.float32)
    @depends([energy_branch])
    @binning("energy")
    def _bremsstrahlung_fit_corrected_energy(events):
        energy = events[energy_branch]
        correction_factors = np.ones(len(events))*correction_factor
        corrected_energy = energy/(1-correction_factors)
        return corrected_energy
    return _bremsstrahlung_fit_corrected_energy

def make_linlog_corrected_energy(energy_branch, a, b):
    @returns(np.float32)
    @depends([energy_branch])
    @binning("energy")
    def _function_corrected_energy(events):
        energy = events[energy_branch]
        corrected_energy = energy/ (a*np.log(energy)+1+b)
        return corrected_energy
    return _function_corrected_energy
        
        


def load_variables(config, workdir, energy_estimator, binnings):
    binnings.register_binning("pair_index", make_int_binning(64))
    binnings.register_binning("tof_cluster_index", make_int_binning(20))
    binnings.register_binning("trk_n_hits", make_int_binning(10))
    binnings.register_binning("trk_layer_pattern", make_int_binning(512))
    binnings.register_binning("chisq", make_log_binning(1e-3, 1e3, 100))
    binnings.register_binning("invariant_mass", make_log_binning(1e-5, 1e3, 200))
    binnings.register_binning("angle", make_lin_binning(0, np.pi / 2, 100))
    binnings.register_binning("deposited_energy", make_log_binning(1e-3, 1e2, 100))
    binnings.register_binning("rigidity_asymmetry", make_lin_binning(-1.5, 1.5, 300))


    yield ("TrkTrackBestPairRigidityAsymmetry", trk_track_best_pair_rigidity_asymmetry)
    yield ("TrkTrackBestPairElectronTrackRigidity", make_trk_track_best_pair_pid_rigidity(3))
    yield ("TrkTrackBestPairPositronTrackRigidity", make_trk_track_best_pair_pid_rigidity(2))

    yield ("TrkTrackBestPairDistanceAtUpperTof", make_distance_between_tracks("TrkTrackBestPairFirstCoordinateXAtUpperTof", "TrkTrackBestPairSecondCoordinateXAtUpperTof", "TrkTrackBestPairFirstCoordinateYAtUpperTof", "TrkTrackBestPairSecondCoordinateYAtUpperTof"))
    yield ("TrkTrackBestPairDistanceAtLowerTof", make_distance_between_tracks("TrkTrackBestPairFirstCoordinateXAtLowerTof", "TrkTrackBestPairSecondCoordinateXAtLowerTof", "TrkTrackBestPairFirstCoordinateYAtLowerTof", "TrkTrackBestPairSecondCoordinateYAtLowerTof"))

    yield ("TrkTrackBestPairFirstHasGoodInnerTrack", make_trk_track_has_good_inner_track("TrkTrackBestPairFirstTrackLayerPattern"))
    yield ("TrkTrackBestPairSecondHasGoodInnerTrack", make_trk_track_has_good_inner_track("TrkTrackBestPairSecondTrackLayerPattern"))

    yield ("TrkTrackBestPairTracksDistanceXAtLowerTRD", make_distance_xy("TrkTrackBestPairFirstExtrapolatedXAtLowerTRD", "TrkTrackBestPairSecondExtrapolatedXAtLowerTRD"))
    yield ("TrkTrackBestPairTracksDistanceYAtLowerTRD", make_distance_xy("TrkTrackBestPairFirstExtrapolatedYAtLowerTRD", "TrkTrackBestPairSecondExtrapolatedYAtLowerTRD"))

    yield ("TofClustersInLayer0", make_tof_clusters_in_layer(0))
    yield ("TofClustersInLayer1", make_tof_clusters_in_layer(1))
    yield ("TofClustersInLayer2", make_tof_clusters_in_layer(2))
    yield ("TofClustersInLayer3", make_tof_clusters_in_layer(3))
    yield ("TofClustersInUpperTof", make_tof_clusters_in_upper_lower_tof('upper'))
    yield ("TofClustersInLowerTof", make_tof_clusters_in_upper_lower_tof('lower'))

    yield ("TofEnergyInLayer0", make_tof_energy_in_layer(0))
    yield ("TofEnergyInLayer1", make_tof_energy_in_layer(1))
    yield ("TofEnergyInLayer2", make_tof_energy_in_layer(2))
    yield ("TofEnergyInLayer3", make_tof_energy_in_layer(3))

    yield ("TofEnergyPerClusterInLayer0", make_tof_energy_per_cluster(0))
    yield ("TofEnergyPerClusterInLayer1", make_tof_energy_per_cluster(1))

    yield ("TofEnergieOneClusterInLayer2", make_tof_energy_in_layer_for_one_cluster_cut(2, default_value=5))

    yield ('TrkTrackBestPairElectronRigidity', make_trk_track_best_pair_electron_positron_rigidity(MC_PARTICLE_IDS['Electron']))
    yield ('TrkTrackBestPairPositronRigidity', make_trk_track_best_pair_electron_positron_rigidity(MC_PARTICLE_IDS['Positron']))

    yield ("TrkTrackBestPairFirstAssociatedTofClusterLayer0YCoordinate", make_associated_tof_cluster_in_layer_coordinate(0, 0))
    yield ("TrkTrackBestPairFirstAssociatedTofClusterLayer1XCoordinate", make_associated_tof_cluster_in_layer_coordinate(0, 1))
    yield ("TrkTrackBestPairFirstAssociatedTofClusterLayer2XCoordinate", make_associated_tof_cluster_in_layer_coordinate(0, 2))
    yield ("TrkTrackBestPairFirstAssociatedTofClusterLayer3YCoordinate", make_associated_tof_cluster_in_layer_coordinate(0, 3))

    yield ("TrkTrackBestPairSecondAssociatedTofClusterLayer0YCoordinate", make_associated_tof_cluster_in_layer_coordinate(1, 0))
    yield ("TrkTrackBestPairSecondAssociatedTofClusterLayer1XCoordinate", make_associated_tof_cluster_in_layer_coordinate(1, 1))
    yield ("TrkTrackBestPairSecondAssociatedTofClusterLayer2XCoordinate", make_associated_tof_cluster_in_layer_coordinate(1, 2))
    yield ("TrkTrackBestPairSecondAssociatedTofClusterLayer3YCoordinate", make_associated_tof_cluster_in_layer_coordinate(1, 3))
    
    yield ("TrkTrackBestPairMinAssociatedTofClusterLayer2XCoordinate", make_min_max_abs('min', "TrkTrackBestPairFirstAssociatedTofClusterLayer2XCoordinate", "TrkTrackBestPairSecondAssociatedTofClusterLayer2XCoordinate"))

    yield ("TrkTrackBestPairDistanceBetweenAssociatedTofClustersLayer0", make_distance_xy("TrkTrackBestPairFirstAssociatedTofClusterLayer0YCoordinate", "TrkTrackBestPairSecondAssociatedTofClusterLayer0YCoordinate"))
    yield ("TrkTrackBestPairDistanceBetweenAssociatedTofClustersLayer1", make_distance_xy("TrkTrackBestPairFirstAssociatedTofClusterLayer1XCoordinate", "TrkTrackBestPairSecondAssociatedTofClusterLayer1XCoordinate"))
    yield ("TrkTrackBestPairDistanceBetweenAssociatedTofClustersLayer2", make_distance_xy("TrkTrackBestPairFirstAssociatedTofClusterLayer2XCoordinate", "TrkTrackBestPairSecondAssociatedTofClusterLayer2XCoordinate"))
    yield ("TrkTrackBestPairDistanceBetweenAssociatedTofClustersLayer3", make_distance_xy("TrkTrackBestPairFirstAssociatedTofClusterLayer3YCoordinate", "TrkTrackBestPairSecondAssociatedTofClusterLayer3YCoordinate"))
    
    yield ("TrkTrackBestPairFirstTrackDistanceToAssociatedTofClusterLayer0", make_distance_xy("TrkTrackBestPairFirstAssociatedTofClusterLayer0YCoordinate", "TrkTrackBestPairFirstCoordinateYAtUpperTof"))
    yield ("TrkTrackBestPairFirstTrackDistanceToAssociatedTofClusterLayer1", make_distance_xy("TrkTrackBestPairFirstAssociatedTofClusterLayer1XCoordinate", "TrkTrackBestPairFirstCoordinateXAtUpperTof"))
    yield ("TrkTrackBestPairFirstTrackDistanceToAssociatedTofClusterLayer2", make_distance_xy("TrkTrackBestPairFirstAssociatedTofClusterLayer2XCoordinate", "TrkTrackBestPairFirstCoordinateXAtLowerTof"))
    yield ("TrkTrackBestPairFirstTrackDistanceToAssociatedTofClusterLayer3", make_distance_xy("TrkTrackBestPairFirstAssociatedTofClusterLayer3YCoordinate", "TrkTrackBestPairFirstCoordinateYAtLowerTof"))

    yield ("TrkTrackBestPairSecondTrackDistanceToAssociatedTofClusterLayer0", make_distance_xy("TrkTrackBestPairSecondAssociatedTofClusterLayer0YCoordinate", "TrkTrackBestPairSecondCoordinateYAtUpperTof"))
    yield ("TrkTrackBestPairSecondTrackDistanceToAssociatedTofClusterLayer1", make_distance_xy("TrkTrackBestPairSecondAssociatedTofClusterLayer1XCoordinate", "TrkTrackBestPairSecondCoordinateXAtUpperTof"))
    yield ("TrkTrackBestPairSecondTrackDistanceToAssociatedTofClusterLayer2", make_distance_xy("TrkTrackBestPairSecondAssociatedTofClusterLayer2XCoordinate", "TrkTrackBestPairSecondCoordinateXAtLowerTof"))
    yield ("TrkTrackBestPairSecondTrackDistanceToAssociatedTofClusterLayer3", make_distance_xy("TrkTrackBestPairSecondAssociatedTofClusterLayer3YCoordinate", "TrkTrackBestPairSecondCoordinateYAtLowerTof"))

    yield ("TrkTrackBestPairMinTrackDistanceToAssociatedTofClusterLayer2", make_min_max('min', "TrkTrackBestPairFirstTrackDistanceToAssociatedTofClusterLayer2", "TrkTrackBestPairSecondTrackDistanceToAssociatedTofClusterLayer2"))
    yield ("TrkTrackBestPairMinTrackDistanceToAssociatedTofClusterLayer3", make_min_max('min', "TrkTrackBestPairFirstTrackDistanceToAssociatedTofClusterLayer3", "TrkTrackBestPairSecondTrackDistanceToAssociatedTofClusterLayer3"))

    for i, tof in enumerate(['Upper', 'Upper', 'Lower', 'Lower']):
        yield (f"TrkTrackBestPairFirstPassTofBarInLayer{i}", make_trk_track_best_pair_pass_tof_bar_in_layer(i, f"TrkTrackBestPairFirstCoordinateXAt{tof}Tof", f"TrkTrackBestPairFirstCoordinateYAt{tof}Tof"))
        yield (f"TrkTrackBestPairSecondPassTofBarInLayer{i}", make_trk_track_best_pair_pass_tof_bar_in_layer(i, f"TrkTrackBestPairFirstCoordinateXAt{tof}Tof", f"TrkTrackBestPairFirstCoordinateYAt{tof}Tof"))

        yield (f"TrkTrackBestPairFirstAssociatedTofClusterBarInLayer{i}", make_trk_track_best_pair_associated_tof_bar_in_layer(0, i))
        yield (f"TrkTrackBestPairSecondAssociatedTofClusterBarInLayer{i}", make_trk_track_best_pair_associated_tof_bar_in_layer(1, i))

        yield (f"TrkTrackBestPairFirstDistanceAssociatedTofBarPassedTofBarLayer{i}", make_trk_track_best_pair_distance_associated_tof_bar_passed_bar_in_layer(f"TrkTrackBestPairFirstAssociatedTofClusterBarInLayer{i}", f"TrkTrackBestPairFirstPassTofBarInLayer{i}"))
        yield (f"TrkTrackBestPairSecondDistanceAssociatedTofBarPassedTofBarLayer{i}", make_trk_track_best_pair_distance_associated_tof_bar_passed_bar_in_layer(f"TrkTrackBestPairSecondAssociatedTofClusterBarInLayer{i}", f"TrkTrackBestPairSecondPassTofBarInLayer{i}"))

        yield (f"TrkTrackBestPairDistanceBetweenMatchingClustersInLayer{i}", make_trk_track_best_pair_distance_associated_tof_bar_passed_bar_in_layer(f"TrkTrackBestPairFirstAssociatedTofClusterBarInLayer{i}", f"TrkTrackBestPairFirstAssociatedTofClusterBarInLayer{i}"))

        yield (f"TrkTrackBestPairHasMatchingTofClusterInLayer{i}", make_trk_track_best_pair_matching_clusters_in_layer(i))
    
    yield ("TrkTrackBestPairMatchingClusterInLayer2InCentralBars", trk_track_best_pair_matching_cluster_in_layer2_in_central_bars) 

    yield ("TrkMaxChargeL1", make_simple_trk_hit_max_charge_in_layer(1))
    
    yield ("TrkTrackBestPairMinDistanceExtrapolatedIsInLowerTRD" ,make_trk_track_best_pair_min_distane_is_in_upper_lower_trd('Lower'))
    yield ("TrkTrackBestPairMinDistanceExtrapolatedIsInUpperTRD" ,make_trk_track_best_pair_min_distane_is_in_upper_lower_trd('Upper'))

    yield ("TrkTrackBestPairEnergyWithBremsstrahlungTotal", make_bremsstrahlung_corrected_energy("TrkTrackBestPairRigidity", "McSecondaryBremsstrahlungEnergyLossTotal"))
    yield ("TrkTrackBestPairEnergyWithBremsstrahlungBeforeL3", make_bremsstrahlung_corrected_energy("TrkTrackBestPairRigidity", "McSecondaryBremsstrahlungEnergyLossBeforeL3"))
    yield ("TrkTrackBestPairEnergyWithBremsstrahlungBeforeL56", make_bremsstrahlung_corrected_energy("TrkTrackBestPairRigidity", "McSecondaryBremsstrahlungEnergyLossBeforeL56"))
    yield ("TrkTrackBestPairEnergyWithBremsstrahlungBeforeL8", make_bremsstrahlung_corrected_energy("TrkTrackBestPairRigidity", "McSecondaryBremsstrahlungEnergyLossBeforeL8"))

    yield ("TrkTrackBestPairBremsstrahlungCorrectedEnergy", make_bremsstrahlung_fit_corrected_energy("TrkTrackBestPairRigidity", 0.139))
    yield ("TrkTrackBestPairCorrectedEnergy", make_linlog_corrected_energy("TrkTrackBestPairBremsstrahlungCorrectedEnergy", 0.00867408, 0.05930206))
