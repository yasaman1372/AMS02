
import numpy as np
import awkward as ak
import healpy as hp
import os

from tools.binnings import make_int_binning, make_lin_binning, make_log_binning
from tools.constants import TRK_LAYER_POSITION_Z, TRD_Z_LOWER, TRD_Z_UPPER, ACC_Z_LOWER, SKY_REGIONS
from tools.variables import returns, depends, annotate_binning as binning, make_healpy_index, make_dxydz, make_theta, make_phi, make_theta_xy, extrapolate_to_z, make_radius, make_difference, make_relative_difference, make_ratio, make_galactic_latitude_and_longitude, extract_float_axis
from tools.healpy_tools import mask_sources, get_regions
from tools.config import get_config

@returns(np.int16)
@depends(["TrkNTracks", "TrkTrackPairMinDistances", "TrkTrackPairTrackIndices", "TrkTrackElectronRigidities"])
@binning("pair_index")
def trk_track_best_pair_index(events):
    best_pair_index = np.zeros(len(events), dtype=np.int16)
    if len(events) == 0:
        return best_pair_index
    max_pairs = int(np.max(ak.num(events.TrkTrackPairMinDistances)))
    has_pair = np.zeros(len(events), dtype=bool)
    has_opposite_pair = np.zeros(len(events), dtype=bool)
    min_distance = np.ones(len(events), dtype=np.float32)
    min_distances = events.TrkTrackPairMinDistances
    indices = events.TrkTrackPairTrackIndices
    rigidities = events.TrkTrackElectronRigidities
    for pair_index in range(max_pairs):
        mask = pair_index < ak.num(min_distances)
        track_index_1 = indices[mask][:,2 * pair_index]
        track_index_2 = indices[mask][:,2 * pair_index + 1]
        array_indices = np.arange(len(events))[mask]
        rig1 = rigidities[array_indices,track_index_1]
        rig2 = rigidities[array_indices,track_index_2]
        opposite = rig1 * rig2 < 0
        distances = min_distances[mask][:,pair_index]
        better = (~has_pair[mask]) | (~has_opposite_pair[mask] & opposite) | (opposite & (distances < min_distance[mask]))
        #better = opposite & ((~has_pair[mask]) | (distances < min_distance[mask]))
        res_index = np.arange(len(events))
        res_id = res_index[mask][better]
        has_pair[res_id] = True
        has_opposite_pair[res_id] = has_opposite_pair[res_id] | opposite[better]
        min_distance[res_id] = distances[better]
        best_pair_index[res_id] = pair_index
    return best_pair_index

@returns(np.int16)
@depends(['TrkNTracks', 'TrkTrackPairTrackIndices', 'TrkTrackElectronRigidities', 'TrkTrackCoordinatesAtUpperTof', 'TrkTrackDirectionsAtUpperTofXYZ'])
@binning('pair_index')
def trk_track_best_pair_index_bastian(events):
    best_pair_index = np.zeros(len(events), dtype=np.int16)
    if len(events) == 0:
        return best_pair_index
    max_pairs = int(np.max(ak.num(events.TrkTrackPairTrackIndices)/2))
    has_pair = np.zeros(len(events), dtype=bool)
    has_opposite_pair = np.zeros(len(events), dtype=bool)
    min_chi = np.ones(len(events), dtype=np.float32)
    xcoor = events.TrkTrackCoordinatesAtUpperTof[:,0::3]
    ycoor = events.TrkTrackCoordinatesAtUpperTof[:,1::3]
    directionxyz = events.TrkTrackDirectionsAtUpperTofXYZ
    directionx = directionxyz[:,0::3]
    directiony = directionxyz[:,1::3]
    directionz = directionxyz[:,2::3]
    indices = events.TrkTrackPairTrackIndices
    rigidities = events.TrkTrackElectronRigidities
    for pair_index in range(max_pairs):
        mask = pair_index < ak.num(events.TrkTrackPairTrackIndices)/2
        track_index_1 = indices[mask][:,2 * pair_index]
        track_index_2 = indices[mask][:,2 * pair_index + 1]
        array_indices = np.arange(len(events))[mask]
        rig1 = rigidities[array_indices,track_index_1]
        rig2 = rigidities[array_indices,track_index_2]
        deltaX = np.abs(xcoor[array_indices, track_index_1] - xcoor[array_indices, track_index_2])
        deltaY = np.abs(ycoor[array_indices, track_index_1] - ycoor[array_indices, track_index_2])
        directions1 = np.array([directionx[array_indices, track_index_1], directiony[array_indices, track_index_1], directionz[array_indices, track_index_1]]).T
        directions2 = np.array([directionx[array_indices, track_index_2], directiony[array_indices, track_index_2], directionz[array_indices, track_index_2]]).T
        #deltaAlpha = np.diagonal(np.inner(directions1, directions2))
        deltaAlpha = directions1[:,0]*directions2[:,0] + directions1[:,1]*directions2[:,1] + directions1[:,2]*directions2[:,2]
        deltaAlpha = deltaAlpha/(np.linalg.norm(directions1, axis = 1)*np.linalg.norm(directions2, axis = 1))
        deltaAlpha = np.arccos(deltaAlpha)
        Chi2 = (deltaY/0.25)**2 + (deltaX/0.2)**2 + (deltaAlpha/0.02)**2
        opposite = rig1 * rig2 < 0
        better = (~has_pair[mask]) | (~has_opposite_pair[mask] & opposite) | (opposite & (Chi2 < min_chi[mask]))
        res_index = np.arange(len(events))
        res_id = res_index[mask][better]
        has_pair[res_id] = True
        has_opposite_pair[res_id] = has_opposite_pair[mask][better] | opposite[better]
        min_chi[res_id] = Chi2[better]
        best_pair_index[res_id] = pair_index
    return best_pair_index



@returns(bool)
@depends(["TrkNTracks", "TrkTrackPairTrackIndices"])
@binning(bool)
def trk_track_has_best_pair(events):
    has_tracks = events.TrkNTracks >= 2
    has_pair = ak.num(events.TrkTrackPairTrackIndices) >= 2
    return has_tracks & has_pair

@returns(np.float32)
@depends(["TrkTrackBestPairIndex", "TrkTrackPairTrackIndices", "TrkTrackElectronRigidities"])
@binning("rigidity")
def trk_track_best_pair_rigidity_product(events):
    mask = (ak.num(events.TrkTrackElectronRigidities) > 0) & (ak.num(events.TrkTrackPairTrackIndices) > 0)
    result = np.zeros(len(events))
    array_indices = np.arange(len(events))[mask]
    best_pair_index = events.TrkTrackBestPairIndex[mask]
    indices = events.TrkTrackPairTrackIndices
    rigidities = events.TrkTrackElectronRigidities
    id1 = indices[array_indices,2 * best_pair_index]
    id2 = indices[array_indices,2 * best_pair_index + 1]
    rig1 = rigidities[array_indices,id1]
    rig2 = rigidities[array_indices,id2]
    result[mask] = rig1 * rig2
    return result

@returns(np.float32)
@depends(["TrkTrackBestPairIndex", "TrkTrackPairTrackIndices", "TrkTrackElectronRigidities"])
@binning("energy")
def trk_track_best_pair_rigidity(events):
    mask = (ak.num(events.TrkTrackElectronRigidities) > 0) & (ak.num(events.TrkTrackPairTrackIndices) > 0)
    result = np.zeros(len(events))
    array_indices = np.arange(len(events))[mask]
    best_pair_index = events.TrkTrackBestPairIndex[mask]
    indices = events.TrkTrackPairTrackIndices
    rigidities = events.TrkTrackElectronRigidities
    id1 = indices[array_indices,2 * best_pair_index]
    id2 = indices[array_indices,2 * best_pair_index + 1]
    rig1 = rigidities[array_indices,id1]
    rig2 = rigidities[array_indices,id2]
    result[mask] = np.abs(rig1 - rig2)
    return result


@returns(np.float32)
@depends(["TrkTrackPairTrackIndices", "TrkTrackRigidities"])
@binning("energy")
def trk_track_pair_rigidities(events):
    array_indices = np.arange(len(events))
    id1 = events.TrkTrackPairTrackIndices[:,0::2]
    id2 = events.TrkTrackPairTrackIndices[:,1::2]
    rig1 = events.TrkTrackRigidities[id1]
    rig2 = events.TrkTrackRigidities[id2]
    result = np.abs(rig1 - rig2)
    return result

@returns(np.float32)
@depends(["TrkTrackPairTrackIndices", "TrkTrackRigidities"])
@binning("rigidity")
def trk_track_pair_rigidity_products(events):
    array_indices = np.arange(len(events))
    id1 = events.TrkTrackPairTrackIndices[:,0::2]
    id2 = events.TrkTrackPairTrackIndices[:,1::2]
    rig1 = events.TrkTrackRigidities[id1]
    rig2 = events.TrkTrackRigidities[id2]
    result = rig1 * rig2
    return result

@returns(np.float32)
@depends(["TrkTrackPairTrackIndices", "TrkTrackRigidities"])
@binning("invariant_mass")
def trk_track_pair_invariant_masses(events):
    array_indices = np.arange(len(events))
    id1 = events.TrkTrackPairTrackIndices[:,0::2]
    id2 = events.TrkTrackPairTrackIndices[:,1::2]
    rig1 = np.abs(events.TrkTrackRigidities[id1])
    rig2 = np.abs(events.TrkTrackRigidities[id2])
    angle = events.TrkTrackPairMinDistanceAngles
    return np.sqrt(2 * rig1 * rig2 * (1 - np.cos(angle)))

def make_trk_track_pair_property(property_name, offset, dtype=np.float32, binning_hint=None):
    @returns(dtype)
    @depends(["TrkTrackPairTrackIndices", property_name])
    @binning(binning_hint)
    def _trk_track_pair_property(events):
        indices = events.TrkTrackPairTrackIndices[:,offset::2]
        return events[property_name][indices]
    return _trk_track_pair_property


@returns(np.float32)
@depends(["TrkTrackBestPairIndex", "TrkTrackPairMinDistances"])
@binning(make_log_binning(1e-5, 100, 100))
def trk_track_best_pair_min_distance(events):
    mask = ak.num(events.TrkTrackPairMinDistances) > 0
    result = np.ones(len(events)) * 1000
    best_pair_index = events.TrkTrackBestPairIndex[mask]
    array_indices = np.arange(len(events))[mask]
    result[mask] = events.TrkTrackPairMinDistances[array_indices,best_pair_index]
    return result

def make_trk_track_best_pair_min_distance_coordinate(axis):
    @returns(np.float32)
    @depends(["TrkTrackBestPairIndex", "TrkTrackPairMinDistanceCoordinates"])
    @binning("coordinate")
    def _trk_track_best_pair_min_distance_coordinate(events):
        mask = ak.num(events.TrkTrackPairMinDistanceCoordinates) > 0
        result = np.ones(len(events)) * 1000
        best_pair_index = events.TrkTrackBestPairIndex[mask]
        array_indices = np.arange(len(events))[mask]
        result[mask] = events.TrkTrackPairMinDistanceCoordinates[array_indices,3 * best_pair_index + axis]
        return result
    return _trk_track_best_pair_min_distance_coordinate

def make_trk_track_best_pair_min_distance_direction(direction_branch, axis, n_axes):
    @returns(np.float32)
    @depends(["TrkTrackBestPairIndex", direction_branch])
    @binning("direction")
    def _trk_track_best_pair_min_distance_direction(events):
        mask = ak.num(events[direction_branch]) > 0
        result = np.ones(len(events)) * 1000
        best_pair_index = events.TrkTrackBestPairIndex[mask]
        array_indices = np.arange(len(events))[mask]
        result[mask] = events[direction_branch][array_indices,n_axes * best_pair_index + axis]
        return result
    return _trk_track_best_pair_min_distance_direction

def make_trk_track_best_pair_track_property(property_branch, track_in_pair_index, axis=0, axes=1, default_value=1000, dtype=np.float32, binning_hint=None):
    @returns(dtype)
    @depends(["TrkTrackBestPairIndex", "TrkTrackPairTrackIndices", "TrkTrackPairMinDistances", property_branch])
    @binning(binning_hint)
    def _trk_track_best_pair_track_property(events):
        mask = ak.num(events.TrkTrackPairMinDistances) > 0
        result = np.ones(len(events), dtype=dtype) * default_value
        array_indices = np.arange(len(events))[mask]
        track_index = events.TrkTrackPairTrackIndices[array_indices,events.TrkTrackBestPairIndex[mask] * 2 + track_in_pair_index]
        result[mask] = events[property_branch][array_indices,(track_index * axes) + axis]
        return result
    return _trk_track_best_pair_track_property


@returns(np.float32)
@depends(["TrkTrackBestPairIndex", "TrkTrackBestPairFirstRigidity", "TrkTrackBestPairSecondRigidity", "TrkTrackPairMinDistanceAngles"])
@binning("invariant_mass")
def trk_track_best_pair_invariant_mass(events):
    result = np.zeros(len(events))
    mask = ak.num(events.TrkTrackPairMinDistanceAngles) > 0
    p1 = np.abs(events.TrkTrackBestPairFirstRigidity[mask])
    p2 = np.abs(events.TrkTrackBestPairSecondRigidity[mask])
    array_indices = np.arange(len(events))
    angle = events.TrkTrackPairMinDistanceAngles[array_indices[mask], events.TrkTrackBestPairIndex[mask]]
    result[mask] = np.sqrt(2 * p1 * p2 * (1 - np.cos(angle)))
    return result


@returns(np.float32)
@depends(["TrkTrackBestPairIndex", "TrkTrackPairMinDistanceAngles"])
@binning("angle")
def trk_track_best_pair_angle(events):
    mask = ak.num(events.TrkTrackPairMinDistanceAngles) > 0
    array_indices = np.arange(len(events))
    result = -100 * np.ones(len(events), dtype=np.float32)
    result[mask] = events.TrkTrackPairMinDistanceAngles[array_indices[mask], events.TrkTrackBestPairIndex[mask]]
    return result

@returns(np.float32)
@depends(["TrkTrackBestPairMinDistanceCoordZ"])
@binning("coordinate")
def trk_track_best_pair_min_distance_coord_distance_to_trk_layer(events):
    coord_z = events.TrkTrackBestPairMinDistanceCoordZ[:,None]
    ref_coords = np.array(TRK_LAYER_POSITION_Z[1:8])[None,:]
    distances = np.abs(coord_z - ref_coords)
    return np.min(distances, axis=1)

def group_trk_clusters(orientation):
    length_branch = f"TrkClusterLengths{orientation}"
    amplitude_branch = f"TrkClusterAmplitudes{orientation}"
    @returns(np.float32)
    @depends([length_branch, amplitude_branch])
    @binning(None)
    def _group_trk_clusters(events):
        cluster_lengths = ak.flatten(events[length_branch])
        cluster_start_indices = np.concatenate(([0], np.cumsum(cluster_lengths)))
        all_clusters = clusters = ak.contents.ListOffsetArray(ak.index.Index(cluster_start_indices), ak.contents.NumpyArray(ak.flatten(events[amplitude_branch])))
        event_start_indices = np.concatenate(([0], np.cumsum(ak.num(events[length_branch]))))
        grouped_clusters = ak.Array(ak.contents.ListOffsetArray(ak.index.Index(event_start_indices), all_clusters))
        return grouped_clusters
    return _group_trk_clusters

def make_trk_has_hit_in_layer(layer_pattern_branch, layer_index):
    @returns(bool)
    @depends([layer_pattern_branch])
    @binning(bool)
    def _trk_has_hit_in_layer(events):
        mask = 0b1 << layer_index
        return (events[layer_pattern_branch] & mask) > 0
    return _trk_has_hit_in_layer

def make_rigidity_ratio(first_branch, second_branch):
    @returns(np.float32)
    @depends([first_branch, second_branch])
    @binning(make_log_binning(1, 100, 100))
    def _rigidity_ratio(events):
        r1 = np.abs(events[first_branch])
        r2 = np.abs(events[second_branch])
        result = np.ones(len(events))
        result[r1 > r2] = (r1 / r2)[r1 > r2]
        result[r2 > r1] = (r2 / r1)[r2 > r1]
        return result
    return _rigidity_ratio

@returns(bool)
@depends(["TofClustersInLayer"])
@binning(bool)
def tof_clusters_in_all_layers(events):
    return ((events.TofClustersInLayer[:,0] > 0)
        & (events.TofClustersInLayer[:,1] > 0)
        & (events.TofClustersInLayer[:,2] > 0)
        & (events.TofClustersInLayer[:,3] > 0))


def make_closest_tof_cluster_to_coordinate(layer, tof_axis, trk_coordinate):
    @returns(np.int16)
    @depends(["TofClusterCoordinates", "TofClusterLayers", trk_coordinate])
    @binning("tof_cluster_index")
    def _closest_tof_cluster_to_coordinate(events):
        if len(events) == 0:
            return np.empty((0,), dtype=np.float32)
        trk_coord = events[trk_coordinate]
        tof_cluster_coordinates = events.TofClusterCoordinates[:,tof_axis::3]
        tof_cluster_indices = np.zeros(len(events), dtype=np.int16)
        min_distances = np.ones(len(events)) * 1000
        layers = events.TofClusterLayers
        for cluster_index in range(np.max(ak.num(events.TofClusterLayers))):
            mask = ak.num(layers) > cluster_index
            sel = (layers[mask,cluster_index] == layer)
            distances = np.abs(events[trk_coordinate][mask] - tof_cluster_coordinates[mask,cluster_index])
            best = sel & (distances < min_distances[mask])
            tof_cluster_indices[mask][best] = cluster_index
            min_distances[mask][best] = distances[best]
        return tof_cluster_indices
    return _closest_tof_cluster_to_coordinate

def make_closest_tof_cluster_distance(tof_axis, cluster_index_branch, trk_coordinate_branch):
    @returns(np.float32)
    @depends([cluster_index_branch, trk_coordinate_branch, "TofClusterCoordinates"])
    @binning("coordinate")
    def _closest_tof_cluster_distance(events):
        if len(events) == 0:
            return np.empty((0,), dtype=np.float32)
        cluster_indices = events[cluster_index_branch]
        mask = ak.num(events.TofClusterCoordinates) > 3 * cluster_indices
        result = np.ones(len(events)) * 1000
        array_indices = np.arange(len(events))[mask]
        tof_coordinates = events.TofClusterCoordinates[array_indices,3 * cluster_indices[mask] + tof_axis]
        trk_coordinates = events[trk_coordinate_branch][mask]
        distance = tof_coordinates - trk_coordinates
        result[mask] = distance
        return result
    return _closest_tof_cluster_distance
    
def make_closest_tof_cluster_energy(cluster_index_branch):
    @returns(np.float32)
    @depends([cluster_index_branch, "TofClusterEnergies"])
    @binning("deposited_energy")
    def _closest_tof_cluster_energy(events):
        mask = ak.num(events.TofClusterEnergies) > 0
        array_indices = np.arange(len(events))[mask]
        result = np.zeros(len(events))
        cluster_indices = events[cluster_index_branch][mask]
        tof_energies = events.TofClusterEnergies[array_indices,cluster_indices]
        result[mask] = tof_energies
        return result
    return _closest_tof_cluster_energy

def make_trk_track_best_pair_min_distance_points_to_region(nside, region, mask_type: bool = True, inclusiv: bool = False):
    @returns(bool)
    @depends(["TrkTrackBestPairMinDistanceDirGalacticLatitude", "TrkTrackBestPairMinDistanceDirGalacticLongitude"])
    @binning(bool)
    def _trk_track_best_pair_min_distance_points_to_region(events):
        npix = hp.nside2npix(nside)
        Map = np.arange(npix)
        InRegionID = mask_sources(Map=Map, region=region, mask_type=mask_type, return_indices=True)
        GaLat = events.TrkTrackBestPairMinDistanceDirGalacticLatitude
        GaLon = events.TrkTrackBestPairMinDistanceDirGalacticLongitude
        healpy_Index = make_healpy_index("TrkTrackBestPairMinDistanceDirGalacticLatitude", "TrkTrackBestPairMinDistanceDirGalacticLongitude", nside=nside)(events)
        result = np.array([not mask_type if idx in InRegionID else mask_type for idx in healpy_Index])
        return result
    return _trk_track_best_pair_min_distance_points_to_region

def make_trk_track_best_pair_min_distance_points_to_region_from_config(config, region_type, mask_type, latitude_branch, longitude_branch, nside=None):
    return make_trk_track_best_pair_min_distance_points_to_region_from_file(config, config["regions"][region_type], mask_type, latitude_branch, longitude_branch, nside=nside)

def make_trk_track_best_pair_min_distance_points_to_region_from_file(config, region_filename, mask_type, latitude_branch, longitude_branch, nside=None):
    region_path = os.path.join(os.environ["PHOTONFRAMEWORK"], "data", "Regions", region_filename)
    regions = get_regions(region_path)
    if nside is None:
        nside = config['analysis']['nside']
    in_region_ids = mask_sources(Map=np.arange(hp.nside2npix(nside)), region=regions, mask_type=not mask_type, inclusive=not mask_type, return_indices=True)

    healpy_index_function = make_healpy_index(latitude_branch, longitude_branch, nside=nside)

    @returns(bool)
    @depends([latitude_branch, longitude_branch])
    @binning(bool)
    def _trk_track_best_pair_min_distance_points_to_region(events):
        result = np.isin(healpy_index_function(events), in_region_ids)
        if not mask_type:
            result = np.invert(result)
        return result
    return _trk_track_best_pair_min_distance_points_to_region



def load_variables(config, workdir, energy_estimator, binnings):
    binnings.register_binning("pair_index", make_int_binning(64))
    binnings.register_binning("tof_cluster_index", make_int_binning(20))
    binnings.register_binning("trk_n_hits", make_int_binning(10))
    binnings.register_binning("trk_layer_pattern", make_int_binning(512))
    binnings.register_binning("chisq", make_log_binning(1e-3, 1e3, 100))
    binnings.register_binning("invariant_mass", make_log_binning(1e-5, 1e3, 200))
    binnings.register_binning("angle", make_lin_binning(0, np.pi / 2, 100))
    binnings.register_binning("deposited_energy", make_log_binning(1e-3, 1e2, 100))

    yield ("TrkTrackPairRigidities", trk_track_pair_rigidities)
    yield ("TrkTrackPairRigidityProducts", trk_track_pair_rigidity_products)
    yield ("TrkTrackPairInvariantMasses", trk_track_pair_invariant_masses)
    
    yield ("TrkTrackPairFirstTrackHits", make_trk_track_pair_property("TrkTrackHits", 0, dtype=np.int16))
    yield ("TrkTrackPairSecondTrackHits", make_trk_track_pair_property("TrkTrackHits", 1, dtype=np.int16))
    
    #yield ("TrkTrackBestPairIndex", trk_track_best_pair_index)
    yield ("TrkTrackBestPairIndex", trk_track_best_pair_index_bastian)
    yield ("TrkTrackHasBestPair", trk_track_has_best_pair)
    
    yield ("TrkTrackBestPairRigidityProduct", trk_track_best_pair_rigidity_product)
    yield ("TrkTrackBestPairRigidity", trk_track_best_pair_rigidity)
    yield ("TrkTrackBestPairInvariantMass", trk_track_best_pair_invariant_mass)

    yield ("TrkTrackBestPairMinDistance", trk_track_best_pair_min_distance)
    yield ("TrkTrackBestPairMinDistanceCoordX", make_trk_track_best_pair_min_distance_coordinate(0))
    yield ("TrkTrackBestPairMinDistanceCoordY", make_trk_track_best_pair_min_distance_coordinate(1))
    yield ("TrkTrackBestPairMinDistanceCoordZ", make_trk_track_best_pair_min_distance_coordinate(2))
    yield ("TrkTrackBestPairMinDistanceDirX", make_trk_track_best_pair_min_distance_direction("TrkTrackPairMinDistanceDirectionsXYZ", 0, 3))
    yield ("TrkTrackBestPairMinDistanceDirY", make_trk_track_best_pair_min_distance_direction("TrkTrackPairMinDistanceDirectionsXYZ", 1, 3))
    yield ("TrkTrackBestPairMinDistanceDirZ", make_trk_track_best_pair_min_distance_direction("TrkTrackPairMinDistanceDirectionsXYZ", 2, 3))
    yield ("TrkTrackBestPairMinDistanceDirXZ", make_dxydz("TrkTrackBestPairMinDistanceDirX", "TrkTrackBestPairMinDistanceDirZ"))
    yield ("TrkTrackBestPairMinDistanceDirYZ", make_dxydz("TrkTrackBestPairMinDistanceDirY", "TrkTrackBestPairMinDistanceDirZ"))
    yield ("TrkTrackBestPairMinDistanceDirTheta", make_theta("TrkTrackBestPairMinDistanceDirZ"))
    yield ("TrkTrackBestPairMinDistanceDirPhi", make_phi("TrkTrackBestPairMinDistanceDirX", "TrkTrackBestPairMinDistanceDirY"))
    yield ("TrkTrackBestPairMinDistanceDirThetaX", make_theta_xy("TrkTrackBestPairMinDistanceDirX", "TrkTrackBestPairMinDistanceDirZ"))
    yield ("TrkTrackBestPairMinDistanceDirThetaY", make_theta_xy("TrkTrackBestPairMinDistanceDirY", "TrkTrackBestPairMinDistanceDirZ"))
    
    yield ("TrkTrackBestPairMinDistanceDirectionsInGalacticCoordinates", make_galactic_latitude_and_longitude("TrkTrackBestPairMinDistanceDirX", "TrkTrackBestPairMinDistanceDirY", "TrkTrackBestPairMinDistanceDirZ", "UTCTime"))
    yield ("TrkTrackBestPairMinDistanceDirGalacticLatitude", extract_float_axis("TrkTrackBestPairMinDistanceDirectionsInGalacticCoordinates", 0, "latitude"))
    yield ("TrkTrackBestPairMinDistanceDirGalacticLongitude", extract_float_axis("TrkTrackBestPairMinDistanceDirectionsInGalacticCoordinates", 1, "longitude"))
   
    yield ("TrkTrackBestPairMinDistanceExtrapolatedXAtL1", extrapolate_to_z("TrkTrackBestPairMinDistanceCoordX", "TrkTrackBestPairMinDistanceDirXZ", "TrkTrackBestPairMinDistanceCoordZ", TRK_LAYER_POSITION_Z[0]))
    yield ("TrkTrackBestPairMinDistanceExtrapolatedYAtL1", extrapolate_to_z("TrkTrackBestPairMinDistanceCoordY", "TrkTrackBestPairMinDistanceDirYZ", "TrkTrackBestPairMinDistanceCoordZ", TRK_LAYER_POSITION_Z[0]))
    yield ("TrkTrackBestPairMinDistanceExtrapolatedXAtLowerTRD", extrapolate_to_z("TrkTrackBestPairMinDistanceCoordX", "TrkTrackBestPairMinDistanceDirXZ", "TrkTrackBestPairMinDistanceCoordZ", TRD_Z_LOWER))
    yield ("TrkTrackBestPairMinDistanceExtrapolatedYAtLowerTRD", extrapolate_to_z("TrkTrackBestPairMinDistanceCoordY", "TrkTrackBestPairMinDistanceDirYZ", "TrkTrackBestPairMinDistanceCoordZ", TRD_Z_LOWER))
    yield ("TrkTrackBestPairMinDistanceExtrapolatedXAtUpperTRD", extrapolate_to_z("TrkTrackBestPairMinDistanceCoordX", "TrkTrackBestPairMinDistanceDirXZ", "TrkTrackBestPairMinDistanceCoordZ", TRD_Z_UPPER))
    yield ("TrkTrackBestPairMinDistanceExtrapolatedYAtUpperTRD", extrapolate_to_z("TrkTrackBestPairMinDistanceCoordY", "TrkTrackBestPairMinDistanceDirYZ", "TrkTrackBestPairMinDistanceCoordZ", TRD_Z_UPPER))
    yield ("TrkTrackBestPairMinDistanceCoordDistanceToTrkLayer", trk_track_best_pair_min_distance_coord_distance_to_trk_layer)
    yield ("TrkTrackBestPairMinDistanceExtrapolatedRAtLowerTRD", make_radius("TrkTrackBestPairMinDistanceExtrapolatedXAtLowerTRD", "TrkTrackBestPairMinDistanceExtrapolatedYAtLowerTRD"))
    yield ("TrkTrackBestPairMinDistanceExtrapolatedRAtUpperTRD", make_radius("TrkTrackBestPairMinDistanceExtrapolatedXAtUpperTRD", "TrkTrackBestPairMinDistanceExtrapolatedYAtUpperTRD"))
    
    yield ("TrkTrackBestPairMinDistanceDirGalacticHealpyIndex128", make_healpy_index("TrkTrackBestPairMinDistanceDirGalacticLatitude", "TrkTrackBestPairMinDistanceDirGalacticLongitude", nside=128))
    yield ("TrkTrackBestPairMinDistanceDirEquatorialHealpyIndex128", make_healpy_index("TrkTrackBestPairMinDistanceDirEquatorialDeclination", "TrkTrackBestPairMinDistanceDirEquatorialRightAscension", nside=128))
    yield ("TrkTrackBestPairMinDistanceDirGalacticHealpyIndex256", make_healpy_index("TrkTrackBestPairMinDistanceDirGalacticLatitude", "TrkTrackBestPairMinDistanceDirGalacticLongitude", nside=256))
    yield ("TrkTrackBestPairMinDistanceDirEquatorialHealpyIndex256", make_healpy_index("TrkTrackBestPairMinDistanceDirEquatorialDeclination", "TrkTrackBestPairMinDistanceDirEquatorialRightAscension", nside=256))
       
    yield ("TrkTrackBestPairMinDistanceAngle", trk_track_best_pair_angle)

    yield ("TrkTrackBestPairFirstTrackHits", make_trk_track_best_pair_track_property("TrkTrackHits", 0, default_value=0, dtype=np.int16, binning_hint="trk_n_hits"))
    yield ("TrkTrackBestPairSecondTrackHits", make_trk_track_best_pair_track_property("TrkTrackHits", 1, default_value=0, dtype=np.int16, binning_hint="trk_n_hits"))
    yield ("TrkTrackBestPairFirstTrackLayerPattern", make_trk_track_best_pair_track_property("TrkTrackLayerPattern", 0, default_value=0, dtype=np.int16, binning_hint="trk_layer_pattern"))
    yield ("TrkTrackBestPairSecondTrackLayerPattern", make_trk_track_best_pair_track_property("TrkTrackLayerPattern", 1, default_value=0, dtype=np.int16, binning_hint="trk_layer_pattern"))
    yield ("TrkTrackBestPairFirstRigidity", make_trk_track_best_pair_track_property("TrkTrackRigidities", 0, default_value=-10000, binning_hint="rigidity"))
    yield ("TrkTrackBestPairSecondRigidity", make_trk_track_best_pair_track_property("TrkTrackRigidities", 1, default_value=-10000, binning_hint="rigidity"))
    yield ("TrkTrackBestPairFirstElectronRigidity", make_trk_track_best_pair_track_property("TrkTrackElectronRigidities", 0, default_value=-10000, binning_hint="rigidity"))
    yield ("TrkTrackBestPairSecondElectronRigidity", make_trk_track_best_pair_track_property("TrkTrackElectronRigidities", 1, default_value=-10000, binning_hint="rigidity"))
    
    yield ("TrkTrackBestPairFirstChiSquareY", make_trk_track_best_pair_track_property("TrkTrackRigidityChiSquaresY", 0, default_value=1000, binning_hint="chisq"))
    yield ("TrkTrackBestPairSecondChiSquareY", make_trk_track_best_pair_track_property("TrkTrackRigidityChiSquaresY", 1, default_value=1000, binning_hint="chisq"))
    yield ("TrkTrackBestPairFirstElectronChiSquareY", make_trk_track_best_pair_track_property("TrkTrackElectronRigidityChiSquaresY", 0, default_value=1000, binning_hint="chisq"))
    yield ("TrkTrackBestPairSecondElectronChiSquareY", make_trk_track_best_pair_track_property("TrkTrackElectronRigidityChiSquaresY", 1, default_value=1000, binning_hint="chisq"))
    yield ("TrkTrackBestPairFirstChiSquareX", make_trk_track_best_pair_track_property("TrkTrackRigidityChiSquaresX", 0, default_value=1000, binning_hint="chisq"))
    yield ("TrkTrackBestPairSecondChiSquareX", make_trk_track_best_pair_track_property("TrkTrackRigidityChiSquaresX", 1, default_value=1000, binning_hint="chisq"))
    yield ("TrkTrackBestPairFirstElectronChiSquareX", make_trk_track_best_pair_track_property("TrkTrackElectronRigidityChiSquaresX", 0, default_value=1000, binning_hint="chisq"))
    yield ("TrkTrackBestPairSecondElectronChiSquareX", make_trk_track_best_pair_track_property("TrkTrackElectronRigidityChiSquaresX", 1, default_value=1000, binning_hint="chisq"))
    
    yield ("TrkTrackBestPairFirstTrackerCharge", make_trk_track_best_pair_track_property('TrkTrackCharges', 0, binning_hint='charge'))
    yield ("TrkTrackBestPairSecondTrackerCharge", make_trk_track_best_pair_track_property('TrkTrackCharges', 1, binning_hint='charge'))

    yield ("TrkTrackBestPairFirstAssociatedTofBeta", make_trk_track_best_pair_track_property('TrkTrackAssociatedTofBetas', 0, binning_hint="beta", default_value=0))
    yield ("TrkTrackBestPairSecondAssociatedTofBeta", make_trk_track_best_pair_track_property('TrkTrackAssociatedTofBetas', 1, binning_hint="beta", default_value=0))

    yield ("TrkTrackBestPairFirstCoordinateXAtUpperTof", make_trk_track_best_pair_track_property("TrkTrackCoordinatesAtUpperTof", 0, axis=0, axes=3, default_value=-200, binning_hint="coordinate"))
    yield ("TrkTrackBestPairFirstCoordinateYAtUpperTof", make_trk_track_best_pair_track_property("TrkTrackCoordinatesAtUpperTof", 0, axis=1, axes=3, default_value=-200, binning_hint="coordinate"))
    yield ("TrkTrackBestPairFirstCoordinateZAtUpperTof", make_trk_track_best_pair_track_property("TrkTrackCoordinatesAtUpperTof", 0, axis=2, axes=3, default_value=-200, binning_hint="coordinate"))
    yield ("TrkTrackBestPairSecondCoordinateXAtUpperTof", make_trk_track_best_pair_track_property("TrkTrackCoordinatesAtUpperTof", 1, axis=0, axes=3, default_value=-200, binning_hint="coordinate"))
    yield ("TrkTrackBestPairSecondCoordinateYAtUpperTof", make_trk_track_best_pair_track_property("TrkTrackCoordinatesAtUpperTof", 1, axis=1, axes=3, default_value=-200, binning_hint="coordinate"))
    yield ("TrkTrackBestPairSecondCoordinateZAtUpperTof", make_trk_track_best_pair_track_property("TrkTrackCoordinatesAtUpperTof", 1, axis=2, axes=3, default_value=-200, binning_hint="coordinate"))

    yield ("TrkTrackBestFirstDirectionsAtUpperTofX", make_trk_track_best_pair_track_property("TrkTrackDirectionsAtUpperTofXYZ", 0, axis = 0, axes = 3, default_value=1000, binning_hint="direction"))
    yield ("TrkTrackBestFirstDirectionsAtUpperTofY", make_trk_track_best_pair_track_property("TrkTrackDirectionsAtUpperTofXYZ", 0, axis = 1, axes = 3, default_value=1000, binning_hint="direction"))
    yield ("TrkTrackBestFirstDirectionsAtUpperTofZ", make_trk_track_best_pair_track_property("TrkTrackDirectionsAtUpperTofXYZ", 0, axis = 2, axes = 3, default_value=1000, binning_hint="direction"))
    yield ("TrkTrackBestSecondDirectionsAtUpperTofX", make_trk_track_best_pair_track_property("TrkTrackDirectionsAtUpperTofXYZ", 1, axis = 0, axes = 3, default_value=1000, binning_hint="direction"))
    yield ("TrkTrackBestSecondDirectionsAtUpperTofY", make_trk_track_best_pair_track_property("TrkTrackDirectionsAtUpperTofXYZ", 1, axis = 1, axes = 3, default_value=1000, binning_hint="direction"))
    yield ("TrkTrackBestSecondDirectionsAtUpperTofZ", make_trk_track_best_pair_track_property("TrkTrackDirectionsAtUpperTofXYZ", 1, axis = 2, axes = 3, default_value=1000, binning_hint="direction"))

    yield ("TrkTrackBestPairFirstDirectionAtUpperTofXZ", make_dxydz("TrkTrackBestFirstDirectionsAtUpperTofX", "TrkTrackBestFirstDirectionsAtUpperTofZ"))
    yield ("TrkTrackBestPairFirstDirectionAtUpperTofYZ", make_dxydz("TrkTrackBestFirstDirectionsAtUpperTofY", "TrkTrackBestFirstDirectionsAtUpperTofZ"))
    yield ("TrkTrackBestPairSecondDirectionAtUpperTofXZ", make_dxydz("TrkTrackBestSecondDirectionsAtUpperTofX", "TrkTrackBestSecondDirectionsAtUpperTofZ"))
    yield ("TrkTrackBestPairSecondDirectionAtUpperTofYZ", make_dxydz("TrkTrackBestSecondDirectionsAtUpperTofY", "TrkTrackBestSecondDirectionsAtUpperTofZ"))

    yield ("TrkTrackBestPairFirstExtrapolatedXAtUpperTRD", extrapolate_to_z("TrkTrackBestPairFirstCoordinateXAtUpperTof", "TrkTrackBestPairFirstDirectionAtUpperTofXZ", "TrkTrackBestPairFirstCoordinateZAtUpperTof", TRD_Z_UPPER))
    yield ("TrkTrackBestPairFirstExtrapolatedYAtUpperTRD", extrapolate_to_z("TrkTrackBestPairFirstCoordinateYAtUpperTof", "TrkTrackBestPairFirstDirectionAtUpperTofYZ", "TrkTrackBestPairFirstCoordinateZAtUpperTof", TRD_Z_UPPER))
    yield ("TrkTrackBestPairSecondExtrapolatedXAtUpperTRD", extrapolate_to_z("TrkTrackBestPairSecondCoordinateXAtUpperTof", "TrkTrackBestPairSecondDirectionAtUpperTofXZ", "TrkTrackBestPairSecondCoordinateZAtUpperTof", TRD_Z_UPPER))
    yield ("TrkTrackBestPairSecondExtrapolatedYAtUpperTRD", extrapolate_to_z("TrkTrackBestPairSecondCoordinateYAtUpperTof", "TrkTrackBestPairSecondDirectionAtUpperTofYZ", "TrkTrackBestPairSecondCoordinateZAtUpperTof", TRD_Z_UPPER))

    yield ("TrkTrackBestPairFirstCoordinateXAtLowerTof", make_trk_track_best_pair_track_property("TrkTrackCoordinatesAtLowerTof", 0, axis=0, axes=3, default_value=-200, binning_hint="coordinate"))
    yield ("TrkTrackBestPairFirstCoordinateYAtLowerTof", make_trk_track_best_pair_track_property("TrkTrackCoordinatesAtLowerTof", 0, axis=1, axes=3, default_value=-200, binning_hint="coordinate"))
    yield ("TrkTrackBestPairFirstCoordinateZAtLowerTof", make_trk_track_best_pair_track_property("TrkTrackCoordinatesAtLowerTof", 0, axis=2, axes=3, default_value=-200, binning_hint="coordinate"))
    yield ("TrkTrackBestPairSecondCoordinateXAtLowerTof", make_trk_track_best_pair_track_property("TrkTrackCoordinatesAtLowerTof", 1, axis=0, axes=3, default_value=-200, binning_hint="coordinate"))
    yield ("TrkTrackBestPairSecondCoordinateYAtLowerTof", make_trk_track_best_pair_track_property("TrkTrackCoordinatesAtLowerTof", 1, axis=1, axes=3, default_value=-200, binning_hint="coordinate"))
    yield ("TrkTrackBestPairSecondCoordinateZAtLowerTof", make_trk_track_best_pair_track_property("TrkTrackCoordinatesAtLowerTof", 1, axis=2, axes=3, default_value=-200, binning_hint="coordinate"))

    yield ("TrkTrackBestFirstDirectionsAtLowerTofX", make_trk_track_best_pair_track_property("TrkTrackDirectionsAtLowerTofXYZ", 0, axis = 0, axes = 3, default_value=1000, binning_hint="direction"))
    yield ("TrkTrackBestFirstDirectionsAtLowerTofY", make_trk_track_best_pair_track_property("TrkTrackDirectionsAtLowerTofXYZ", 0, axis = 1, axes = 3, default_value=1000, binning_hint="direction"))
    yield ("TrkTrackBestFirstDirectionsAtLowerTofZ", make_trk_track_best_pair_track_property("TrkTrackDirectionsAtLowerTofXYZ", 0, axis = 2, axes = 3, default_value=1000, binning_hint="direction"))
    yield ("TrkTrackBestSecondDirectionsAtLowerTofX", make_trk_track_best_pair_track_property("TrkTrackDirectionsAtLowerTofXYZ", 1, axis = 0, axes = 3, default_value=1000, binning_hint="direction"))
    yield ("TrkTrackBestSecondDirectionsAtLowerTofY", make_trk_track_best_pair_track_property("TrkTrackDirectionsAtLowerTofXYZ", 1, axis = 1, axes = 3, default_value=1000, binning_hint="direction"))
    yield ("TrkTrackBestSecondDirectionsAtLowerTofZ", make_trk_track_best_pair_track_property("TrkTrackDirectionsAtLowerTofXYZ", 1, axis = 2, axes = 3, default_value=1000, binning_hint="direction"))

    yield ("TrkTrackBestPairFirstDirectionAtLowerTofXZ", make_dxydz("TrkTrackBestFirstDirectionsAtLowerTofX", "TrkTrackBestFirstDirectionsAtLowerTofZ"))
    yield ("TrkTrackBestPairFirstDirectionAtLowerTofYZ", make_dxydz("TrkTrackBestFirstDirectionsAtLowerTofY", "TrkTrackBestFirstDirectionsAtLowerTofZ"))
    yield ("TrkTrackBestPairSecondDirectionAtLowerTofXZ", make_dxydz("TrkTrackBestSecondDirectionsAtLowerTofX", "TrkTrackBestSecondDirectionsAtLowerTofZ"))
    yield ("TrkTrackBestPairSecondDirectionAtLowerTofYZ", make_dxydz("TrkTrackBestSecondDirectionsAtLowerTofY", "TrkTrackBestSecondDirectionsAtLowerTofZ"))

    yield ("TrkTrackBestPairFirstExtrapolatedXAtLowerTRD", extrapolate_to_z("TrkTrackBestPairFirstCoordinateXAtUpperTof", "TrkTrackBestPairFirstDirectionAtUpperTofXZ", "TrkTrackBestPairFirstCoordinateZAtUpperTof", TRD_Z_LOWER))
    yield ("TrkTrackBestPairFirstExtrapolatedYAtLowerTRD", extrapolate_to_z("TrkTrackBestPairFirstCoordinateYAtUpperTof", "TrkTrackBestPairFirstDirectionAtUpperTofYZ", "TrkTrackBestPairFirstCoordinateZAtUpperTof", TRD_Z_LOWER))
    yield ("TrkTrackBestPairSecondExtrapolatedXAtLowerTRD", extrapolate_to_z("TrkTrackBestPairSecondCoordinateXAtUpperTof", "TrkTrackBestPairSecondDirectionAtUpperTofXZ", "TrkTrackBestPairSecondCoordinateZAtUpperTof", TRD_Z_LOWER))
    yield ("TrkTrackBestPairSecondExtrapolatedYAtLowerTRD", extrapolate_to_z("TrkTrackBestPairSecondCoordinateYAtUpperTof", "TrkTrackBestPairSecondDirectionAtUpperTofYZ", "TrkTrackBestPairSecondCoordinateZAtUpperTof", TRD_Z_LOWER))

    yield ("TrkTrackBestPairFirstExtrapolatedXAtLowerACC", extrapolate_to_z("TrkTrackBestPairFirstCoordinateXAtLowerTof", "TrkTrackBestPairFirstDirectionAtLowerTofXZ", "TrkTrackBestPairFirstCoordinateZAtLowerTof", ACC_Z_LOWER))
    yield ("TrkTrackBestPairSecondExtrapolatedXAtLowerACC", extrapolate_to_z("TrkTrackBestPairSecondCoordinateXAtLowerTof", "TrkTrackBestPairSecondDirectionAtLowerTofXZ", "TrkTrackBestPairSecondCoordinateZAtLowerTof", ACC_Z_LOWER))
    yield ("TrkTrackBestPairFirstExtrapolatedYAtLowerACC", extrapolate_to_z("TrkTrackBestPairFirstCoordinateYAtLowerTof", "TrkTrackBestPairFirstDirectionAtLowerTofYZ", "TrkTrackBestPairFirstCoordinateZAtLowerTof", ACC_Z_LOWER))
    yield ("TrkTrackBestPairSecondExtrapolatedYAtLowerACC", extrapolate_to_z("TrkTrackBestPairSecondCoordinateYAtLowerTof", "TrkTrackBestPairSecondDirectionAtLowerTofYZ", "TrkTrackBestPairSecondCoordinateZAtLowerTof", ACC_Z_LOWER))
    
    yield ("TrkTrackBestPairFirstExtrapolatedRAtLowerACC", make_radius("TrkTrackBestPairFirstExtrapolatedXAtLowerACC", "TrkTrackBestPairFirstExtrapolatedYAtLowerACC"))
    yield ("TrkTrackBestPairSecondExtrapolatedRAtLowerACC", make_radius("TrkTrackBestPairSecondExtrapolatedXAtLowerACC", "TrkTrackBestPairSecondExtrapolatedYAtLowerACC"))

    yield ("TrkTrackBestPairFirstCoordinateRAtUpperTof", make_radius("TrkTrackBestPairFirstCoordinateXAtUpperTof", "TrkTrackBestPairFirstCoordinateYAtUpperTof"))
    yield ("TrkTrackBestPairSecondCoordinateRAtUpperTof", make_radius("TrkTrackBestPairSecondCoordinateXAtUpperTof", "TrkTrackBestPairSecondCoordinateYAtUpperTof"))
    
    yield ("TrkTrackBestPairFirstHasHitInL1", make_trk_has_hit_in_layer("TrkTrackBestPairFirstTrackLayerPattern", 1))
    yield ("TrkTrackBestPairSecondHasHitInL1", make_trk_has_hit_in_layer("TrkTrackBestPairSecondTrackLayerPattern", 1))
    yield ("TrkTrackBestPairRigidityRatio", make_rigidity_ratio("TrkTrackBestPairFirstRigidity", "TrkTrackBestPairSecondRigidity"))
    yield ("TrkTrackBestPairElectronRigidityRatio", make_rigidity_ratio("TrkTrackBestPairFirstElectronRigidity", "TrkTrackBestPairSecondElectronRigidity"))

    yield ("TofClustersInAllLayers", tof_clusters_in_all_layers)
    for index, index_name in ((0, "First"), (1, "Second")):
        yield (f"TrkTrackBestPair{index_name}ClosestUTofXClusterIndex", make_closest_tof_cluster_to_coordinate(0, 0, f"TrkTrackBestPair{index_name}CoordinateXAtUpperTof"))
        yield (f"TrkTrackBestPair{index_name}ClosestUTofYClusterIndex", make_closest_tof_cluster_to_coordinate(1, 1, f"TrkTrackBestPair{index_name}CoordinateYAtUpperTof"))
        yield (f"TrkTrackBestPair{index_name}ClosestLTofYClusterIndex", make_closest_tof_cluster_to_coordinate(2, 1, f"TrkTrackBestPair{index_name}CoordinateYAtLowerTof"))
        yield (f"TrkTrackBestPair{index_name}ClosestLTofXClusterIndex", make_closest_tof_cluster_to_coordinate(3, 0, f"TrkTrackBestPair{index_name}CoordinateXAtLowerTof"))
        yield (f"TrkTrackBestPair{index_name}ClosestUTofXDistance", make_closest_tof_cluster_distance(0, f"TrkTrackBestPair{index_name}ClosestUTofXClusterIndex", f"TrkTrackBestPair{index_name}CoordinateXAtUpperTof"))
        yield (f"TrkTrackBestPair{index_name}ClosestUTofYDistance", make_closest_tof_cluster_distance(1, f"TrkTrackBestPair{index_name}ClosestUTofYClusterIndex", f"TrkTrackBestPair{index_name}CoordinateYAtUpperTof"))
        yield (f"TrkTrackBestPair{index_name}ClosestLTofYDistance", make_closest_tof_cluster_distance(1, f"TrkTrackBestPair{index_name}ClosestLTofYClusterIndex", f"TrkTrackBestPair{index_name}CoordinateYAtLowerTof"))
        yield (f"TrkTrackBestPair{index_name}ClosestLTofXDistance", make_closest_tof_cluster_distance(0, f"TrkTrackBestPair{index_name}ClosestLTofXClusterIndex", f"TrkTrackBestPair{index_name}CoordinateXAtLowerTof"))
        yield (f"TrkTrackBestPair{index_name}ClosestUTofXEnergy", make_closest_tof_cluster_energy(f"TrkTrackBestPair{index_name}ClosestUTofXClusterIndex"))
        yield (f"TrkTrackBestPair{index_name}ClosestUTofYEnergy", make_closest_tof_cluster_energy(f"TrkTrackBestPair{index_name}ClosestUTofYClusterIndex"))
        yield (f"TrkTrackBestPair{index_name}ClosestLTofYEnergy", make_closest_tof_cluster_energy(f"TrkTrackBestPair{index_name}ClosestLTofYClusterIndex"))
        yield (f"TrkTrackBestPair{index_name}ClosestLTofXEnergy", make_closest_tof_cluster_energy(f"TrkTrackBestPair{index_name}ClosestLTofXClusterIndex"))

    yield ("TrkTrackBestPairRigidityMinusMcMomentum", make_difference("TrkTrackBestPairRigidity", "McMomentum"))
    yield ("TrkTrackBestPairRigidityMinusMcMomentumPerMcMomentum", make_relative_difference("TrkTrackBestPairRigidity", "McMomentum"))
    yield ("TrkTrackBestPairRigidityPerMcMomentum", make_ratio("TrkTrackBestPairRigidity", "McMomentum"))
    yield ("TrkTrackBestPairPositionResolutionX", make_difference("TrkTrackBestPairMinDistanceCoordX", "McHighestMomentumSecondaryCoordX"))
    yield ("TrkTrackBestPairPositionResolutionY", make_difference("TrkTrackBestPairMinDistanceCoordY", "McHighestMomentumSecondaryCoordY"))
    yield ("TrkTrackBestPairPositionResolutionZ", make_difference("TrkTrackBestPairMinDistanceCoordZ", "McHighestMomentumSecondaryCoordZ"))
    yield ("TrkTrackBestPairMinDistanceCoordR", make_radius("TrkTrackBestPairMinDistanceCoordX", "TrkTrackBestPairMinDistanceCoordY"))


    # Cluster shape
    yield ("TrkClustersX", group_trk_clusters("X"))
    yield ("TrkClustersY", group_trk_clusters("Y"))

    yield ("TrkTrackBestPairMinDistanceDirectionPointsToBackgroundRegion", make_trk_track_best_pair_min_distance_points_to_region_from_config(config, 'background', False, "TrkTrackBestPairMinDistanceDirGalacticLatitude", "TrkTrackBestPairMinDistanceDirGalacticLongitude"))
    yield ("TrkTrackBestPairMinDistanceDirectionPointsToSignalRegion", make_trk_track_best_pair_min_distance_points_to_region_from_config(config, 'signal', True, "TrkTrackBestPairMinDistanceDirGalacticLatitude", "TrkTrackBestPairMinDistanceDirGalacticLongitude"))
    yield ("TrkTrackBestPairMinDistanceDirectionPointsToSourceBlazar3C4543", make_trk_track_best_pair_min_distance_points_to_region_from_file(config, "sources/blazar_3c_454_3.reg", True, "TrkTrackBestPairMinDistanceDirGalacticLatitude", "TrkTrackBestPairMinDistanceDirGalacticLongitude"))
    yield ("TrkTrackBestPairMinDistanceDirectionPointsToSourceBlazarB32322396", make_trk_track_best_pair_min_distance_points_to_region_from_file(config, "sources/blazar_b3_2322_396.reg", True, "TrkTrackBestPairMinDistanceDirGalacticLatitude", "TrkTrackBestPairMinDistanceDirGalacticLongitude"))
    yield ("TrkTrackBestPairMinDistanceDirectionPointsToSourceBlazarBLLac", make_trk_track_best_pair_min_distance_points_to_region_from_file(config, "sources/blazar_bl_lac.reg", True, "TrkTrackBestPairMinDistanceDirGalacticLatitude", "TrkTrackBestPairMinDistanceDirGalacticLongitude"))
    yield ("TrkTrackBestPairMinDistanceDirectionPointsToSourceCrab", make_trk_track_best_pair_min_distance_points_to_region_from_file(config, "sources/crab.reg", True, "TrkTrackBestPairMinDistanceDirGalacticLatitude", "TrkTrackBestPairMinDistanceDirGalacticLongitude"))
    yield ("TrkTrackBestPairMinDistanceDirectionPointsToSourceCTA102", make_trk_track_best_pair_min_distance_points_to_region_from_file(config, "sources/cta_102.reg", True, "TrkTrackBestPairMinDistanceDirGalacticLatitude", "TrkTrackBestPairMinDistanceDirGalacticLongitude"))
    yield ("TrkTrackBestPairMinDistanceDirectionPointsToSourceLSI61303", make_trk_track_best_pair_min_distance_points_to_region_from_file(config, "sources/lsi_61_303.reg", True, "TrkTrackBestPairMinDistanceDirGalacticLatitude", "TrkTrackBestPairMinDistanceDirGalacticLongitude"))
    yield ("TrkTrackBestPairMinDistanceDirectionPointsToSourceNGC1275", make_trk_track_best_pair_min_distance_points_to_region_from_file(config, "sources/ngc_1275.reg", True, "TrkTrackBestPairMinDistanceDirGalacticLatitude", "TrkTrackBestPairMinDistanceDirGalacticLongitude"))
    yield ("TrkTrackBestPairMinDistanceDirectionPointsToSourceGeminga", make_trk_track_best_pair_min_distance_points_to_region_from_file(config, "sources/psr_geminga.reg", True, "TrkTrackBestPairMinDistanceDirGalacticLatitude", "TrkTrackBestPairMinDistanceDirGalacticLongitude"))
    yield ("TrkTrackBestPairMinDistanceDirectionPointsToSourceVela", make_trk_track_best_pair_min_distance_points_to_region_from_file(config, "sources/psr_vela.reg", True, "TrkTrackBestPairMinDistanceDirGalacticLatitude", "TrkTrackBestPairMinDistanceDirGalacticLongitude"))
    yield ("TrkTrackBestPairMinDistanceDirectionPointsToSourcePSRJ00077303", make_trk_track_best_pair_min_distance_points_to_region_from_file(config, "sources/psr_j0007_7303.reg", True, "TrkTrackBestPairMinDistanceDirGalacticLatitude", "TrkTrackBestPairMinDistanceDirGalacticLongitude"))
    yield ("TrkTrackBestPairMinDistanceDirectionPointsToSourcePSRJ17094429", make_trk_track_best_pair_min_distance_points_to_region_from_file(config, "sources/psr_j1709_4429.reg", True, "TrkTrackBestPairMinDistanceDirGalacticLatitude", "TrkTrackBestPairMinDistanceDirGalacticLongitude"))
    yield ("TrkTrackBestPairMinDistanceDirectionPointsToSourcePSRJ18365925", make_trk_track_best_pair_min_distance_points_to_region_from_file(config, "sources/psr_j1836_5925.reg", True, "TrkTrackBestPairMinDistanceDirGalacticLatitude", "TrkTrackBestPairMinDistanceDirGalacticLongitude"))
    yield ("TrkTrackBestPairMinDistanceDirectionPointsToSourcePSRJ22296114", make_trk_track_best_pair_min_distance_points_to_region_from_file(config, "sources/psr_j2229_6114.reg", True, "TrkTrackBestPairMinDistanceDirGalacticLatitude", "TrkTrackBestPairMinDistanceDirGalacticLongitude"))
    yield ("TrkTrackBestPairMinDistanceDirectionPointsToSourceQuasar3C279", make_trk_track_best_pair_min_distance_points_to_region_from_file(config, "sources/quasar_3c_279.reg", True, "TrkTrackBestPairMinDistanceDirGalacticLatitude", "TrkTrackBestPairMinDistanceDirGalacticLongitude"))
    yield ("TrkTrackBestPairMinDistanceDirectionPointsToSourceSNRIC443", make_trk_track_best_pair_min_distance_points_to_region_from_file(config, "sources/snr_ic_443.reg", True, "TrkTrackBestPairMinDistanceDirGalacticLatitude", "TrkTrackBestPairMinDistanceDirGalacticLongitude"))
