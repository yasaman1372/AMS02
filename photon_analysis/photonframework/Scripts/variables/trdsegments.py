
import numpy as np
import awkward as ak

from tools.binnings import make_int_binning, make_lin_binning, make_log_binning
from tools.variables import depends, returns, annotate_binning as binning, make_sum


def make_n_trd_segments(nhits_branch):
    @returns(np.int16)
    @depends([nhits_branch])
    @binning("n_trd_segments")
    def _n_trd_segments(events):
        return ak.num(events[nhits_branch])
    return _n_trd_segments


def make_top_trd_sublayer_in_one_segment(segment_projection): # for exactly 1 segment
    @returns(np.float32)
    @depends([f"TrdSegments{segment_projection}SublayerPattern", f"NTrdSegments{segment_projection}"])
    @binning(make_int_binning(20))
    def _top_trd_layer_in_segment(events):
        result = np.zeros(len(events))
        mask = events[f"NTrdSegments{segment_projection}"] == 1
        Sublayerpattern = ak.to_numpy(ak.flatten(events[f'TrdSegments{segment_projection}SublayerPattern'][mask]))
        HighestBit = np.floor(np.log2(Sublayerpattern))
        if segment_projection == "XZ":
            HighestBit = HighestBit+8
        if segment_projection == "YZ":
            HighestBit = np.array([l if l <= 7 else l+24 for l in HighestBit])
        result[mask] = HighestBit
        return result
    return _top_trd_layer_in_segment

def make_vertex_one_trd_segment_abs_alpha_y(segment_projection):
    @returns(np.float32)
    @depends([f"TrdSegments{segment_projection}SublayerPattern", f"NTrdSegments{segment_projection}", f"TrdSegments{segment_projection}FitParameters", "TrkTrackBestPairMinDistanceDirPhi", "TrkTrackBestPairMinDistanceDirTheta"])
    @binning("phi")
    def _vertex_one_trd_segment_abs_alpha(events):
        result = np.zeros(len(events))
        mask = events[f"NTrdSegments{segment_projection}"] == 1
        slope = events[f"TrdSegments{segment_projection}FitParameters"][mask][:,0]
        TrdTheta = np.arctan(slope)
        vertexTheta = events.TrkTrackBestPairMinDistanceDirTheta[mask]
        vertexPhi = events.TrkTrackBestPairMinDistanceDirPhi[mask]
        alpha = np.rad2deg(TrdTheta - np.arctan(np.cos(vertexPhi) * np.tan(vertexTheta)))
        result[mask] = np.abs(alpha)
        return result
    return _vertex_one_trd_segment_abs_alpha

def make_NTrdHits_if_zero_xz_yz_segments(segment_projection):
    @returns(np.float32)
    @depends([f"NTrdSegments{segment_projection}", "NTrdHits"])
    @binning(make_int_binning(50))
    def _NTrdHits_if_zero_xz_yz_segments(events):
        result = np.ones(len(events))*-1
        mask = events[f"NTrdSegments{segment_projection}"] == 0
        NTrdHits = events.NTrdHits[mask]
        result[mask] = NTrdHits
        return result
    return _NTrdHits_if_zero_xz_yz_segments



def load_variables(config, workdir, energy_estimator, binnings):
    binnings.register_binning("n_trd_segments", make_int_binning(21))

    yield ("NTrdSegmentsXZ", make_n_trd_segments("TrdSegmentsXZNHits"))
    yield ("NTrdSegmentsYZ", make_n_trd_segments("TrdSegmentsYZNHits"))
    yield ("NTrdSegments", make_sum("NTrdSegmentsXZ", "NTrdSegmentsYZ", np.int16, "n_trd_segments"))
    yield ("HighestTrdSubLayerOneXZSegment", make_top_trd_sublayer_in_one_segment("XZ"))
    yield ("HighestTrdSubLayerOneYZSegment", make_top_trd_sublayer_in_one_segment("YZ"))
    yield ("TrdSegmentVertexAbsAlphaY", make_vertex_one_trd_segment_abs_alpha_y("YZ"))
    yield ("NTrdHitsZeroYZSegments", make_NTrdHits_if_zero_xz_yz_segments('YZ'))


