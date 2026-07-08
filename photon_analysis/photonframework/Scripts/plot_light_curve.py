#!/usr/bin/env python3

from datetime import datetime
import os
import multiprocessing as mp

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import awkward as ak
import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u
import time
import matplotlib.dates as mdates
from datetime import datetime

from scipy.stats import binomtest
from scipy.special import erfcinv
from scipy.stats import beta
from glob import glob

from tools.config import get_config
from tools.constants import BASTIAN_SKYMAP_PIXEL_AREA, DATA_TIME_RANGES, SKY_REGIONS, SOURCES, TRUE_SOURCES
from tools.binnings import make_day_binning, make_healpy_binning, make_energy_binning_from_config, make_lin_binning
from tools.histograms import load_histogram, Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d, load_histograms_from_files, plot_2d

from tools.roottree import read_tree
from tools.utilities import save_figure
from tools.healpy_tools import mask_sources
from tools.constants import BASTIAN_SKYMAP_PIXEL_AREA, DATA_TIME_RANGES, SKY_REGIONS, SOURCES, TRUE_SOURCES
from tools.coordinates import convert_equatorial_to_galactic_coordinates
from datetime import datetime, timezone

from plot_skymap import plot_skymap, apply_smoothing, no_normalization, normalize_per_square_degree, normalize_per_steradiant, normalize_per_bastian_pixel, NORMALIZATIONS

from calculate_point_spread_function import resolution_parametrization

BRANCHES = ["GalacticLongitude", "GalacticLatitude", "RunNumber", "Energy", "Time"]


def parse_datetime(datetime_str):
    if datetime_str.isnumeric():
        return float(datetime_str)
    return datetime.fromisoformat(datetime_str).replace(tzinfo=timezone.utc)



def rebin_nonzero_exposure(values, edges, max_gap_days=3):
    """
    Rebin exposure data:
    - Group non-zero exposure clusters with small inter-bin gaps
    - Group zero-exposure gaps as separate bins (value = 0)

    Parameters:
        values (array): Exposure values for each bin (shape: N)
        edges (array): Time bin edges (shape: N+1)
        max_gap_days (float): Max allowed time gap (in days) between grouped non-zero bins

    Returns:
        rebinned_values (array): Average exposure values (0 for gap bins)
        rebinned_edges (array): Reconstructed bin edges
    """
    values = np.asarray(values)
    edges = np.asarray(edges)
    max_gap_sec = max_gap_days * 86400

    rebinned_values = []
    rebinned_edges = []

    i = 0
    while i < len(values):
        if values[i] > 0:
            # Start of a non-zero group
            start_edge = edges[i]
            group_vals = [values[i]]
            i += 1
            while i < len(values) and values[i] > 0 and (edges[i] - edges[i - 1]) <= max_gap_sec:
                group_vals.append(values[i])
                i += 1
            end_edge = edges[i] if i < len(edges) else edges[-1]
            rebinned_values.append(np.sum(group_vals))
            rebinned_edges.append(start_edge)
            rebinned_edges.append(end_edge)
        else:
            # Start of a zero-exposure gap group
            start_edge = edges[i]
            i += 1
            while i < len(values) and values[i] == 0:
                i += 1
            end_edge = edges[i] if i < len(edges) else edges[-1]
            rebinned_values.append(0.0)
            rebinned_edges.append(start_edge)
            rebinned_edges.append(end_edge)

    # Collapse edge pairs into continuous edge array
    cleaned_edges = [rebinned_edges[0]]
    for j in range(1, len(rebinned_edges), 2):
        cleaned_edges.append(rebinned_edges[j])

    return np.array(rebinned_values), np.array(cleaned_edges)




def load_altitude(arg):
    filename, effective_area_2d, nside, outputprefix, plotdir, kwargs = arg 
    energy_binning, cos_theta_binning = effective_area_2d.binnings
    healpy_index = kwargs["healpy_bin_index"]
    time_range = kwargs["time_range"]

    with np.load(filename) as np_file:
        start_time = np_file["start_time"].item()
        end_time = np_file["end_time"].item()

        if start_time >= 1e12 or end_time <= 0:
            return None
        if time_range is not None:
            min_time, max_time = map(lambda t: t.timestamp(), time_range)
            if start_time > max_time or end_time < min_time:
                return None

        altitude_time_hist = load_histogram(np_file, "altitude_time")
        exposure = (altitude_time_hist.values[healpy_index,None,:] * effective_area_2d.values[:,:]).sum(axis=1)
        return (start_time, end_time, exposure)


def compute_li_ma_significance(N_on, N_off, alpha):
    N_on = np.asarray(N_on, dtype=float)
    N_off = np.asarray(N_off, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    significance = np.zeros_like(N_on)

    small_alpha_mask = alpha < 1e-4
    significance[small_alpha_mask] = np.sqrt(N_on[small_alpha_mask])

    valid_mask = ~small_alpha_mask
    N_on = N_on[valid_mask]
    N_off = N_off[valid_mask]
    alpha = alpha[valid_mask]

    excess = N_on - alpha * N_off

    # Arrays for adjusted values
    n1 = np.where(excess > 0, N_on, N_off)
    n2 = np.where(excess > 0, N_off, N_on)
    a = np.where(excess > 0, alpha, 1.0 / alpha)
    sign = np.where(excess > 0, 1.0, -1.0)

    # Special case: n2 == 0 or n1 == 0
    zero_n2 = n2 == 0
    zero_n1 = n1 == 0
    general_case = ~(zero_n2 | zero_n1)

    sig = np.zeros_like(n1)

    # Case: n2 == 0
    sig[zero_n2] = np.sqrt(2 * n1[zero_n2] * np.log((1 + a[zero_n2]) / a[zero_n2]))

    # Case: n1 == 0
    sig[zero_n1] = np.sqrt(2 * n2[zero_n1] * np.log(1 + a[zero_n1]))

    # General case (standard formula)
    nt = n1 + n2
    pa = 1 + a
    t1 = n1 * np.log((pa / a) * (n1 / nt))
    t2 = n2 * np.log(pa * (n2 / nt))
    sig[general_case] = np.sqrt(2 * np.abs(t1[general_case] + t2[general_case]))

    final_significance = np.zeros_like(significance)
    final_significance[valid_mask] = sign * sig

    return final_significance



def generate_offset_positions_icrs(source_eq, offset_deg, n_offsets=6):
    """
    Returns a list of offset SkyCoord objects in ICRS (RA/Dec),
    symmetric in RA around the source.
    """
    offsets = []
    for i in range(1, n_offsets // 2 + 1):
        for direction in [-1, 1]:
            ra_offset = (source_eq.ra + direction * i * offset_deg * u.deg) 
            offsets.append(SkyCoord(ra=ra_offset, dec=source_eq.dec, frame='icrs'))
    return offsets



def compute_offset_position(source_position, offset_deg):
    """
    Computes the offset position in Galactic coordinates by shifting the source
    in Right Ascension (equatorial) and converting back.

    Parameters:
        source_position: tuple (lon, lat) in Galactic degrees
        offset_deg     : offset distance in RA (deg)

    Returns:
        offset_position: tuple (lon, lat) in Galactic degrees
    """
    lon_src, lat_src = source_position

    source_gal = SkyCoord(l=lon_src * u.deg, b=lat_src * u.deg, frame="galactic")
    source_eq = source_gal.transform_to("icrs")

    ra_off = (source_eq.ra + offset_deg * u.deg) % (360 * u.deg)
    dec_off = source_eq.dec

    offset_eq = SkyCoord(ra=ra_off, dec=dec_off, frame="icrs")
    offset_gal = offset_eq.transform_to("galactic")

    return offset_gal.l.deg, offset_gal.b.deg



def filter_events_in_window(events, centeral_position, window_radius):
    """
    Filters events within a circular window (angular radius in degrees)
    around a given Galactic source position (lon, lat).
    Returns events within that circular window.
    """
    lon, lat = centeral_position  # Galactic coordinates in degrees

    center_coord = SkyCoord(lon, lat, unit="deg", frame="galactic")
    event_coords = SkyCoord(
        events["GalacticLongitude"],
        events["GalacticLatitude"],
        unit="deg",
        frame="galactic"
    )

    # Compute angular separation and apply mask
    separation = center_coord.separation(event_coords).deg
    mask = separation <= window_radius

    return events[mask]

def parse_reg_file(file_path):
    """
    Reads a .reg file and extracts source names with their equatorial coordinates (RA, Dec).
    Returns a dictionary {source_name: (RA, Dec)}.
    """
    sources = {}
    with open(file_path, "r") as file:
        for line in file:
            if "fk5;point" in line:
                parts = line.split("# text={")
                if len(parts) < 2:
                    continue
                coords_part, name_part = parts[0], parts[1].strip("}\n")
                ra_dec = coords_part.split("(")[1].split(")")[0].split(",")
                ra, dec = float(ra_dec[0]), float(ra_dec[1])
                sources[name_part] = (ra, dec)
    return sources


def filter_sources_by_type(sources, target_type="Blazar"):
    """
    Filters the source list to include only those matching a specific type.
    Uses keyword matching in source names.
    """
    keywords = {
        "Blazar": ["PKS", "TXS", "B3", "PMN", "GB6", "FBQS", "2MASX", "CTA"],
        "Pulsar": ["PSR"],
        "Magnetar": ["SGR"]} #other source names can be added later
    selected_sources = {
        name: coords
        for name, coords in sources.items()
        if any(keyword in name for keyword in keywords.get(target_type, []))
    }
    return selected_sources



def filter_events_in_window(events, source_position, window_size):
    """
    Filters events within the rectangular observation window.
    Returns:
    - Masked events that fall within the window.
    """
    lon, lat = source_position  # Galactic (lon, lat) in degrees
    half_window = window_size / 2

    # Compute window boundaries
    lat_min, lat_max = lat - half_window, lat + half_window
    lon_min, lon_max = lon - half_window, lon + half_window
    
    
    # Filter events inside the window
    mask = (events["GalacticLatitude"] >= lat_min) & (events["GalacticLatitude"] <= lat_max) & \
           (events["GalacticLongitude"] >= lon_min) & (events["GalacticLongitude"] <= lon_max)

    return events[mask]  # Return only the selected events


def handle_file(arg):
    filenames, treename, chunk_size, rank, nranks, kwargs = arg

    # Get source position and window size
    selected_sources = kwargs["selected_sources"]
    min_energy = kwargs["min_energy"]
    max_energy = kwargs["max_energy"]
    min_time = kwargs["min_time"]
    max_time = kwargs["max_time"]
    time_bins = kwargs["time_bins"]
    healpy_binning = kwargs["healpy_binning"]
    energy_binning = kwargs["energy_binning"]
    window = kwargs["window_radius"]

    light_curve = Histogram(time_bins)
    bkg_curve = Histogram(time_bins)
    sky_histogram_2d = Histogram(energy_binning, healpy_binning)
    sky_histogram_2d_in_window = Histogram(energy_binning, healpy_binning) 
    all_events_in_window = []
    all_signal_energies = []
    
    # Get source name and position from kwargs
    source_name = kwargs["source_name"]
    gal_l, gal_b = kwargs["signal_position"]

    print(f"\nProcessing source: {source_name} (l={gal_l}, b={gal_b})", flush=True)

    # Convert source position from Galactic to Equatorial (ICRS)
    source_gal = SkyCoord(l=gal_l * u.deg, b=gal_b * u.deg, frame='galactic')
    source_eq = source_gal.transform_to('icrs')
    offset_positions = kwargs["background_positions"]  # already in (l, b) galactic degrees
    
    sky_histogram_2d_list = {}
    sky_histogram_2d_in_window_list = {}
    all_events_in_window_list = {}


    selected_sources_limited = dict(list(selected_sources.items())[:6]) # here we can limit the number of sources if we want
   
    # Read events from ROOT file
    print("Starting to read events from ROOT files...",flush=True)
    for events in read_tree(filenames, treename, rank=rank, nranks=nranks, chunk_size=chunk_size, branches=BRANCHES, cache_file=False):
        
        if len(events) == 0:   # Check if events is empty before processing
            continue
        
        print(f"Processing a new batch of events... (Chunk size: {len(events)})")    
        
        if kwargs["min_energy"] is not None and kwargs["max_energy"] is not None:
            events = events[(events.Energy >= kwargs["min_energy"]) & (events.Energy <= kwargs["max_energy"])]
              
        # Convert to Healpix index
        healpy_index = hp.ang2pix(nside=kwargs["nside"],theta=events.GalacticLongitude,phi=events.GalacticLatitude,lonlat=True)
                                        
        # Fill per-source histogram
        sky_histogram_2d.fill(events.Energy, healpy_index)
        
        # Filter by time range if specified
        if kwargs["min_time"] is not None and kwargs["max_time"] is not None:
            events = events[(events.Time >= kwargs["min_time"]) & (events.Time <= kwargs["max_time"])]
            print(f"[handle_file] Events after time filtering: {len(events)}", flush=True)

        
        all_offset_events = []
        for offset_position in offset_positions:  # offset_positions is a list of (l, b) tuples
             events_offset = filter_events_in_window(events, offset_position, window)
             all_offset_events.append(events_offset)
        
        
        if len(all_offset_events) == 0:
            print(f"[WARN] No events found in any offset window!")
        else:
            all_offset_events = ak.Array({branch: ak.concatenate([a[branch] for a in all_offset_events]) for branch in BRANCHES})
            # Fill background curve with time stamps
            bkg_curve.fill(all_offset_events["Time"])


        # Filter events inside the window
        events_in_window = filter_events_in_window(events, (gal_l, gal_b), window)
        light_curve.fill(events_in_window["Time"])
        if len(events_in_window) > 0:
            all_signal_energies.append(events_in_window["Energy"])
        
        
        # Convert to Healpix index
        healpy_index_in_window = hp.ang2pix(nside=kwargs["nside"], theta=events_in_window.GalacticLongitude, phi=events_in_window.GalacticLatitude, lonlat=True)
        # Fill per-source histogram
        sky_histogram_2d_in_window.fill(events_in_window.Energy, healpy_index_in_window)

    if len(all_signal_energies) > 0:
        all_energies_flat = ak.flatten(ak.Array(all_signal_energies))
        median_energy = np.median(all_energies_flat)
    else:
        median_energy = np.nan

    # Return the reduced array, histograms, and background estimate
    return light_curve, sky_histogram_2d, sky_histogram_2d_in_window, bkg_curve, median_energy
      

def make_event_args(filename, treename, chunk_size, nranks, **kwargs):
    for rank in range(nranks):
        yield (filename, treename, chunk_size, rank, nranks, kwargs)



def make_exposure_args(filenames, effective_area, nside, outputprefix, plotdir, **kwargs):
    for filename in filenames:
        yield (filename, effective_area, nside, outputprefix, plotdir, kwargs)


def main():

    # Argument parser setup
    import argparse
    parser = argparse.ArgumentParser(description="Plot light curve from photon events.")
    parser.add_argument("--config", nargs=2, required=True, help="Path to config file and working directory.")
    parser.add_argument("--tree", nargs="+", required=True, help="ROOT files to read.")
    parser.add_argument("--treename", default="GammaTree", help="Name of the tree in the ROOT files.")
    parser.add_argument("--source-name", type=str, required=True)
    parser.add_argument("--lon", type=float, required=True, help="Galactic longitude of the source")
    parser.add_argument("--lat", type=float, required=True, help="Galactic latitude of the source")
    parser.add_argument("--offset-degree", type=float, required=True, help="RA offset between signal and background positions (deg)")
    parser.add_argument("--n-offset", type=int, required=True, help="Number of offset background positions (even number)")
    parser.add_argument("--window-radius", type=float, help="Radius (deg) of signal and background circular windows")
    parser.add_argument("--altitude-histograms", required=True, nargs="+", help="Path to NPZ files containing the altitude histograms.")
    parser.add_argument("--effective-area", required=True, help="Path to NPZ file containing the effective area to be integrated.")
    parser.add_argument("--nside", type=int, default=256, help="HEALPix nside resolution")
    parser.add_argument("--time-range", type=parse_datetime, nargs=2, required=True, help="Time range to read and plot.")
    parser.add_argument("--energy-min", type=float, help="Minimum energy [GeV] for exposure integration.")
    parser.add_argument("--energy-max", type=float, help="Maximum energy [GeV] for exposure integration.")
    parser.add_argument("--outputprefix", default="LightCurve", help="Prefix for plots and result files.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--title", default="Gamma Ray Sky Map", help="Title for plots.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Number of events to read in parallel.")
    parser.add_argument("--psf", help="Path to a file containing a parametrized point spread function. If passed, the skymap will be smoothed with it.") 

    args = parser.parse_args()

    config_filename, workdir = args.config
    config = get_config(config_filename)
    
    os.makedirs(args.resultdir, exist_ok=True)
    os.makedirs(args.plotdir, exist_ok=True)
    
    # Convert source position to SkyCoord
    source_gal = SkyCoord(l=args.lon * u.deg, b=args.lat * u.deg, frame="galactic")
    source_eq = source_gal.transform_to("icrs")
    signal_position = (args.lon,args.lat)
    
    # Generate offset background positions in ICRS and convert to Galactic
    offsets_eq = generate_offset_positions_icrs(source_eq, args.offset_degree, args.n_offset)
    offsets_gal = [coord.transform_to("galactic") for coord in offsets_eq]
    background_positions = [(coord.l.deg, coord.b.deg) for coord in offsets_gal]
    time_binning = make_day_binning(*args.time_range)

    healpy_binning = make_healpy_binning(args.nside)
    energy_binning = make_energy_binning_from_config(config)

    nside = args.nside
    source_name = args.source_name
    #window_radius= args.window_radius


    with np.load(args.psf) as psf_file:
       resolution_parameters = psf_file["resolution_parameters_y"]
   
    if args.window_radius:
        window_radius= args.window_radius
    else:    
        window_radius = resolution_parametrization(2.84,*resolution_parameters)
        


    # Store parameters in kwargs
    kwargs = {
        "nside": nside,
        "source_name": source_name,
        "signal_position": signal_position,
        "background_positions": background_positions,
        "window_radius": window_radius,
        "min_energy": args.energy_min,
        "max_energy": args.energy_max,
        "min_time": args.time_range[0].timestamp(),
        "max_time": args.time_range[1].timestamp(),
        "time_bins": time_binning,
        "energy_binning": energy_binning,
        "healpy_binning": healpy_binning,
    }

    # Create directories if they don't exist
    os.makedirs(args.plotdir, exist_ok=True)
    os.makedirs(args.resultdir, exist_ok=True)
    
    source_light_curve = None
    sky_histogram_2d = None
    sky_histogram_2d_in_window = None
    source_bkg_curve = None
    median_energies = []

    print("Starting multiprocessing pool for event processing...", flush=True)
    
    with mp.Pool(args.nprocesses) as pool:
        pool_args = make_event_args(
            args.tree,
            args.treename,
            chunk_size=args.chunk_size,
            nranks=args.nprocesses,
            **kwargs
        )
    
        for result in pool.imap_unordered(handle_file, pool_args):
            light_curve, hist_2d, hist_2d_in_window, bkg_curve, median_en = result
    
           
            if not np.isnan(median_en):
                median_energies.append(median_en)

            if source_light_curve is None:
                source_light_curve = light_curve
            else:
                source_light_curve.add(light_curve)
            
            if source_bkg_curve is None:
                source_bkg_curve = bkg_curve
            else:
                source_bkg_curve.add(bkg_curve)
            
            
            if sky_histogram_2d is None:
                sky_histogram_2d = hist_2d
            else:
                sky_histogram_2d.add(hist_2d)
    
            if sky_histogram_2d_in_window is None:
                sky_histogram_2d_in_window = hist_2d_in_window
            else:
                sky_histogram_2d_in_window.add(hist_2d_in_window)
    
        
        if median_energies:
            median_energy = np.median(median_energies)
            print(f"✅ Final global median energy: {median_energy:.2f} GeV")
        else:
            raise ValueError("No valid median energies found from any chunk.")
        
    
        with np.load(args.effective_area) as effective_area_file:
            effective_area_3d = load_histogram(effective_area_file, "effective_area_3d")
            effective_area_2d = load_histogram(effective_area_file, "effective_area_2d")

        ex_energy_binning, _ = effective_area_2d.binnings
        all_positions = [("signal", signal_position)] + [(f"background_{i+1}", pos) for i, pos in enumerate(background_positions)]
        exposures={}
        
        print("all_positions: ", all_positions, flush=True)


        for label, pos in all_positions:
            
            print(f"\n?~_~S~M Calculating exposure for {label} at {pos}")
            exposure_hist = WeightedHistogram(ex_energy_binning, time_binning, labels=("Energy / GeV", "Time"))
           # WeightedHistogram(time_binning, labels=("Time",))

            healpy_bin_index = hp.ang2pix(args.nside, *pos, lonlat=True)
            print(f"Looking for Healpy bin {healpy_bin_index}")

            
            print(f"Using {len(args.altitude_histograms)} altitude histogram patterns.")
            altitude_histograms = [filename for pattern in args.altitude_histograms for filename in glob(pattern)]

            pool_args = make_exposure_args(altitude_histograms, effective_area_2d, args.nside, args.outputprefix, args.plotdir, time_range=args.time_range, healpy_bin_index=healpy_bin_index)
            progress = 0 
            total = len(altitude_histograms)
            print(total, flush = True)
            for return_value in pool.imap_unordered(load_altitude, pool_args):
                progress += 1
                if return_value is not None:
                    start_time, end_time, exposure_per_energy = return_value
                    mean_time = (start_time + end_time) / 2 
                    exposure_hist.fill(ex_energy_binning.bin_centers, np.full_like(exposure_per_energy, mean_time), weights=exposure_per_energy)         
                print(f"{progress:>8}/{total:>8}", end="\r", flush=True)
            exposures[label] = exposure_hist         
   

    energy_bin_index = np.searchsorted(ex_energy_binning.edges, median_energy, side="right") - 1
    exposure_1d_sig = exposures["signal"].project(energy_bin_index)
    exposure_1d_bkgs = [exposures[key].project(energy_bin_index) for key in exposures if key.startswith("background_")]    
    
    print(f"processing source {source_name}", flush=True)
   
    # Save light curve histogram for each source
    lightcurve_output = os.path.join(args.resultdir, f"{args.outputprefix}_{source_name}_lightcurve.npz")
    np.savez_compressed(lightcurve_output, light_curve=source_light_curve, time_bins=time_binning)
    print(f"Saved light curve histogram for {source_name} to {lightcurve_output}", flush=True)
    
    
    # Photon counts per bin
    N_on = source_light_curve.values[1:-1]
    N_off = source_bkg_curve.values[1:-1]
    
    signal_exposure = exposure_1d_sig.values[1:-1]
    background_exposures = [exposure_1d.values[1:-1] for exposure_1d in exposure_1d_bkgs]
    background_exposures_array = np.stack(background_exposures)
    background_exposure_sum = np.sum(background_exposures_array, axis=0)
    
    sig_time_edges = source_light_curve.binnings[0].edges[1:-1]
    bkg_time_edges = source_bkg_curve.binnings[0].edges[1:-1]
    ex_time_edges = exposure_1d_sig.binnings[0].edges[1:-1]

    rebinned_sig_exp, rebinned_time_edges = rebin_nonzero_exposure(signal_exposure, ex_time_edges)
    sig_time_centers = 0.5 * (sig_time_edges[:-1] + sig_time_edges[1:])
    rebinned_signal_counts, _ = np.histogram(sig_time_centers, bins=rebinned_time_edges, weights=N_on)
     
    rebinned_background_exposures = [np.histogram(0.5 * (bkg_time_edges[:-1] + bkg_time_edges[1:]),bins=rebinned_time_edges,weights=bkg_exp)[0] for bkg_exp in background_exposures]    
    rebinned_bkg_exp_sum = np.sum(rebinned_background_exposures,axis=0)
    bkg_time_centers = 0.5 * (bkg_time_edges[:-1] + bkg_time_edges[1:])
    rebinned_background_counts, _ = np.histogram(bkg_time_centers, bins=rebinned_time_edges, weights=N_off)

    
    signal_flux = np.where(signal_exposure > 0, N_on / signal_exposure, 0.0)
    background_flux = np.where(background_exposure_sum > 0, N_off / background_exposure_sum, 0.0)
    alpha = np.where(background_exposure_sum > 0, signal_exposure / background_exposure_sum, 0.0)
    
   

    rebinned_signal_flux = np.where(rebinned_sig_exp > 0,rebinned_signal_counts/ rebinned_sig_exp, 0.0)
    rebinned_background_flux = np.where(rebinned_bkg_exp_sum > 0, rebinned_background_counts / rebinned_bkg_exp_sum, 0.0)
    rebinned_alpha = np.where(rebinned_bkg_exp_sum > 0, rebinned_sig_exp / rebinned_bkg_exp_sum, 0.0)
    significance_li_ma = compute_li_ma_significance(rebinned_signal_counts, rebinned_background_counts, rebinned_alpha)

   

    # Create figure and axis
    fig_ex, ax_ex = plt.subplots(figsize=(12, 4))
    
    # Time conversion
    rebinned_time_dates = [datetime.utcfromtimestamp(t) for t in rebinned_time_edges if np.isfinite(t) and t > 0]
    time_dates_ex = [datetime.utcfromtimestamp(t) for t in exposure_1d_sig.binnings[0].edges if np.isfinite(t) and t > 0]
    
    # Use a perceptually uniform colormap with enough contrast
    colormap = cm.get_cmap("tab10", len(background_exposures) + 1)
    colors = [colormap(i) for i in range(len(background_exposures) + 1)]
    
    # Signal: color[0]
    ax_ex.stairs(signal_exposure, time_dates_ex, label="Signal (daily)", color=colors[0], alpha=0.6, linewidth=1.0)
    ax_ex.stairs(rebinned_sig_exp, rebinned_time_dates, label="Signal (rebinned)", color=colors[0], linewidth=1.5)
    
    # Backgrounds
    for i, (bkg_exp, rebinned_bkg_exp) in enumerate(zip(background_exposures, rebinned_background_exposures)):
        color = colors[i + 1]
        label = f"Background {i+1}"
        ax_ex.stairs(bkg_exp, time_dates_ex, label=f"{label} (daily)", color=color, alpha=0.4, linewidth=1.0)
        ax_ex.stairs(rebinned_bkg_exp, rebinned_time_dates, label=f"{label} (rebinned)", color=color, linewidth=1.5)
    
    # Axes formatting
    ax_ex.set_ylabel("Exposure / cm²s", fontsize=12)
    ax_ex.set_xlabel("Date", fontsize=12)
    ax_ex.set_title(f"{source_name} — Rebinned Exposure", fontsize=14)
    ax_ex.tick_params(axis='y', labelcolor="black")
    ax_ex.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax_ex.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax_ex.xaxis.get_majorticklabels(), rotation=90)
    
    # Place legend outside
    ax_ex.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    
    # Save and layout
    fig_ex.subplots_adjust(bottom=0.3, right=0.78)
    save_figure(fig_ex, args.plotdir, f"{source_name}_rebinned_exposure")
    
    
    # Plot per-bin significance
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    # Convert time edges to datetime for the x-axis
    rebinned_time_edges_dt = [datetime.utcfromtimestamp(t) for t in rebinned_time_edges if np.isfinite(t)]
    ax.stairs(significance_li_ma, rebinned_time_edges_dt, label="Li & Ma", color="purple")
    
    # Axis labels and title
    ax.set_ylabel("Significance (σ)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax.set_title(f"{source_name} — Per-bin Significance", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    
    # Format date axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    
    
    # Adjust layout and save
    fig.subplots_adjust(bottom=0.3)
    save_figure(fig, args.plotdir, f"{args.outputprefix}_{source_name}_significance.png")    
    
    
    #Plot rebinned light curve and exposure
    
    from matplotlib.lines import Line2D

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    # Convert time to datetime
    time_dates = [datetime.utcfromtimestamp(t) for t in rebinned_time_edges if np.isfinite(t) and t > 0]
    
    # Plot light curves
    signal_line = ax1.stairs(rebinned_signal_flux, time_dates, label="Signal Flux", color="blue")
    background_line = ax1.stairs(rebinned_background_flux, time_dates, label="Background Flux", color="red")
    
    # Plot exposure
    exposure_line = ax2.stairs(rebinned_sig_exp, time_dates, color="gray", alpha=0.3, zorder=0, linewidth=1.0)
    
    # Labels
    ax1.set_ylabel("Photons per Exposure", fontsize=12)
    ax2.set_ylabel("Exposure / cm²s", fontsize=12, color="gray")
    ax2.tick_params(axis='y', labelcolor="gray")
    ax2.set_ylim(0, 1.1 * np.nanmax(rebinned_sig_exp))
    
    # Title & x-axis
    ax1.set_title(f"{source_name} — Light Curve per Exposure", fontsize=14)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=45)
    
    # Combine legends
    custom_lines = [
        Line2D([0], [0], color="blue", label="Signal Flux"),
        Line2D([0], [0], color="red", label="Background Flux"),
        Line2D([0], [0], color="gray", alpha=0.3, linewidth=3, label="Signal Exposure"),
    ]
    ax1.legend(handles=custom_lines, loc="upper left")
    
    # Annotate median energy
    ax1.text(
        0.98, 0.95,
        f"Median Energy ≈ {median_energy:.2f} GeV",
        transform=ax1.transAxes,
        ha="right", va="top",
        fontsize=10, color="gray", alpha=0.6
    )
    
    # Save
    plt.tight_layout()
    save_figure(fig, args.plotdir, f"LightCurve_{source_name}_per_exposure.png")


    # Read the source position (already in galactic l, b degrees) from kwargs
    gal_l, gal_b = kwargs["signal_position"]
    
    # Read background positions (already in galactic l, b degrees) from kwargs
    background_positions = kwargs["background_positions"]
    
    # Convert source to Equatorial (ICRS)
    source_gal = SkyCoord(l=gal_l * u.deg, b=gal_b * u.deg, frame="galactic")
    source_eq = source_gal.transform_to("icrs")
    
    # Convert background points to Equatorial
    mark_offset_point_eq = []
    mark_offset_point_gal = []
    
    for bg_pos in background_positions:
        bg_gal = SkyCoord(l=bg_pos[0] * u.deg, b=bg_pos[1] * u.deg, frame="galactic")
        bg_eq = bg_gal.transform_to("icrs")
        mark_offset_point_eq.append((bg_eq.ra.deg, bg_eq.dec.deg))
        mark_offset_point_gal.append((bg_gal.l.deg, bg_gal.b.deg))


    smoothed_sky_histogram_2d = apply_smoothing(sky_histogram_2d, nside, resolution_parameters)
    plot_skymap(smoothed_sky_histogram_2d.project_axis(axis=0),rotate=["G","C"], 
            resultdir=args.resultdir, plotdir=args.plotdir, 
            outputprefix=f"{args.outputprefix}_{source_name}_skymap_Equatorial_smoothed", 
            title=f"{args.title} ({source_name})", 
            mark_point=(source_eq.ra.deg, source_eq.dec.deg), mark_point_size=window_radius, mark_offset_point=mark_offset_point_eq, mark_offset_point_size=window_radius,vmin=4.94, vmax=50)
    
    
    smoothed_sky_histogram_2d_in_window = apply_smoothing(sky_histogram_2d_in_window, nside, resolution_parameters)
    plot_skymap(smoothed_sky_histogram_2d_in_window.project_axis(axis=0), rotate=["G","C"],
            resultdir=args.resultdir, plotdir=args.plotdir, 
            outputprefix=f"{args.outputprefix}_{source_name}_skymap_Equatorial_smoothed_inside_window", 
            title=f"{args.title} ({source_name})", 
            mark_point=(source_eq.ra.deg, source_eq.dec.deg), mark_point_size=window_radius, 
            vmin=0, vmax=7)
    
    
    
    smoothed_sky_histogram_2d = apply_smoothing(sky_histogram_2d, nside, resolution_parameters)
    plot_skymap(smoothed_sky_histogram_2d.project_axis(axis=0), 
            resultdir=args.resultdir, plotdir=args.plotdir, 
            outputprefix=f"{args.outputprefix}_{source_name}_skymap_smoothed", 
            title=f"{args.title} ({source_name})", 
            mark_point=signal_position, mark_point_size=window_radius, mark_offset_point=mark_offset_point_gal, mark_offset_point_size=window_radius,vmin=4.94, vmax=50)
    
    
    smoothed_sky_histogram_2d_in_window = apply_smoothing(sky_histogram_2d_in_window, nside, resolution_parameters)
    plot_skymap(smoothed_sky_histogram_2d_in_window.project_axis(axis=0), 
            resultdir=args.resultdir, plotdir=args.plotdir, 
            outputprefix=f"{args.outputprefix}_{source_name}_skymap_smoothed_inside_window", 
            title=f"{args.title} ({source_name})", 
            mark_point=signal_position, mark_point_size=window_radius, 
            vmin=0, vmax=7)
        
if __name__ == "__main__":
    main()
