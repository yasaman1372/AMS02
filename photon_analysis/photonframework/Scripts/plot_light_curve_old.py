#!/usr/bin/env python3

import os
import multiprocessing as mp

from matplotlib.colors import PowerNorm
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.dates as mdates

from tools.config import get_config
from tools.binnings import make_healpy_binning, make_energy_binning_from_config
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d, load_histograms_from_files, plot_2d
from tools.roottree import read_tree
from tools.utilities import save_figure
from tools.healpy_tools import mask_sources
from tools.constants import BASTIAN_SKYMAP_PIXEL_AREA, DATA_TIME_RANGES, SKY_REGIONS, SOURCES, TRUE_SOURCES
from tools.coordinates import convert_equatorial_to_galactic_coordinates
from datetime import datetime, timezone

from plot_skymap import plot_skymap, apply_smoothing, no_normalization, normalize_per_square_degree, normalize_per_steradiant, normalize_per_bastian_pixel, NORMALIZATIONS

from calculate_point_spread_function import resolution_parametrization

BRANCHES = ["GalacticLongitude", "GalacticLatitude", "RunNumber", "Energy", "Time"]

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
    window = kwargs["window"]   
    min_energy = kwargs["min_energy"]
    max_energy = kwargs["max_energy"]
    min_time = kwargs["min_time"]
    max_time = kwargs["max_time"]
    healpy_binning = kwargs["healpy_binning"]
    energy_binning = kwargs["energy_binning"]

    
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

        
        # Filter by time range if specified
        if kwargs["min_time"] is not None and kwargs["max_time"] is not None:
            events = events[(events.Time >= kwargs["min_time"]) & (events.Time <= kwargs["max_time"])]
            print(f"[handle_file] Events after time filtering: {len(events)}", flush=True) 

        
        if kwargs["min_energy"] is not None and kwargs["max_energy"] is not None:
            events = events[(events.Energy >= kwargs["min_energy"]) & (events.Energy <= kwargs["max_energy"])]
        
        
        for source_name, (gal_l, gal_b) in selected_sources_limited.items():
           print(f"\nProcessing source: {source_name} (l={gal_l}, b={gal_b})",flush =True) 
        
           # Initialize per-source histogram and event list
           if source_name not in sky_histogram_2d_in_window_list:
               sky_histogram_2d_list[source_name] = Histogram(energy_binning, healpy_binning)               
               sky_histogram_2d_in_window_list[source_name] = Histogram(energy_binning, healpy_binning)
               all_events_in_window_list[source_name] = []
               
              
           # Convert to Healpix index
           healpy_index = hp.ang2pix(nside=kwargs["nside"],
                                                   theta=events.GalacticLongitude,
                                                   phi=events.GalacticLatitude,
                                                   lonlat=True)
                                        
           # Fill per-source histogram
           sky_histogram_2d_list[source_name].fill(events.Energy, healpy_index)


           # Filter events inside the window
           events_in_window = filter_events_in_window(events, (gal_l, gal_b), window)
           print(f"[handle_file] Events in window:{len(events_in_window)}", flush = True)

           all_events_in_window_list[source_name].append(events_in_window.get_array())

           # Convert to Healpix index
           healpy_index_in_window = hp.ang2pix(nside=kwargs["nside"], 
                                               theta=events_in_window.GalacticLongitude, 
                                               phi=events_in_window.GalacticLatitude, 
                                               lonlat=True)
           # Fill per-source histogram
           sky_histogram_2d_in_window_list[source_name].fill(events_in_window.Energy, healpy_index_in_window)




   # print("event list", all_events_in_window_list)
    reduced_array_per_source = {}
    for source_name, array_list in all_events_in_window_list.items():
        source_array = ak.Array({branch: ak.concatenate([a[branch] for a in array_list]) for branch in BRANCHES})
        reduced_array_per_source[source_name] = ak.to_packed(source_array)
    
    return reduced_array_per_source, sky_histogram_2d_list, sky_histogram_2d_in_window_list  
    



def make_args(filename, treename, chunk_size, nranks, **kwargs):
    for rank in range(nranks):
        yield (filename, treename, chunk_size, rank, nranks, kwargs)

def main():

    # Argument parser setup
    import argparse
    parser = argparse.ArgumentParser(description="Plot light curve from photon events.")
    
    parser.add_argument("--config", nargs=2, required=True, help="Path to config file and working directory.")
    parser.add_argument("--regfile", help="Path to the .reg file with sources.")
    parser.add_argument("--target-type", default="Blazar", help="Type of source to analyze (Blazar, Pulsar, GRB, etc.).")
    parser.add_argument("--tree", nargs="+", required=True, help="ROOT files to read.")
    parser.add_argument("--treename", default="GammaTree", help="Name of the tree in the ROOT files.")
    parser.add_argument("--nside", type=int, default=128, help="Number of sides in the healpix binning.")
    parser.add_argument("--outputprefix", default="LightCurve", help="Prefix for plots and result files.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--title", default="Gamma Ray Sky Map", help="Title for plots.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Number of events to read in parallel.")
    parser.add_argument("--single-source", type=str, nargs=3, metavar=("NAME", "LON", "LAT"), help="Name and position of the source (Name GalacticLatitude GalacticLongitude).")
    parser.add_argument("--window", type=float, required=True, help="Window size around the source position.")
    parser.add_argument("--min-energy", type=float, default=None, help="Minimum energy threshold.")
    parser.add_argument("--max-energy", type=float, default=None, help="Maximum energy threshold.")
    parser.add_argument("--min-time", type=str, default=None, help="Minimum time (format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).")
    parser.add_argument("--max-time", type=str, default=None, help="Maximum time (format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).")
    parser.add_argument("--time-bin-size", type=float, default = 150, help="Time bin size in days.")
    parser.add_argument("--psf", help="Path to a file containing a parametrized point spread function. If passed, the skymap will be smoothed with it.") 
    parser.add_argument("--no-title", action='store_true', help='Use this to prevent titles from being generated.')
    parser.add_argument("--transparent", action="store_true", help='set this to make the background of the png transparent')
    parser.add_argument("--save-pdf", action="store_true", help='Store the Plots as psf.')
    
    args = parser.parse_args()
    config_filename, workdir = args.config
    config = get_config(config_filename)
    healpy_binning = make_healpy_binning(args.nside)
    energy_binning = make_energy_binning_from_config(config)


    # Convert human-readable dates to Unix timestamps
    
    
    
    min_time_unix = datetime.fromisoformat(args.min_time).replace(tzinfo=timezone.utc).timestamp()
    max_time_unix = datetime.fromisoformat(args.max_time).replace(tzinfo=timezone.utc).timestamp()
   

    print("openning the reg file.", flush =True)
    
    if args.regfile:
        print(f"Parsing {args.regfile} for {args.target_type} sources...")
        sources = parse_reg_file(args.regfile)
    
        print("Converting coordinates...")
        sources_galactic = {name: convert_equatorial_to_galactic_coordinates(ra, dec) for name, (ra, dec) in sources.items()}
    
        print(f"Filtering for {args.target_type} sources...")
        selected_sources = filter_sources_by_type(sources_galactic, args.target_type)
    
        print(f"Found {len(selected_sources)} sources of type {args.target_type}.")
    
    
    elif args.single_source:
        name = args.single_source[0]  # Extract source name
        lon, lat = map(float, args.single_source[1:])  # Convert lat/lon to float
        selected_sources = {name: (lon, lat)}
    
    else:
        raise ValueError("You must provide either a source file (`--regfile`) or a single source position (`--single_source_position`).")
        

    # Store parameters in kwargs
    kwargs = {
        "nside": args.nside,
        "selected_sources": selected_sources,
        "window": args.window,
        "min_energy": args.min_energy,
        "max_energy": args.max_energy,
        "min_time": min_time_unix,  # Use Unix timestamp
        "max_time": max_time_unix,  # Use Unix timestamp
        "time_bin_size": args.time_bin_size,
        "energy_binning": energy_binning,
        "healpy_binning": healpy_binning
    }
        

    # Create directories if they don't exist
    os.makedirs(args.plotdir, exist_ok=True)
    os.makedirs(args.resultdir, exist_ok=True)
    
    
    # Load events using multiprocessing
    events_per_source = {}  # Dictionary to store events per source
    sky_histogram_2d_per_source = {}  # Dictionary to store sky histograms per source
    sky_histogram_2d_in_window_per_source = {}  # Dictionary to store in-window histograms per source
    
    
    print("Starting multiprocessing pool for event processing...",flush=True)
    
    with mp.Pool(args.nprocesses) as pool:
        pool_args = make_args(args.tree, args.treename, chunk_size=args.chunk_size, nranks=args.nprocesses, **kwargs)
       
        for result in pool.imap_unordered(handle_file, pool_args):
            reduced_array_per_source, hist_2d_per_source, hist_2d_in_window_per_source = result
            

            # Merge results for each source
            for source_name in reduced_array_per_source:
                if source_name in events_per_source:
                    events_per_source[source_name] = ak.concatenate([events_per_source[source_name], reduced_array_per_source[source_name]])
                else:
                    events_per_source[source_name] = reduced_array_per_source[source_name]

                # Merge sky histograms
                if source_name in sky_histogram_2d_per_source:
                    sky_histogram_2d_per_source[source_name].add(hist_2d_per_source[source_name])
                else:
                    sky_histogram_2d_per_source[source_name] = hist_2d_per_source[source_name]
                
                # Merge in-window histograms
                if source_name in sky_histogram_2d_in_window_per_source:
                    sky_histogram_2d_in_window_per_source[source_name].add(hist_2d_in_window_per_source[source_name])
                else:
                  sky_histogram_2d_in_window_per_source[source_name] = hist_2d_in_window_per_source[source_name]
    print("Multiprocessing completed. Now processing individual sources...", flush=True)
   # Loop through each source and process results separately
   
    with np.load(args.psf) as psf_file:
       resolution_parameters = psf_file["resolution_parameters_y"]

    count = 0
    for source_name in events_per_source:
        print(f"Processing source: {source_name}", flush =True)
    
        # Extract events for this source
        masked_events = events_per_source[source_name]
   

        # Flatten the time data (UNIX timestamp)
        Time = ak.flatten(masked_events["Time"], axis=None)
        Time = ak.to_numpy(Time)
    
    
        # Compute time bins
        Time_bin_size_seconds = args.time_bin_size * 24 * 3600
        if len(Time) > 0:  # Ensure there is data before proceeding
            count = count+1 
            Time_min = Time.min()
            Time_max = Time.max()
            Time_bins = np.arange(Time_min, Time_max + Time_bin_size_seconds, Time_bin_size_seconds)
            light_curve, _ = np.histogram(Time, bins=Time_bins)
    
            # Save light curve histogram for each source
            outputname = f"{args.outputprefix}_{source_name}"
            if args.min_time is not None:
                outputname = f"{outputname}_from_{datetime.fromisoformat(args.min_time).year}_{datetime.fromisoformat(args.min_time).month}_{datetime.fromisoformat(args.min_time).day}"
            if args.max_time is not None:
                outputname = f"{outputname}_to_{datetime.fromisoformat(args.max_time).year}_{datetime.fromisoformat(args.max_time).month}_{datetime.fromisoformat(args.max_time).day}"
            lightcurve_output = os.path.join(args.resultdir, f"{outputname}_lightcurve.npz")
            np.savez_compressed(lightcurve_output, light_curve=light_curve, time_bins=Time_bins)
            print(f"Saved light curve histogram for {source_name} to {lightcurve_output}", flush=True)
    
            # Plot Light Curve for this source
            plt.figure(figsize=(8, 6))
    
            # Convert time bins to days relative to start
            Time_days = (Time_bins - Time_bins[0]) / (24 * 3600)
            # Convert Unix timestamps to readable dates
            time_dates = [datetime.utcfromtimestamp(t) for t in Time_bins]
    
            plt.step(time_dates[:-1], light_curve, where="mid", label=f"Event Count ({source_name})", color="tab:blue")
    
            # Format x-axis for readability
            plt.xlim((time_dates[0], time_dates[-1]))
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    
            plt.xlabel("Date")
            plt.ylabel("Number of Events")
            plt.title(f"Light Curve: {source_name}")
    
            # Increase tick size and bold labels
            plt.xticks(rotation=90)
            #plt.yticks(fontsize=14, fontweight='bold')
    
            plt.ylim(bottom=0)
    
            # Make plot borders bold
            # for spine in plt.gca().spines.values():
            #     spine.set_linewidth(2)
    
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.23)
    
            # Save plot            
            lightcurve_plot_output = os.path.join(args.plotdir, f"{outputname}_lightcurve.png")
            plt.savefig(lightcurve_plot_output)
            if args.no_title:
                lightcurve_plot_output = os.path.join(args.plotdir, f"{outputname}_lightcurve_NoTitle.png")
                plt.title(None)
                plt.savefig(lightcurve_plot_output)
            plt.close()
            
            print(f"Saved light curve plot for {source_name} to {lightcurve_plot_output}",flush=True)
    
       
            if count ==1: # just plot one sky map as an example! this can be done giving a specific source name as well!

                smoothed_sky_histogram_2d = apply_smoothing(sky_histogram_2d_per_source[source_name], args.nside, resolution_parameters)
                plot_skymap(smoothed_sky_histogram_2d.project_axis(axis=0), 
                        resultdir=args.resultdir, plotdir=args.plotdir, 
                        outputprefix=f"{outputname}_skymap_smoothed", 
                        title=f"{args.title} ({source_name})", 
                        mark_point=selected_sources[source_name], mark_point_size=args.window, 
                        vmin=4.94, vmax=50, transparent=args.transparent, save_pdf=args.save_pdf)
                if args.no_title:
                    plot_skymap(smoothed_sky_histogram_2d.project_axis(axis=0), 
                        resultdir=args.resultdir, plotdir=args.plotdir, 
                        outputprefix=f"{outputname}_skymap_smoothed_NoTitle", 
                        title=None, 
                        mark_point=selected_sources[source_name], mark_point_size=args.window, 
                        vmin=4.94, vmax=50, transparent=args.transparent, save_pdf=args.save_pdf)    
                
                smoothed_sky_histogram_2d_in_window = apply_smoothing(sky_histogram_2d_in_window_per_source[source_name], args.nside, resolution_parameters)
                plot_skymap(smoothed_sky_histogram_2d_in_window.project_axis(axis=0), 
                        resultdir=args.resultdir, plotdir=args.plotdir, 
                        outputprefix=f"{outputname}_skymap_smoothed_inside_window", 
                        title=f"{args.title} ({source_name})", 
                        mark_point=selected_sources[source_name], mark_point_size=args.window, 
                        vmin=0, vmax=7, transparent=args.transparent, save_pdf=args.save_pdf)
                if args.no_title:
                    plot_skymap(smoothed_sky_histogram_2d_in_window.project_axis(axis=0), 
                        resultdir=args.resultdir, plotdir=args.plotdir, 
                        outputprefix=f"{outputname}_skymap_smoothed_inside_window_NoTitle", 
                        title=None, 
                        mark_point=selected_sources[source_name], mark_point_size=args.window, 
                        vmin=0, vmax=7, transparent=args.transparent, save_pdf=args.save_pdf)
                    
        else:
            print(f"No valid time data found for {source_name}. Skipping light curve generation.", flush=True)    
        
if __name__ == "__main__":
    main()






