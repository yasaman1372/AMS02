#!/usr/bin/env python3

import numpy as np
import multiprocessing as mp
import os
import awkward as ak
import healpy as hp
from astropy.io import fits

from tools.binnings import make_lin_binning, make_healpy_binning, Binning
from tools.histograms import Histogram, WeightedHistogram, load_histogram
from tools.fits_tools import read_fits_file

from plot_skymap import plot_skymap

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fermi-file', required=True, help='The Fermi-Lat fits file for the diffuse emission model')
    parser.add_argument("--nside", type=int, default=128, help="Number of HEALPix sides to use in the skymap binning.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--outputprefix", default="Fermi", help="Prefix for plots and result files.")
    parser.add_argument("--no-title", action='store_true', help='Use this to prevent titles from being generated.')
    parser.add_argument("--transparent", action="store_true", help='set this to make the background of the png transparent')
    parser.add_argument('--save-pdf', action="store_true", help='Store the Plots as psf.')
    #parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")

    args = parser.parse_args()

    os.makedirs(args.resultdir, exist_ok=True)
    os.makedirs(args.plotdir, exist_ok=True)

    fermi_iem_hdu = read_fits_file(args.fermi_file)
    fermi_iem_energy_hdu = read_fits_file(args.fermi_file, 'ENERGIES')
    fermi_header = fermi_iem_hdu.header
    fermi_energy_header = fermi_iem_energy_hdu.header
    fermi_data = fermi_iem_hdu.data
    fermi_energy_data = fermi_iem_energy_hdu.data

    healpy_binning = make_healpy_binning(nside=args.nside)

    longitude_bins = int(fermi_header['NAXIS1'])
    longitude_delta = abs(float(fermi_header['CDELT1']))
    longitude_min = longitude_delta*(0.5 - float(fermi_header['CRPIX1'])) + float(fermi_header['CRVAL1'])
    longitude_max = longitude_min + longitude_bins * longitude_delta
    longitude_bin_centers = longitude_delta/2 + longitude_min + np.arange(longitude_bins) * longitude_delta

    latitude_bins = int(fermi_header['NAXIS2'])
    latitude_delta = abs(float(fermi_header['CDELT2']))
    latitude_min = latitude_delta*(0.5 - float(fermi_header['CRPIX2'])) + float(fermi_header['CRVAL2'])
    latitude_max = latitude_min + latitude_bins * latitude_delta
    latitude_bin_centers = latitude_delta/2 + latitude_min + np.arange(latitude_bins) * latitude_delta

    if 'energy' in fermi_energy_data.names:
        fermi_energies = fermi_energy_data.energy
    elif 'Energy' in fermi_energy_data.names:
        fermi_energies = fermi_energy_data.Energy
    else:
        raise NotImplementedError('Invalid input file, please check energy axis.')
    
    fermi_energy_bins = len(fermi_energies)
    fermi_energy_bin_centers = np.array(list(fermi_energies))

    fermi_energy_bin_width = np.log10(fermi_energy_bin_centers[1:])-np.log10(fermi_energy_bin_centers[:-1])
    fermi_energy_bin_width = list(fermi_energy_bin_width)
    fermi_energy_bin_width.append(fermi_energy_bin_width[-1])
   
    fermi_energy_bin_edges = np.zeros(fermi_energy_bins+1)
    fermi_energy_bin_edges[0] = 10**(np.log10(fermi_energy_bin_centers[0])- fermi_energy_bin_width[0]/2)
    
    for i, center in enumerate(fermi_energy_bin_centers):
        fermi_energy_bin_edges[i+1] = 10**(np.log10(center)+fermi_energy_bin_width[i]/2)

    fermi_energy_binning = Binning(np.array(fermi_energy_bin_edges)/1000, log=True)
    print(list(fermi_energy_binning.edges))
    flux_histogram = WeightedHistogram(fermi_energy_binning, healpy_binning, labels=('energy', 'healpy_index'))
    nop_histogram = Histogram(fermi_energy_binning, healpy_binning, labels=('energy', 'healpy_index'))

    for energy_index in range(fermi_energy_bins):
        print(fermi_energy_binning.bin_centers[1:-1][energy_index], flush=True)
        fermi_data_energy_bin = fermi_data[energy_index]  #1/cm²/s/MeV/sr
        fermi_data_energy_bin = fermi_data_energy_bin.astype(np.float32)
        fermi_data_energy_bin = fermi_data_energy_bin*1000 #1/cm²/s/GeV/sr
        fermi_data_energy_bin = np.nan_to_num(fermi_data_energy_bin)
        longitude, latitude = np.meshgrid(-longitude_bin_centers, latitude_bin_centers, indexing='ij')
        grid_pix = hp.ang2pix(args.nside, longitude, latitude, lonlat=True)
        flux_histogram.fill(fermi_energy_binning.bin_centers[1:-1][energy_index]*np.ones(len(grid_pix.ravel())), grid_pix.ravel(), weights=fermi_data_energy_bin.T.ravel())
        nop_histogram.fill(fermi_energy_binning.bin_centers[1:-1][energy_index]*np.ones(len(grid_pix.ravel())), grid_pix.ravel())

    flux_histogram = flux_histogram/nop_histogram.values

    results = {}
    results['longitude_nbins'] = longitude_bins
    results['longitude_min'] = longitude_min
    results['longitude_max'] = longitude_max
    results['longitude_delta'] = longitude_delta
    results['latitude_nbins'] = latitude_bins
    results['latitude_min'] = latitude_min
    results['latitude_max'] = latitude_max
    results['latitude_delta'] = latitude_delta
    flux_histogram.add_to_file(results, 'fluxes')
    np.savez(os.path.join(args.resultdir, f"{args.outputprefix}_fluxes.npz"), **results)

    vmin = np.ones(28) * 3*1e-7
    vmax = np.ones(28) * 3*1e-4
    for i, (hist, energy_min, energy_max) in enumerate(flux_histogram.project_all(axis=0)):
        title = f"Fermi Diffuse Model, {energy_min:.4g}<=E/GeV<{energy_max:.4g}"
        plot_skymap(hist, args.resultdir, args.plotdir, f'{args.outputprefix}_diffuse_flux_{i}', title=title, scale = 'log', vmin = vmin[i], vmax = vmax[i], transparent= args.transparent, save_pdf = args.save_pdf, normalization="flux")
        if args.no_title:
            title=None
            plot_skymap(hist, args.resultdir, args.plotdir, f'{args.outputprefix}_diffuse_flux_{i}_NoTitle', title=title, scale = 'log', vmin = vmin[i], vmax = vmax[i], transparent= args.transparent, save_pdf = args.save_pdf, normalization="flux")

    
    

 


    


if __name__ == "__main__":
    main()
        


    
