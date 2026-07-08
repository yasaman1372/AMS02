#!/usr/bin/env python3

import numpy as np
import multiprocessing as mp
import os
import awkward as ak
import healpy as hp
from astropy.io import fits
from scipy.integrate import quad
from scipy import stats

from tools.binnings import make_lin_binning, make_healpy_binning, Binning, make_flux_energy_binning
from tools.histograms import Histogram, WeightedHistogram, load_histograms, load_histograms_from_files
from tools.fits_tools import read_fits_file
from tools.fermi import integrated_source_flux, integrated_source_flux_no_quad, integrated_source_flux_power_law, integrated_source_flux_log_parabola, integrated_source_flux_pl_super_exp_cutoff

from plot_skymap import plot_skymap

def get_angle_between(point_vector, grid_vectors):
    point_vectors = np.tile(point_vector, [len(grid_vectors), 1])
    return np.rad2deg(np.arccos(np.clip(np.sum(point_vectors*grid_vectors, axis = 1), -1.0, 1.0)))

def get_angel_between_array(point_vectors, grid_vectors):
    point_vectors_expanded = np.expand_dims(point_vectors, axis=1).repeat(len(grid_vectors), axis=1)
    grid_vectors_expanded = np.expand_dims(grid_vectors, axis=0).repeat(len(point_vectors), axis=0)
    return np.rad2deg(np.arccos(np.clip(np.sum(point_vectors_expanded*grid_vectors_expanded, axis = 2), -1.0, 1.0)))

def resolution_parametrization(energy, a, b, c):
    return a / (energy + b) + c

def get_psf_sigma(psf_filename, e_low, e_up, mode = 'center'):
    psf = np.load(psf_filename)
    paras = psf['resolution_parameters_y']
    if mode == 'center':
        E = 10**(0.5*(np.log10(e_low)+ np.log10(e_up)))
        sigma = resolution_parametrization(E, *paras)
    elif mode == 'integrate':
        sigma = quad(lambda E: resolution_parametrization(E, *paras)/E)
        sigma = sigma/(np.log(e_up)- np.log(e_low))
    else:
        raise NotImplementedError('Unknown mode please use "center" or "integrate"')
    return sigma

def make_args_sources(source_file, nrank, grid_vectors, psf_sigma, energy_center, e_low, e_up, nside, exposure, energy_index):
    for rank in range(nrank):
        yield (source_file, nrank, rank, grid_vectors, psf_sigma, energy_center, e_low, e_up, nside, exposure, energy_index)

def process_sources(args):
    source_file, nrank, rank, grid_vectors, psf_sigma, energy_center, e_low, e_up, nside, exposure, energy_index = args
    energy_binning = make_flux_energy_binning()
    healpy_binning = make_healpy_binning(nside=nside)

    sources_hdu = read_fits_file(source_file)
    sources = sources_hdu.data
    sources = np.array_split(sources, nrank)[rank]

    exposure_maps = load_histograms_from_files([exposure])['exposure']
    exposure_map = exposure_maps.values.T[1:-1][energy_index]

    flux_histogram = WeightedHistogram(energy_binning, healpy_binning, labels=('energy', 'healpy_index'))
    pl_sources = sources[sources.SpectrumType == 'PowerLaw']
    lp_sources = sources[sources.SpectrumType == 'LogParabola']
    ec_sources = sources[sources.SpectrumType == 'PLSuperExpCutoff']
    
    
    glon = pl_sources.GLON
    glat = pl_sources.GLAT
    healpy_indices = hp.ang2pix(nside, glon, glat, lonlat=True)
    vectors = np.array(hp.ang2vec(glon, glat, lonlat=True))
    separations = get_angel_between_array(vectors, grid_vectors)
    v = stats.norm.pdf(separations, 0, psf_sigma)
    flux = integrated_source_flux_power_law(pl_sources, e_low, e_up) * exposure_map[1:-1][healpy_indices] * energy_binning.bin_widths[1:-1][energy_index]
    flux = np.nan_to_num(flux, nan=0)
    flux_psf = v*flux.reshape(len(flux), 1) / v.sum()
    if np.isnan(np.sum(flux_psf)):
        print('pl nan', flush=True)
    flux_psf = np.nan_to_num(flux_psf, nan = 0.0)
    for source_flux_psf in flux_psf:
        flux_histogram.fill(energy_center*np.ones(hp.nside2npix(nside)), np.arange(hp.nside2npix(nside)), weights=source_flux_psf)  

    glon = lp_sources.GLON
    glat = lp_sources.GLAT
    healpy_indices = hp.ang2pix(nside, glon, glat, lonlat=True)
    vectors = np.array(hp.ang2vec(glon, glat, lonlat=True))
    separations = get_angel_between_array(vectors, grid_vectors)
    v = stats.norm.pdf(separations, 0, psf_sigma)
    flux = integrated_source_flux_log_parabola(lp_sources, e_low, e_up) * exposure_map[1:-1][healpy_indices] * energy_binning.bin_widths[1:-1][energy_index]
    flux = np.nan_to_num(flux, nan=0)
    flux_psf = v*flux.reshape(len(flux), 1) / v.sum()
    if np.isnan(np.sum(flux_psf)):
        print('lp nan', flush=True)
    flux_psf = np.nan_to_num(flux_psf, nan = 0.0)
    for source_flux_psf in flux_psf:
        flux_histogram.fill(energy_center*np.ones(hp.nside2npix(nside)), np.arange(hp.nside2npix(nside)), weights=source_flux_psf)


    glon = ec_sources.GLON
    glat = ec_sources.GLAT
    healpy_indices = hp.ang2pix(nside, glon, glat, lonlat=True)
    vectors = np.array(hp.ang2vec(glon, glat, lonlat=True))
    separations = get_angel_between_array(vectors, grid_vectors)
    v = stats.norm.pdf(separations, 0, psf_sigma)
    flux = integrated_source_flux_pl_super_exp_cutoff(ec_sources, e_low, e_up) * exposure_map[1:-1][healpy_indices] * energy_binning.bin_widths[1:-1][energy_index]
    flux = np.nan_to_num(flux, nan=0)
    flux_psf = v*flux.reshape(len(flux), 1) / v.sum()
    if np.isnan(np.sum(flux_psf)):
        print('ec nan', flush=True)
    flux_psf = np.nan_to_num(flux_psf, nan = 0.0)
    for source_flux_psf in flux_psf:
        flux_histogram.fill(energy_center*np.ones(hp.nside2npix(nside)), np.arange(hp.nside2npix(nside)), weights=source_flux_psf)
    return flux_histogram

def make_args_energy(nrankE, source_file, nrankSources, grid_vectors, psf, nside):
    for rankE in range(nrankE):
        yield (nrankE, rankE, source_file, nrankSources, grid_vectors, psf, nside)

def process_energy(args):
    nrankE, rankE, source_file, nrankSources, grid_vectors, psf, nside = args
    energy_binning = make_flux_energy_binning()
    healpy_binning = make_healpy_binning(nside=nside)
    flux_histogram = WeightedHistogram(energy_binning, healpy_binning, labels=('energy', 'healpy_index'))
    
    energy_centers = np.array_split(energy_binning.bin_centers[1:-1], nrankE)[rankE]
    energy_edgeS = np.array_split(energy_binning.edges[1:-1], nrankE)[rankE]

    for energy_index, energy_center in enumerate(energy_centers):
        print(energy_center, flush=True)
        e_low = energy_edgeS[energy_index]
        e_up = energy_edgeS[energy_index+1]

        psf_sigma = get_psf_sigma(psf, e_low, e_up)

        with mp.Pool(nrankSources, maxtasksperchild=nrankSources) as pool:
            pool_args = make_args_sources(source_file, nrankSources, grid_vectors, psf_sigma, energy_center, e_low, e_up, nside)
            for hist in pool.imap_unordered(process_sources, pool_args):
                flux_histogram.add(hist)

    return flux_histogram


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fermi-file', required=True, help='The Fermi-Lat fits file for the diffuse emission model')
    parser.add_argument('--exposure', required=True, help='Path to the npz file containing the exposur maps')
    parser.add_argument('--psf', required=True, help="Path to a file containing a parametrized point spread function.")
    parser.add_argument("--nside", type=int, default=128, help="Number of HEALPix sides to use in the skymap binning.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--outputprefix", default="Fermi", help="Prefix for plots and result files.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--plot-only", default=None, help='Give an result file here if you only want to plot the results')
    parser.add_argument("--transparent", action="store_true", help='set this to make the background of the png transparent')

    args = parser.parse_args()
    

    os.makedirs(args.resultdir, exist_ok=True)
    os.makedirs(args.plotdir, exist_ok=True)

    energy_binning = make_flux_energy_binning()
    healpy_binning = make_healpy_binning(nside=args.nside)

    flux_histogram = WeightedHistogram(energy_binning, healpy_binning, labels=('energy', 'healpy_index'))
    grid_vectors = np.array(hp.pix2vec(args.nside, np.arange(hp.nside2npix(args.nside)))).T


    # with mp.Pool(4) as pool:
    #     pool_args = make_args_energy(4, args.fermi_file, int(args.nprocesses/4), grid_vectors, args.psf, args.nside)
    #     for hist in pool.imap_unordered(process_energy, pool_args):
    #         flux_histogram.add(hist)
    if args.plot_only is None:
        for energy_index, energy_center in enumerate(energy_binning.bin_centers[1:-1]):
            print(energy_center, flush=True)
            e_low = energy_binning.edges[1:-1][energy_index]
            e_up = energy_binning.edges[1:-1][energy_index+1]

            psf_sigma = get_psf_sigma(args.psf, e_low, e_up)

            with mp.Pool(args.nprocesses) as pool:
                pool_args = make_args_sources(args.fermi_file, args.nprocesses, grid_vectors, psf_sigma, energy_center, e_low, e_up, args.nside, args.exposure, energy_index)
                for hist in pool.imap_unordered(process_sources, pool_args):
                    flux_histogram.add(hist)

            #print('total',flux_histogram.values)

        print('Final sum', flux_histogram.values.sum())    
        results = {}
        flux_histogram.add_to_file(results, 'fluxes')
        np.savez(os.path.join(args.resultdir, f"{args.outputprefix}_fluxes.npz"), **results)

    else:
        flux_histogram = load_histograms_from_files([args.plot_only])['fluxes']

    for i, (hist, energy_min, energy_max) in enumerate(flux_histogram.project_all(axis=0)):
        plot_skymap(hist, args.resultdir, args.plotdir, f'{args.outputprefix}_source_flux_{i}', title = f"Fermi Source Model, {energy_min:.4g}<=E/GeV<{energy_max:.4g}", transparent=args.transparent)
    plot_skymap(flux_histogram.project_axis(axis=0), args.resultdir, args.plotdir, f'{args.outputprefix}_source_flux', title = f"Fermi Source Model", transparent=args.transparent)








if __name__ == "__main__":
    main()
