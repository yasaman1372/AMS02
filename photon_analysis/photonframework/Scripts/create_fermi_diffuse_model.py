#!/usr/bin/env python3

import numpy as np
import multiprocessing as mp
import os
import awkward as ak
import healpy as hp
from astropy.io import fits
from scipy.integrate import quad
from scipy import stats
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
from numpy.fft import fft2, ifft2, fft, ifft

from tools.binnings import make_lin_binning, make_healpy_binning, Binning, make_flux_energy_binning
from tools.histograms import Histogram, WeightedHistogram, load_histogram, load_histograms_from_files
from tools.fits_tools import read_fits_file

from plot_skymap import plot_skymap, apply_smoothing

def numpy_convolve(data, psf):
    data_fft = fft(data)
    psf_fft = fft([psf])
    return ifft((data_fft*psf_fft))

def convolve(data, psf):   
    data_fft = fftshift(fftn(data))   
    psf_fft = fftshift(fftn(psf))   
    return fftshift(ifftn(ifftshift(data_fft*psf_fft))) 

def deconvolve(data, psf):   
    data_fft = fftshift(fftn(data))   
    psf_fft = fftshift(fftn([psf]))   
    return fftshift(ifftn(ifftshift(data_fft/psf_fft)))

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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--diffuse-file', required=True, help='The npz file with the fermi diffuse emission model')
    parser.add_argument('--exposure', required=True, help='The npz file with the exposure map')
    parser.add_argument('--psf', required=True, help = 'The npz file with the psf')
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

    diffuse_dict = np.load(args.diffuse_file)
    diffuse = load_histograms_from_files([args.diffuse_file])['fluxes']
    exposure = load_histograms_from_files([args.exposure])['exposure']

    healpy_binning = make_healpy_binning(nside=args.nside)
    energy_binning = make_flux_energy_binning()

    dE = energy_binning.bin_widths
    dOmega = hp.nside2pixarea(args.nside)*np.ones(hp.nside2npix(args.nside)+2)
    dEOmega = dOmega*dE.reshape(len(dE),1)

    N_histogram = Histogram(energy_binning, healpy_binning, labels=('energy', 'healpy_index'))
    convoluted_N_histogram = WeightedHistogram(energy_binning, healpy_binning, labels=('energy', 'healpy_index'))

    N_histogram.values = np.nan_to_num(diffuse.values) * exposure.values.T * dEOmega

    longitude_nbins = diffuse_dict['longitude_nbins']
    longitude_min = diffuse_dict['longitude_min']
    longitude_max = diffuse_dict['longitude_max']
    longitude_delta = diffuse_dict['longitude_delta']

    latitude_nbins = diffuse_dict['latitude_nbins']
    latitude_min = diffuse_dict['latitude_min']
    latitude_max = diffuse_dict['latitude_max']
    latitude_delta = diffuse_dict['latitude_delta']
    
    X, Y = np.ogrid[latitude_min  : latitude_max  : latitude_delta,
                    longitude_min : longitude_max : longitude_delta]
    
    if latitude_nbins % 2 != 0:
        X -= 0.5*latitude_delta
    if longitude_nbins % 2 != 0:
        Y -= 0.5*longitude_delta

    healpy_bin_centes = np.array(hp.pix2ang(args.nside, np.arange(hp.nside2npix(args.nside)), lonlat=True))
    
    psf = np.load(args.psf)
    psf_paras = psf['resolution_parameters_y']
    convoluted_N_histogram = apply_smoothing(N_histogram, nside=args.nside, resolution_parameters=psf_paras)
    
    # for energy_index, energy_center in enumerate(energy_binning.bin_centers[1:-1]):
    #     print(energy_center, flush=True)
    #     norm = N_histogram.values[1:-1][energy_index].sum()
    #     e_low = energy_binning.edges[1:-1][energy_index]
    #     e_up = energy_binning.edges[1:-1][energy_index+1]

    #     psf_sigma = get_psf_sigma(args.psf, e_low, e_up)
        
    #     #psf = stats.norm.pdf(np.sqrt(X**2+Y**2), 0 ,psf_sigma)
    #     psf = stats.norm.pdf(np.sqrt(healpy_bin_centes[0,:]**2+healpy_bin_centes[1,:]**2), 0 ,psf_sigma)
    #     psf *= 1.0 / psf.sum()
    #     fix_zeros = np.vectorize(lambda x: x if x!=0 else 1e-10)     
    #     psf = fix_zeros(psf)
    #     test = np.zeros(hp.nside2npix(256))
    #     convoluted_diffuse = np.real(convolve(N_histogram.values[1:-1][energy_index][1:-1], psf))
    #     convoluted_N_histogram.fill(energy_center*np.ones(hp.nside2npix(args.nside)), np.arange(hp.nside2npix(args.nside)), weights=norm/convoluted_diffuse.sum()* convoluted_diffuse)
    #     convoluted_N_histogram.values[1:-1][energy_index][1:-1] = norm/convoluted_diffuse.sum()* convoluted_diffuse

    results = {}
    N_histogram.add_to_file(results, 'N')
    convoluted_N_histogram.add_to_file(results, 'convoluted_N')
    np.savez(os.path.join(args.resultdir, f"{args.outputprefix}_diffuse_model.npz"), **results)

    vmin = np.ones(28) * 1e-1
    vmax = np.ones(28) * 5*1e1
    for i, (hist, energy_min, energy_max) in enumerate(convoluted_N_histogram.project_all(axis=0)):
        title = f"Fermi Diffuse Model, {energy_min:.4g}<=E/GeV<{energy_max:.4g}"
        plot_skymap(hist, args.resultdir, args.plotdir, f'{args.outputprefix}_diffuse_model_convoluted_{i}', title = title , transparent=args.transparent, scale = 'log', vmin = vmin[i], vmax = vmax[i], save_pdf=args.save_pdf)
        if args.no_title:
            title=None
            plot_skymap(hist, args.resultdir, args.plotdir, f'{args.outputprefix}_diffuse_model_convoluted_{i}_NoTitle', title = title , transparent=args.transparent, scale = 'log', vmin = vmin[i], vmax = vmax[i], save_pdf=args.save_pdf)



if __name__ == "__main__":
    main()







