#!/usr/bin/env python3

import numpy as np
import multiprocessing as mp
import os
import awkward as ak
import healpy as hp
from astropy.io import fits
from astropy import units
from astropy.coordinates import SkyCoord
from scipy.integrate import quad
from scipy import stats
from xml.etree import ElementTree
import pandas as pd

from tools.binnings import make_lin_binning, make_healpy_binning, Binning, make_flux_energy_binning
from tools.histograms import Histogram, WeightedHistogram, load_histograms, load_histograms_from_files
from tools.fits_tools import read_fits_file
from tools.fermi import integrated_source_flux, integrated_source_flux_no_quad, integrated_source_flux_power_law_xml, integrated_source_flux_log_parabola_xml, integrated_source_flux_pl_super_exp_cutoff_xml, integrated_source_flux_pl_super_exp_cutoff_4_xml

from plot_skymap import plot_skymap, apply_smoothing

def read_xml(xml):
    xml_tree = ElementTree.parse(xml)
    root = xml_tree.getroot()
    SpectrumType = []
    Parameter = []
    GLon = []
    GLat = []
    for source in root.iter('source'):
        if source.attrib['type'] != 'PointSource':
            continue
        ra = None
        dec = None
        for spectrum in source.iter('spectrum'):
            SpectrumType.append(spectrum.attrib['type'])
            paras = {}
            for parameter in spectrum.iter('parameter'):
                paras[parameter.attrib['name']] = float(parameter.attrib['value'])*float(parameter.attrib['scale'])
            Parameter.append(paras)
        for spatial in source.iter("spatialModel"):
            for param in spatial.iter("parameter"):
                if param.attrib["name"] == "RA":
                    ra = float(param.attrib["value"])
                elif param.attrib["name"] == "DEC":                     
                    dec = float(param.attrib["value"])
            break
        if ra is None or dec is None:
            GLon.append(None)
            GLat.append(None)
        coord  = SkyCoord(ra, dec, unit=units.deg, frame="fk5")
        lonlat = coord.galactic
        GLon.append(lonlat.l.value)
        GLat.append(lonlat.b.value)

    return np.array(SpectrumType), np.array(Parameter), np.array(GLon), np.array(GLat)


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

def debug_source(SpectrumType, Parameters, GLON, GLAT, energy_center, e_low, e_up, nside, exposure, energy_index):
    energy_binning = make_flux_energy_binning()
    healpy_binning = make_healpy_binning(nside=nside)

    exposure_maps = load_histograms_from_files([exposure])['exposure']
    exposure_map = exposure_maps.values.T[1:-1][energy_index]
    flux_histogram = WeightedHistogram(energy_binning, healpy_binning, labels=('energy', 'healpy_index'))

    healpy_indices = hp.ang2pix(nside, GLON, GLAT, lonlat=True)
    if SpectrumType == 'PowerLaw':
        flux = integrated_source_flux_power_law_xml(Parameters, e_low, e_up) * exposure_map[1:-1][healpy_indices] * energy_binning.bin_widths[1:-1][energy_index]
    elif SpectrumType == 'LogParabola':
        flux = integrated_source_flux_log_parabola_xml(Parameters, e_low, e_up) * exposure_map[1:-1][healpy_indices] * energy_binning.bin_widths[1:-1][energy_index]
    elif SpectrumType == 'PLSuperExpCutoff4':
        flux = integrated_source_flux_pl_super_exp_cutoff_4_xml(Parameters, e_low, e_up) * exposure_map[1:-1][healpy_indices] * energy_binning.bin_widths[1:-1][energy_index]


    flux = np.nan_to_num(flux, nan = 0.0)

    flux_histogram.fill(np.array([energy_center]), np.array([healpy_indices]), weights=np.array([flux]))

    return flux_histogram




def process_sources(args):
    source_file, nrank, rank, grid_vectors, psf_sigma, energy_center, e_low, e_up, nside, exposure, energy_index = args
    energy_binning = make_flux_energy_binning()
    healpy_binning = make_healpy_binning(nside=nside)

    SpectrumType, Parameters, GLON, GLAT = read_xml(source_file)
    SpectrumType = np.array_split(SpectrumType, nrank)[rank]
    Parameters = np.array_split(Parameters, nrank)[rank]
    GLON = np.array_split(GLON, nrank)[rank]
    GLAT = np.array_split(GLAT, nrank)[rank]

    exposure_maps = load_histograms_from_files([exposure])['exposure']
    exposure_map = exposure_maps.values.T[1:-1][energy_index]

    flux_histogram = WeightedHistogram(energy_binning, healpy_binning, labels=('energy', 'healpy_index'))
    pl_mask = (SpectrumType == 'PowerLaw')
    lp_mask = (SpectrumType == 'LogParabola')
    ec_mask = (SpectrumType == 'PLSuperExpCutoff4')

    if sum(pl_mask) > 0:
        pl_parameters = Parameters[pl_mask]
        pl_parameters = {k: [dic[k] for dic in pl_parameters] for k in pl_parameters[0]}
        for key in pl_parameters.keys():
            pl_parameters[key] = np.array(pl_parameters[key])
        
        glon = GLON[pl_mask]
        glat = GLAT[pl_mask]
        healpy_indices = hp.ang2pix(nside, glon, glat, lonlat=True)
        #vectors = np.array(hp.ang2vec(glon, glat, lonlat=True))
        #separations = get_angel_between_array(vectors, grid_vectors)
        #v = stats.norm.pdf(separations, 0, psf_sigma)
        flux = integrated_source_flux_power_law_xml(pl_parameters, e_low, e_up) * exposure_map[1:-1][healpy_indices] * energy_binning.bin_widths[1:-1][energy_index]
        #flux_psf = v*flux.reshape(len(flux), 1) / v.sum()
        if np.isnan(np.sum(flux)):#_psf)):
            print('pl nan', flush=True)
        #flux_psf = np.nan_to_num(flux_psf, nan = 0.0)
        flux = np.nan_to_num(flux, nan = 0.0)
        flux_histogram.fill(np.ones(len(flux))*energy_center, healpy_indices,weights=flux)
        # for i,source_flux in enumerate(flux):
        #     flux_histogram.fill(energy_center, healpy_indices[i], weights=source_flux)  

    if sum(lp_mask) > 0:
        lp_parameters = Parameters[lp_mask]
        lp_parameters = {k: [dic[k] for dic in lp_parameters] for k in lp_parameters[0]}
        for key in lp_parameters.keys():
            lp_parameters[key] = np.array(lp_parameters[key])
        
        glon = GLON[lp_mask]
        glat = GLAT[lp_mask]
        healpy_indices = hp.ang2pix(nside, glon, glat, lonlat=True)
        vectors = np.array(hp.ang2vec(glon, glat, lonlat=True))
        separations = get_angel_between_array(vectors, grid_vectors)
        #v = stats.norm.pdf(separations, 0, psf_sigma)
        flux = integrated_source_flux_log_parabola_xml(lp_parameters, e_low, e_up) * exposure_map[1:-1][healpy_indices] * energy_binning.bin_widths[1:-1][energy_index]
        #flux_psf = v*flux.reshape(len(flux), 1) / v.sum()
        if np.isnan(np.sum(flux)):#_psf)):
            print('lp nan', flush=True)
        #flux_psf = np.nan_to_num(flux_psf, nan = 0.0)
        flux = np.nan_to_num(flux, nan = 0.0)
        flux_histogram.fill(np.ones(len(flux))*energy_center, healpy_indices,weights=flux)
        # for i,source_flux in enumerate(flux):
        #     flux_histogram.fill(energy_center, healpy_indices[i], weights=source_flux) 

    if sum(ec_mask) > 0:
        ec_parameters = Parameters[ec_mask]
        ec_parameters = {k: [dic[k] for dic in ec_parameters] for k in ec_parameters[0]}
        for key in ec_parameters.keys():
            ec_parameters[key] = np.array(ec_parameters[key])

        glon = GLON[ec_mask]
        glat = GLAT[ec_mask]
        healpy_indices = hp.ang2pix(nside, glon, glat, lonlat=True)
        vectors = np.array(hp.ang2vec(glon, glat, lonlat=True))
        separations = get_angel_between_array(vectors, grid_vectors)
        #v = stats.norm.pdf(separations, 0, psf_sigma)
        flux = integrated_source_flux_pl_super_exp_cutoff_4_xml(ec_parameters, e_low, e_up) * exposure_map[1:-1][healpy_indices] * energy_binning.bin_widths[1:-1][energy_index]
        #flux_psf = v*flux.reshape(len(flux), 1) / v.sum()
        if np.isnan(np.sum(flux)):#_psf)):
            print('ec nan', flush=True)
        flux = np.nan_to_num(flux, nan = 0.0)
        #flux_psf = np.nan_to_num(flux_psf, nan = 0.0)
        flux_histogram.fill(np.ones(len(flux))*energy_center, healpy_indices,weights=flux)
        # for i,source_flux in enumerate(flux):
        #     flux_histogram.fill(energy_center, healpy_indices[i], weights=source_flux) 

    return flux_histogram


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fermi-file', required=True, help='The Fermi-Lat xml file for the diffuse emission model')
    parser.add_argument('--exposure', required=True, help='Path to the npz file containing the exposur maps')
    parser.add_argument('--psf', required=True, help="Path to a file containing a parametrized point spread function.")
    parser.add_argument("--nside", type=int, default=128, help="Number of HEALPix sides to use in the skymap binning.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--outputprefix", default="Fermi", help="Prefix for plots and result files.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--plot-only", default=None, help='Give an result file here if you only want to plot the results')
    parser.add_argument("--no-title", action='store_true', help='Use this to prevent titles from being generated.')
    parser.add_argument("--transparent", action="store_true", help='set this to make the background of the png transparent')
    parser.add_argument('--save-pdf', action="store_true", help='Store the Plots as psf.')
    
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
            
            # Parameters = {'Prefactor': 2.638452015e-10*1.782054164, 'IndexS': -2.226355939, 'Scale': 1000*1.860980729, 'ExpfactorS':0.1*5.625108103, 'Index2':0.4922263025}
            # coord  = SkyCoord(128.837, -45.1781, unit=units.deg, frame="fk5")
            # lonlat = coord.galactic
            # hist = debug_source('PLSuperExpCutoff4',Parameters, lonlat.l.value, lonlat.b.value, energy_center, e_low,e_up, args.nside,args.exposure, energy_index )
            # flux_histogram.add(hist)

            with mp.Pool(args.nprocesses) as pool:
                pool_args = make_args_sources(args.fermi_file, args.nprocesses, grid_vectors, psf_sigma, energy_center, e_low, e_up, args.nside, args.exposure, energy_index)
                for hist in pool.imap_unordered(process_sources, pool_args):
                    flux_histogram.add(hist)

            #print('total',flux_histogram.values)
        psf = np.load(args.psf)
        psf_paras = psf['resolution_parameters_y']
        flux_histogram = apply_smoothing(flux_histogram, nside=args.nside, resolution_parameters=psf_paras, min_smoothing=True)
        print('Final sum', flux_histogram.values.sum())    
        results = {}
        flux_histogram.add_to_file(results, 'fluxes')
        np.savez(os.path.join(args.resultdir, f"{args.outputprefix}_fluxes.npz"), **results)

    else:
        flux_histogram = load_histograms_from_files([args.plot_only])['fluxes']

    vmin = np.ones(29) * 5*1e-3
    vmax = np.ones(29) * 1e1
    for i, (hist, energy_min, energy_max) in enumerate(flux_histogram.project_all(axis=0)):
        plot_skymap(hist, args.resultdir, args.plotdir, f'{args.outputprefix}_source_flux_{i}', title = f"Fermi Source Model, {energy_min:.4g}<=E/GeV<{energy_max:.4g}", transparent=args.transparent, scale='log', vmin=vmin[i], vmax=vmax[i], save_pdf=args.save_pdf)
        if args.no_title:
            plot_skymap(hist, args.resultdir, args.plotdir, f'{args.outputprefix}_source_flux_{i}_NoTitle', title = None, transparent=args.transparent, scale='log', vmin=vmin[i], vmax=vmax[i], save_pdf=args.save_pdf)
    
    plot_skymap(flux_histogram.project_axis(axis=0), args.resultdir, args.plotdir, f'{args.outputprefix}_source_flux', title = f"Fermi Source Model", transparent=args.transparent, scale='log', vmin=vmin[-1], vmax=vmax[-1], save_pdf=args.save_pdf)
    if args.no_title:
        plot_skymap(flux_histogram.project_axis(axis=0), args.resultdir, args.plotdir, f'{args.outputprefix}_source_flux_NoTitle', title = None, transparent=args.transparent, scale='log', vmin=vmin[-1], vmax=vmax[-1], save_pdf=args.save_pdf)








if __name__ == "__main__":
    main()
