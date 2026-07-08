#!/usr/bin/env python3

from datetime import datetime
import os
import multiprocessing as mp

import matplotlib as mpl

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm, Normalize
from matplotlib.projections.geo import GeoAxes
from matplotlib.patches import Circle
import healpy as hp
from healpy.rotator import Rotator
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline, BSpline


from regions import Regions, EllipseSkyRegion, PointSkyRegion, CircleSkyRegion, PolygonSkyRegion, RectangleSkyRegion
import astropy.units as u
from matplotlib.patches import Circle

from tools.config import get_config
from tools.constants import BASTIAN_SKYMAP_PIXEL_AREA, DATA_TIME_RANGES, SKY_REGIONS, SOURCES, TRUE_SOURCES
from tools.binnings import make_healpy_binning, make_energy_binning_from_config, make_flux_energy_binning, make_lin_binning
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d, load_histograms_from_files, plot_2d
from tools.roottree import read_tree
from tools.utilities import save_figure
from tools.healpy_tools import mask_sources, get_regions, get_sources, rectangle_to_healpy_polygon, ellipse_to_healpy_polygon
from tools.statistics import hist_mean_and_std, calculate_chisq

from plot_skymap import NORMALIZATIONS


def resolution_parametrization(energy, a, b, c):
    return a / (energy + b) + c

def plot_skymap(sky_histogram, resultdir, plotdir, outputprefix, title, colormap="jet", vmin=0, vmax=None, scale=None, mask_region=None, invert_mask=False, rotate=None, resolution=30, save_pickle=False, plot_sources=False, normalization="events-per-square-degree", transparent=False, white_zero = False):
    figure = plt.figure(figsize=(9.3, 4.5 if title is not None else 4.2))
    plot = figure.subplots(1, 1, subplot_kw=dict(projection="hammer"))
    nside = hp.npix2nside(len(sky_histogram.values) - 2)
    if title is not None:
        plot.set_title(title, pad=15)
    theta_pixels = 90 * resolution
    phi_pixels = 180 * resolution
    theta = np.linspace(np.pi, 0, theta_pixels)
    phi = np.linspace(-np.pi, np.pi, phi_pixels)
    phi, theta = np.meshgrid(phi, theta)
    if rotate is not None:
        r = Rotator(coord=rotate, inv=True)
        theta, phi = r(theta.flatten(), phi.flatten())
        theta = theta.reshape(theta_pixels, phi_pixels)
        phi = phi.reshape(theta_pixels, phi_pixels)
    grid_pix = hp.ang2pix(nside, theta, phi)
    map_values = sky_histogram.values[1:-1]
    if mask_region is not None:
        map_values = mask_sources(map_values, mask_region, mask_type=not invert_mask, inclusive=not invert_mask)
    if white_zero:
        map_values = np.ma.masked_array(map_values, mask=(map_values==0))
    grid_map = map_values[grid_pix]
    pixel_norm_func, colorscale_label = NORMALIZATIONS[normalization]
    pixel_norm = pixel_norm_func(nside)

    theta_plot = np.linspace(np.pi, 0, theta_pixels)
    phi_plot = np.linspace(-np.pi, np.pi, phi_pixels)
    if not scale or scale == 'sqrt':
        norm = PowerNorm(0.5, vmin=vmin, vmax=vmax, clip=True)
    elif scale == 'log':
        norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
    elif scale == 'lin':
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    mesh = plot.pcolormesh(-phi_plot, -theta_plot + np.pi / 2, grid_map / pixel_norm, norm=norm, cmap=colormap)

    # Add grid and color bar
    plot.grid(alpha=0.3)
    cbar = plt.colorbar(mesh, ax=plot, extend="both", fraction=0.08)
    cbar.set_label(colorscale_label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    if rotate is None or rotate[-1] == "G":
        plot.text(1, 0, "Galactic Coordinates", transform=plot.transAxes, ha="right", va="bottom")
    elif rotate[-1] == "C":
        plot.text(1, 0, "Equatorial Coordinates", transform=plot.transAxes, ha="right", va="bottom")
    elif rotate[-1] == "E":
        plot.text(1, 0, "Ecliptic Coordinates", transform=plot.transAxes, ha="right", va="bottom")
    else:
        raise ValueError(f"Unknown coordinate system {rotate!r}.")
    plot.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{(-x / np.pi * 180) % 360:.0f}°"))

    if plot_sources:
        for name, source in TRUE_SOURCES.items():
            plot.scatter(np.radians(source[0][0]), np.radians(source[0][1]), s=5, c=source[1], marker='x', label = name)
        for name, source in SOURCES.items():
            source_circle = Circle(np.radians(source[0][0:2]), np.radians(source[0][2]), color = source[1], fill = False, linewidth = 1)
            plot.add_patch(source_circle)

        plot.legend(loc='lower right', fontsize='x-small', labelcolor='mfc')

    # Finalize plot
    if title is not None:
        figure.subplots_adjust(left=0.033, right=0.975, bottom=0.015, top=0.95)
    else:
        figure.subplots_adjust(left=0.033, right=0.975, bottom=0.0175, top=0.985)
    save_figure(figure, plotdir, f"{outputprefix}", dpi=600, save_pickle=save_pickle, transparent=transparent)
    if rotate is None:
        coord = 'G'
    else:
        coord = rotate[-1]

    header = {
        "BUNIT": "deg^2",       # Units of the map 
        "MAPTYPE": "events",    # Description of the data
        "NSIDE": nside,         # HEALPix nside parameter
        "COORDSYS": coord,      # Coordinate system 
    }

    if isinstance(map_values, np.ma.masked_array):
        map_values = map_values.filled(0)
    
    hp.fitsfunc.write_map(os.path.join(resultdir, f"{outputprefix}.fits"), map_values, fits_IDL=True, extra_header=header, overwrite=True)

def get_regions(region_file, format='ds9', psf_sigma=0):
    Regs = []
    SkyRegions = Regions.read(region_file, format=format)
    for reg in SkyRegions:
        if isinstance(reg, CircleSkyRegion):
            l = reg.center.l.deg
            if l < 180:
                l = -l
            elif l >= 180:
                l = 360-l
            b = reg.center.b.deg
            r = reg.radius.to(u.deg).value
            if r >= 2*psf_sigma:
                healpy_reg = ('Disc', (l, b, r))
            else:
                healpy_reg = ('Disc', (l, b, 2*psf_sigma))
        elif isinstance(reg, PolygonSkyRegion):
            vertices = []
            for v in reg.vertices:
                l = v.l.deg
                if l < 180:
                    l = -l
                elif l >= 180:
                    l = 360 - l
                b = v.b.deg
                vertices.append(l)
                vertices.append(b)
            healpy_reg = ('Polygon', tuple(vertices))
        elif isinstance(reg, EllipseSkyRegion):
            vertices = ellipse_to_healpy_polygon(reg)
            healpy_reg = ('Polygon', tuple(vertices))
        elif isinstance(reg, RectangleSkyRegion):
            height = reg.height.to(u.deg).value / 2
            if height > psf_sigma:
                healpy_reg = rectangle_to_healpy_polygon(reg)
            else:
                healpy_reg = rectangle_to_healpy_polygon(reg, height=2*psf_sigma)
        Regs.append(healpy_reg)
    return Regs

def poly20(x, *params):
    x = np.array(x)
    return np.where(x >= -75, sum(p * np.array(x)**i for i, p in enumerate(params)), 0)

def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--skymap', required=True, help='The npz file with the Skymap histograms')
    parser.add_argument('--source-model', required=True, help='The npz file with the source model histograms')
    parser.add_argument('--diffuse-model', required=True, help='The npz file with the diffuse model histograms')
    parser.add_argument('--psf', required=True, help = 'The npz file with the psf')
    parser.add_argument("--outputprefix", default="SkyMap", help="Prefix for plots and result files.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--title", default="Gamma Ray Sky Map", help="Title for plots.")
    parser.add_argument("--vmin", type=float, default=None, help="Number of events per bin to use as the minimum of the colorscale.")
    parser.add_argument("--vmax", type=float, default=None, help="Number of events per bin to use as the maximum of the colorscale.")
    parser.add_argument('--scale', default='sqrt', choices=['sqrt', 'log', 'lin'], help='Scale of the map')
    parser.add_argument("--residuum-colormap", choices=['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu','RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic','berlin', 'managua', 'vanimo'], default="bwr", help="Colormap to use.")
    parser.add_argument("--coord", nargs="+", type=str, default=None, choices=['G','C','E'], help="Either one of ‘G’ (Galactic), ‘E’ (Ecliptic) or ‘C’ (Celestial/Equatorial) to describe the coordinate system of the map, or a sequence of 2 of these to rotate the map from the first to the second coordinate system. The standard map is in 'G'.")
    parser.add_argument("--s-bg-regions", required=True, nargs=2, type=str, help='Path to .reg file for signal and bg region')
    parser.add_argument("--normalization", choices=list(NORMALIZATIONS.keys()), default="events", help="Pixel normalization to apply.")
    parser.add_argument('--save-pickle', action="store_true", help='Store the Plots as pickeld matplotlib plotts.')
    parser.add_argument("--no-title", action='store_true', help='Use this to prevent titles from being generated.')
    parser.add_argument("--transparent", action="store_true", help='set this to make the background of the png transparent')
    parser.add_argument("--migration", default=None, help="Npz file with matrix")

    args = parser.parse_args()

    os.makedirs(args.plotdir, exist_ok=True)
    os.makedirs(args.resultdir, exist_ok=True)

    Skymap = load_histograms_from_files([args.skymap])['smoothed']
    
    Source_map = load_histograms_from_files([args.source_model])['fluxes']
    Diffuse_map = load_histograms_from_files([args.diffuse_model])['convoluted_N']
    #Diffuse_map.values[1:-1] *= Diffuse_map.binnings[0].bin_widths[1:-1].reshape(len(Diffuse_map.binnings[0].bin_widths[1:-1]), -1)
    #Diffuse_map.values[1:-1] /= 2
    
    psf = np.load(args.psf)
    psf_paras = psf['resolution_parameters_y']
    

    Model = Histogram(*Source_map.binnings)

    Model.add(Source_map)
    Model.add(Diffuse_map)

    if args.migration is not None:
        Source_map.values = np.nan_to_num(Source_map.values, nan=0)
        Diffuse_map.values = np.nan_to_num(Diffuse_map.values, nan=0)
        Model.values = np.nan_to_num(Model.values, nan=0)
        
        migration_matrix = load_histograms_from_files([args.migration])["migration_hist"]
        migration = migration_matrix.values
        migration = migration/np.sum(migration, axis = 1)
        
        Source_map.values = np.tensordot(migration, Source_map.values, axes=(0,0))
        Diffuse_map.values = np.tensordot(migration, Diffuse_map.values, axes=(0,0))
        Model.values = np.tensordot(migration, Model.values, axes = (0,0))
        

    Skymap_100_100, vmin, vmax = Skymap.project_by_value(min_value=0.1, max_value=100, axis = 0, return_bin_edges=True)
    #print(f'\n\n{vmin}, {vmax}\n\n')
    Model_100_100 = Model.project_by_value(min_value=0.1, max_value=100, axis = 0)
    Source_100_100 = Source_map.project_by_value(min_value=0.1, max_value = 100, axis = 0)
    Diffuse_100_100 = Diffuse_map.project_by_value(min_value=0.1, max_value = 100, axis = 0)

    Skymap_100_100_BG = Histogram(*Skymap_100_100.binnings)
    Model_100_100_BG = Histogram(*Model_100_100.binnings)
    Residuum_100_100_BG = Histogram(*Skymap_100_100.binnings)
    Skymap_BG = Histogram(*Skymap.binnings)
    Model_BG = Histogram(*Model.binnings)
    Residuum_BG = Histogram(*Skymap.binnings)
    signal_region, background_region = args.s_bg_regions
    Skymap_100_100_BG.values[1:-1] = mask_sources(Skymap_100_100.values[1:-1], get_regions(background_region), True, True).filled(0)
    Model_100_100_BG.values[1:-1] = mask_sources(Model_100_100.values[1:-1], get_regions(background_region), True, True).filled(0)
    Model_mask = Model_100_100_BG.values >= 2
    Model_100_100_BG.values[Model_mask] = np.zeros(np.sum(Model_mask))
    Skymap_100_100_BG.values[Model_mask] = np.zeros(np.sum(Model_mask))
    Residuum_100_100_BG.values[1:-1] = Skymap_100_100_BG.values[1:-1] - Model_100_100_BG.values[1:-1]
    for i in range(len(Skymap_BG.binnings[0].bin_centers)):
        psf_sigma = resolution_parametrization(Skymap_BG.binnings[0].bin_centers[i], *psf_paras)
        Skymap_BG.values[i][1:-1] = mask_sources(Skymap.values[i][1:-1], get_regions(background_region, psf_sigma=psf_sigma), True, True).filled(0)
        Model_BG.values[i][1:-1] = mask_sources(Model.values[i][1:-1], get_regions(background_region, psf_sigma=psf_sigma), True, True).filled(0)
        Residuum_BG.values[i][1:-1] = Skymap_BG.values[i][1:-1] - Model_BG.values[i][1:-1]
        Residuum_BG.values[i][Model_mask] = np.zeros(np.sum(Model_mask))

    plot_skymap(Residuum_100_100_BG, args.resultdir, args.plotdir, f'{args.outputprefix}_residuum_map',title= f"{args.title} Residuum Data Model", scale = 'lin', vmin = -1, vmax = 1, normalization="bg_residuum", rotate='G', colormap='coolwarm', white_zero = True)#, mask_region=get_regions(background_region), invert_mask=False)
    if args.no_title:
        plot_skymap(Residuum_100_100_BG, args.resultdir, args.plotdir, f'{args.outputprefix}_residuum_map_noTitle',title= None, scale = 'lin', vmin = -1, vmax = 1, normalization="bg_residuum", rotate='G', colormap='coolwarm', white_zero = True)#, mask_region=get_regions(background_region), invert_mask=False)
    # for i, (hist, energy_min, energy_max) in enumerate(Residuum_BG.project_all(axis=0)):
    #     title = f"BG, {energy_min:.4g}<=E/GeV<{energy_max:.4g}"
    #     plot_skymap(hist, args.resultdir, args.plotdir, f'{args.outputprefix}_residuum_map_{i}', title=title, scale = 'lin', vmin = -0.5, vmax = 0.5, normalization="bg_residuum", rotate='G', colormap='coolwarm', white_zero = True)


    nside = hp.npix2nside(len(Residuum_100_100_BG.values[1:-1]))
    pix = np.arange(hp.nside2npix(nside))
    ang = hp.pix2ang(nside, pix)
    rot_ang = hp.Rotator(coord=['G','C'], inv=True)(*ang)
    rot_pix = hp.ang2pix(nside, *rot_ang)
    Residuum_100_100_Eq = Histogram(*Residuum_100_100_BG.binnings)
    Residuum_Eq = Histogram(*Residuum_BG.binnings)

    for i, n in enumerate(rot_pix):
        Residuum_100_100_Eq.values[1:-1][i] = Residuum_100_100_BG.values[1:-1][n]
        for j in range(len(Residuum_Eq.binnings[0].bin_centers)):
            Residuum_Eq.values[j][1:-1][i] = Residuum_BG.values[j][1:-1][n]

    # plot_skymap(Residuum_100_100_Eq, args.resultdir, args.plotdir, f'{args.outputprefix}_residuum_equatorial_map',title= f"{args.title} Residuum Data Model", scale = 'sqrt', vmin = 0, vmax = 100, normalization="events", rotate='C')#, mask_region=get_regions(background_region), invert_mask=False)
    # for i, (hist, energy_min, energy_max) in enumerate(Residuum_Eq.project_all(axis=0)):
    #     title = f"BG, {energy_min:.4g}<=E/GeV<{energy_max:.4g}"
    #     plot_skymap(hist, args.resultdir, args.plotdir, f'{args.outputprefix}_BG_{i}', title=title, scale = 'sqrt', vmin = 0, vmax = 100, normalization=args.normalization, rotate='C')

    
    declanation = hp.pix2ang(nside,pix, lonlat=True)[1]
    dec_rings = sorted(list(set(declanation)))
    Photons_per_pixel_in_dec = np.zeros(len(dec_rings))
    Photons_per_pixel_in_dec_std = np.zeros(len(dec_rings))
    Photons_per_pixel_in_dec_all_E = np.zeros((len(Residuum_BG.binnings[0].bin_centers), len(dec_rings)))
    Photons_per_pixel_in_dec_all_E_std = np.zeros((len(Residuum_BG.binnings[0].bin_centers), len(dec_rings)))


    Residuum_100_100_Eq.values[1:-1] = Residuum_100_100_Eq.values[1:-1]/hp.nside2pixarea(nside)
    Residuum_Eq.values = Residuum_Eq.values/hp.nside2pixarea(nside)
    scale_factor = np.zeros(len(Residuum_Eq.binnings[0].bin_centers))
    for i, dec in enumerate(dec_rings):
        value_mask = Residuum_100_100_Eq.values[1:-1] != 0
        dec_mask = declanation == dec
        mask = value_mask & dec_mask
        Photons_per_pixel_in_dec[i] = np.mean(Residuum_100_100_Eq.values[1:-1][mask])
        Photons_per_pixel_in_dec_std[i] = np.std(Residuum_100_100_Eq.values[1:-1][mask])/np.sqrt(np.sum(mask))
        for j in range(len(Residuum_Eq.binnings[0].bin_centers)):
            value_mask = Residuum_Eq.values[j][1:-1] != 0
            mask = value_mask & dec_mask
            Photons_per_pixel_in_dec_all_E[j][i] = np.mean(Residuum_Eq.values[j][1:-1][mask])
            Photons_per_pixel_in_dec_all_E_std[j][i] = np.std(Residuum_Eq.values[j][1:-1][mask])/np.sqrt(np.sum(mask))
            scale_factor[j] = np.mean(Photons_per_pixel_in_dec_all_E[j])

    # for i in range(25):
    #     popt, pcov = curve_fit(poly20, dec_rings, Photons_per_pixel_in_dec, sigma=Photons_per_pixel_in_dec_std, absolute_sigma=True, p0=np.ones(i+1), maxfev = 100000)
    #     _,_,chiq  = calculate_chisq(Photons_per_pixel_in_dec, poly20(dec_rings, *popt), Photons_per_pixel_in_dec_std, len(popt))
    #     print(i,chiq)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.scatter(Residuum_Eq.binnings[0].bin_centers[1:-1], scale_factor[1:-1])
    ax.set_ylabel("Scale Factor / events/pixel")
    ax.set_xlabel("Energy / GeV")
    ax.set_title("Scale Factor for each Energy bin")
    ax.semilogx()
    ax.semilogy()
    save_figure(fig, args.plotdir, f"{args.outputprefix}_Scale_Factor", transparent=args.transparent)

    
    popt, pcov = curve_fit(poly20, dec_rings, Photons_per_pixel_in_dec, sigma=Photons_per_pixel_in_dec_std, absolute_sigma=True, p0=np.ones(25))
    _,_,chiq  = calculate_chisq(Photons_per_pixel_in_dec, poly20(dec_rings, *popt), Photons_per_pixel_in_dec_std, len(popt))
    print(chiq)
    # Photons_per_pixel_in_dec[:81] = np.zeros(81)
    # Photons_per_pixel_in_dec[996:] = np.zeros(27)
    bspline = UnivariateSpline(dec_rings, Photons_per_pixel_in_dec, w = 1/Photons_per_pixel_in_dec_std, s = 1000)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.errorbar(dec_rings, Photons_per_pixel_in_dec, yerr = Photons_per_pixel_in_dec_std, marker=".", ls='', ms=1, elinewidth=1 )
    #ax.plot(np.linspace(-90,90,360), poly20(np.linspace(-90,90,360), *popt), color = 'red')
    ax.plot(np.linspace(-90,90,360), bspline(np.linspace(-90,90,360)), color = 'red')
    ax.set_xlabel("Declanation / deg")
    ax.set_ylabel("(Data - Model)/$\Delta\Omega$ / events/sr")
    save_figure(fig, args.plotdir, f"{args.outputprefix}_Background_Model", transparent=args.transparent)


    
    for j in range(len(Residuum_Eq.binnings[0].bin_centers)):
        Photons_per_pixel_in_dec_all_E[j] = np.nan_to_num(Photons_per_pixel_in_dec_all_E[j], nan=0)
        Photons_per_pixel_in_dec_all_E_std[j] = np.nan_to_num(Photons_per_pixel_in_dec_all_E_std[j], nan=0)
        popt_all_E, pcov = curve_fit(poly20, dec_rings, Photons_per_pixel_in_dec_all_E[j], sigma=Photons_per_pixel_in_dec_all_E_std[j], absolute_sigma=True, p0=np.ones(25), maxfev = 100000)
        _,_,chiq  = calculate_chisq(Photons_per_pixel_in_dec_all_E[j], poly20(dec_rings, *popt_all_E), Photons_per_pixel_in_dec_all_E_std[j], len(popt_all_E))
        print(chiq)
        # Photons_per_pixel_in_dec_all_E[j][:81] = np.zeros(81)
        # Photons_per_pixel_in_dec_all_E[j][996:] = np.zeros(27)
        bspline_all_E = UnivariateSpline(dec_rings, Photons_per_pixel_in_dec_all_E[j], w = 1/Photons_per_pixel_in_dec_all_E_std[j], s = 1000)
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.errorbar(dec_rings, Photons_per_pixel_in_dec_all_E[j], yerr = Photons_per_pixel_in_dec_all_E_std[j], marker=".", ls='', ms=1, elinewidth=1 )
        #ax.plot(np.linspace(-90,90,360), poly20(np.linspace(-90,90,360), *popt_all_E), color = 'red')
        ax.plot(np.linspace(-90,90,360), bspline_all_E(np.linspace(-90,90,360)), color = 'red')
        ax.set_xlabel("Declanation / deg")
        ax.set_ylabel("(Data - Model)/$\Delta\Omega$ / events/sr")
        title = f"BG, {Residuum_Eq.binnings[0].edges[j]:.4g}<=E/GeV<{Residuum_Eq.binnings[0].edges[j+1]:.4g}"
        ax.set_title(title)
        save_figure(fig, args.plotdir, f"{args.outputprefix}_Background_Model_{j}", transparent=args.transparent)
        

    Bg_model_map_eq = Histogram(*Residuum_100_100_Eq.binnings)
    Bg_model_map_gal = Histogram(*Residuum_100_100_Eq.binnings)

    for i, dec in enumerate(dec_rings):
        dec_mask = declanation == dec
        n_pix = np.sum(dec_mask)
        #Bg_model_map_eq.values[1:-1][dec_mask] = np.ones(n_pix)*poly20(dec, *popt) * hp.nside2pixarea(nside)
        Bg_model_map_eq.values[1:-1][dec_mask] = np.ones(n_pix)*bspline(dec) * hp.nside2pixarea(nside)

    rev_rot_ang = hp.Rotator(coord=['C','G'], inv=True)(*ang)
    rev_rot_pix = hp.ang2pix(nside, *rev_rot_ang)
    for i, n in enumerate(rev_rot_pix):
        Bg_model_map_gal.values[1:-1][i] = Bg_model_map_eq.values[1:-1][n]

    # plot_skymap(Bg_model_map_eq, args.resultdir, args.plotdir, f'{args.outputprefix}_background_model_equatorial',title= f"{args.title} BG Model", scale = 'log', vmin = None, vmax = None, normalization='events', rotate='C', transparent=args.transparent)#, mask_region=get_regions(background_region), invert_mask=False)
    # plot_skymap(Bg_model_map_gal, args.resultdir, args.plotdir, f'{args.outputprefix}_background_model',title= f"{args.title} BG Model", scale = 'log', vmin = 1e-4, vmax = 1e-1, normalization='events', rotate='G', transparent=args.transparent)#, mask_region=get_regions(background_region), invert_mask=False)
    # if args.no_title:
    #     plot_skymap(Bg_model_map_eq, args.resultdir, args.plotdir, f'{args.outputprefix}_background_model_equatorial_NoTitle',title= None, scale = 'log', vmin = None, vmax = None, normalization='events', rotate='C', transparent=args.transparent)#, mask_region=get_regions(background_region), invert_mask=False)
    #     plot_skymap(Bg_model_map_gal, args.resultdir, args.plotdir, f'{args.outputprefix}_background_model_NoTitle',title= None, scale = 'log', vmin = 1e-4, vmax = 1e-1, normalization='events', rotate='G', transparent=args.transparent)#, mask_region=get_regions(background_region), invert_mask=False)


    Bg_model_map_eq_norm = Histogram(*Residuum_100_100_Eq.binnings)
    Bg_model_map_gal_norm = Histogram(*Residuum_100_100_Eq.binnings)
    Bg_model_map_eq_norm.values[1:-1] = Bg_model_map_eq.values[1:-1]/np.max(Bg_model_map_eq.values[1:-1])
    Bg_model_map_gal_norm.values[1:-1] = Bg_model_map_gal.values[1:-1]/np.max(Bg_model_map_gal.values[1:-1])
    # plot_skymap(Bg_model_map_eq_norm, args.resultdir, args.plotdir, f'{args.outputprefix}_background_model_equatorial_normalized',title= f"{args.title} BG Model", scale = 'lin', vmin = 0, vmax = 1, normalization='rel_background', rotate='C', transparent=args.transparent)#, mask_region=get_regions(background_region), invert_mask=False)
    # plot_skymap(Bg_model_map_gal_norm, args.resultdir, args.plotdir, f'{args.outputprefix}_background_model_normalized',title= f"{args.title} BG Model", scale = 'lin', vmin = 0, vmax = 1, normalization='rel_background', rotate='G', transparent=args.transparent)#, mask_region=get_regions(background_region), invert_mask=False)
    # if args.no_title:
    #     plot_skymap(Bg_model_map_eq_norm, args.resultdir, args.plotdir, f'{args.outputprefix}_background_model_equatorial_normalized_NoTitle',title= None, scale = 'lin', vmin = 0, vmax = 1, normalization='rel_background', rotate='C', transparent=args.transparent)#, mask_region=get_regions(background_region), invert_mask=False)
    #     plot_skymap(Bg_model_map_gal_norm, args.resultdir, args.plotdir, f'{args.outputprefix}_background_model_normalized_NoTitle',title= None, scale = 'lin', vmin = 0, vmax = 1, normalization='rel_background', rotate='G', transparent=args.transparent)#, mask_region=get_regions(background_region), invert_mask=False)


    Full_model_map_gal = Histogram(*Bg_model_map_gal.binnings)
    Full_model_map_gal.add(Bg_model_map_gal)
    Full_model_map_gal.add(Source_100_100)
    Full_model_map_gal.add(Diffuse_100_100)

    lat_mask = np.abs(declanation) <= 8
    lat_zero_mask = np.zeros(len(declanation))
    lat_zero_mask[lat_mask] = np.ones(len(declanation))[lat_mask]

    
    longitude = hp.pix2ang(nside,pix, lonlat=True)[0]
    longitude_set = np.array(sorted(list(set(longitude))))
    

    lon_binning = make_lin_binning(0, 360, 180)
    
    
    Data_gal_plane = Histogram(lon_binning)#np.zeros(len(longitude_set))
    Data_gal_plane_error = Histogram(lon_binning)#np.zeros(len(longitude_set))
    Source_gal_plane = Histogram(lon_binning)#np.zeros(len(longitude_set))
    Diffuse_gal_plane = Histogram(lon_binning)#np.zeros(len(longitude_set))
    BG_gal_plane = Histogram(lon_binning)#np.zeros(len(longitude_set))
    Full_gal_plane = Histogram(lon_binning)

    for i, lon in enumerate(longitude_set):
        lon_mask = longitude == lon
        lonlat_mask = lon_mask & lat_mask
        bin_id = lon_binning.get_indices([lon])
        Data_gal_plane.values[bin_id] += np.sum(Skymap_100_100.values[1:-1][lonlat_mask])
        Data_gal_plane_error.values[bin_id] += np.sqrt(np.sum(np.square(np.sqrt(Skymap_100_100.values[1:-1][lonlat_mask]))))
        Source_gal_plane.values[bin_id] += np.sum(Source_100_100.values[1:-1][lonlat_mask])
        Diffuse_gal_plane.values[bin_id] += np.sum(Diffuse_100_100.values[1:-1][lonlat_mask])
        BG_gal_plane.values[bin_id] += np.sum(Bg_model_map_gal.values[1:-1][lonlat_mask])
        Full_gal_plane.values[bin_id] += np.sum(Full_model_map_gal.values[1:-1][lonlat_mask])


    lon_centers = lon_binning.bin_centers[1:-1]
    lon_centers = np.array([x if x <180 else x%180-180 for x in lon_centers])
    sort_id = lon_centers.argsort()
    lon_centers = lon_centers[sort_id[::-1]]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.scatter(lon_centers, Data_gal_plane.values[1:-1][sort_id[::-1]], marker="o", ls='', s=1, label = 'Data', zorder = 5)
    ax.plot(lon_centers, Source_gal_plane.values[1:-1][sort_id[::-1]], label = 'Source', color = 'cyan')
    ax.plot(lon_centers, Diffuse_gal_plane.values[1:-1][sort_id[::-1]], label = 'Diffuse', color = 'green')
    ax.plot(lon_centers, BG_gal_plane.values[1:-1][sort_id[::-1]], label = 'BG', color = 'magenta')
    ax.plot(lon_centers, Full_gal_plane.values[1:-1][sort_id[::-1]], label = 'Full', zorder = 0, color = 'red')
    ax.set_xlabel("Galactic Longitude / deg")
    ax.set_ylabel("Events")
    ax.legend()
    ax.xaxis.set_inverted(True)
    save_figure(fig, args.plotdir, f"{args.outputprefix}_Model_validation_galactic_plane", transparent=args.transparent)

    fig, ax = plt.subplots(figsize=(8,4.2))
    ax.scatter(lon_centers, (Data_gal_plane.values[1:-1][sort_id[::-1]] - Full_gal_plane.values[1:-1][sort_id[::-1]])/Full_gal_plane.values[1:-1][sort_id[::-1]], marker='o')
    ax.set_xlabel("Galactic Longitude / deg")
    ax.set_ylabel("Data-Model/Model / Events")
    ax.xaxis.set_inverted(True)
    save_figure(fig, args.plotdir, f"{args.outputprefix}_Model_validation_galactic_plane_residuum", transparent=args.transparent)



            



    # residuum_map = load_histograms_from_files([args.skymap])['smoothed']
    # for hist in other_maps.values():
    #     residuum_map.values = residuum_map.values - hist.values

    # total_events = Skymap.values.sum()
    # total_events_r = residuum_map.values.sum()
    # title_r = f"{args.title} Residuum: {total_events_r:.0f}/{total_events:.0f} events"
    # if args.no_title:
    #     title_r = None


    

    # print(residuum_map.values)
    # print(np.min(residuum_map.values))
    # print(np.max(residuum_map.values))
    # v = np.max([np.abs(np.min(residuum_map.project_axis(axis=0).values)), np.abs(np.max(residuum_map.project_axis(axis=0).values))])
    # plot_skymap(residuum_map.project_axis(axis=0), resultdir=args.resultdir, plotdir=args.plotdir, outputprefix=f"{args.outputprefix}_residuum", title=title_r, colormap=args.residuum_colormap, vmin=-v, vmax=v, rotate=args.coord, save_pickle=args.save_pickle, normalization=args.normalization, transparent=args.transparent, scale='lin')
    # for energy_bin, (sky_histogram_1d, energy_min, energy_max) in enumerate(residuum_map.project_all(axis=0, include_overflow=True)):
    #             v = np.max([np.abs(np.min(sky_histogram_1d.values)),np.abs(np.max(sky_histogram_1d.values))])
    #             plot_skymap(sky_histogram_1d, resultdir=args.resultdir, plotdir=args.plotdir, outputprefix=f"{args.outputprefix}_residuum_{energy_bin}", title=f"{args.title} Residuum:, {energy_min:.4g}<=E/GeV<{energy_max:.4g}", colormap=args.residuum_colormap, vmin=-v, vmax=v, rotate=args.coord, save_pickle=args.save_pickle, normalization=args.normalization, transparent=args.transparent, scale='lin')


    # ratio_map = load_histograms_from_files([args.skymap])['smoothed']
    # # for i, (hist, _, _) in enumerate(ratio_map.project_all(axis=0, include_overflow=True)):
    # #     ratio_map.values[i] = hist.values/hist.values.sum()

    # ratio_other_hist = other_maps[list(other_maps.keys())[0]]
    # ratio_other_hist.values = ratio_other_hist.values
    # for i,hist in enumerate(other_maps.values()):
    #     if i == 0:
    #         continue
    #     # for i, (histo, _, _) in enumerate(hist.project_all(axis=0, include_overflow=True)):
    #     #     hist.values[i] = histo.values/histo.values.sum()
    #     ratio_other_hist.add(hist)

    # # print(ratio_other_hist.values)
    # ratio_map.values = ratio_map.values/ratio_other_hist.values
    # ratio_map.values = np.nan_to_num(ratio_map.values, nan= 0)
    # # print(ratio_map.values)
    # # print(np.min(ratio_map.values))
    # # print(np.max(ratio_map.values))

    # plot_skymap(ratio_map.project_axis(axis=0), resultdir=args.resultdir, plotdir=args.plotdir, outputprefix=f"{args.outputprefix}_ratio", title=title_r, colormap=args.residuum_colormap, vmin=1e-3, vmax=1e3, rotate=args.coord, save_pickle=args.save_pickle, normalization=args.normalization, transparent=args.transparent, scale='log')
    # for energy_bin, (sky_histogram_1d, energy_min, energy_max) in enumerate(ratio_map.project_all(axis=0, include_overflow=True)):
    #             plot_skymap(sky_histogram_1d, resultdir=args.resultdir, plotdir=args.plotdir, outputprefix=f"{args.outputprefix}_ratio_{energy_bin}", title=f"{args.title} Ratio:, {energy_min:.4g}<=E/GeV<{energy_max:.4g}", colormap=args.residuum_colormap, vmin=1e-3, vmax=1e3, rotate=args.coord, save_pickle=args.save_pickle, normalization=args.normalization, transparent=args.transparent, scale='log')








if __name__ == "__main__":
    main()