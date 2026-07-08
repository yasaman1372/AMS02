
import numpy as np
import healpy as hp
from regions import Regions, EllipseSkyRegion, PointSkyRegion, CircleSkyRegion, PolygonSkyRegion, RectangleSkyRegion
import astropy.units as u
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

from tools.coordinates import convert_galactic_to_equatorial_coordinates


def mask_sources(Map, region, mask_type: bool = True, inclusive: bool = True, return_indices: bool = False):
    nside = hp.npix2nside(len(Map))
    if return_indices:
        indices = []
    if mask_type:
        mask = np.zeros(len(Map), dtype=bool)
    else:
        mask = np.ones(len(Map), dtype=bool)
    for source in region:

        if source[0] == 'Strip':
            Theta1, Theta2 = np.radians([-source[1][1]+90, -source[1][0]+90])
            pix = hp.query_strip(nside, Theta1, Theta2, inclusive)
            mask[pix] = mask_type
            if return_indices:
                indices.append(pix)

        elif source[0] in ('Rectangle', 'Rect'):
            Phi, Theta, deltaPhi, deltaTheta = np.radians([-source[1][0], -source[1][1] + 90, source[1][2], source[1][3]])
            vertices = [(Phi - deltaPhi, Theta - deltaTheta),
                        (Phi - deltaPhi, Theta + deltaTheta),
                        (Phi + deltaPhi, Theta + deltaTheta),
                        (Phi + deltaPhi, Theta - deltaTheta)]
            vectors = [hp.ang2vec(theta, phi) for phi, theta in vertices]
            pix = hp.query_polygon(nside, vectors, inclusive = inclusive)
            mask[pix] = mask_type
            if return_indices:
                indices.append(pix)

        elif source[0] == 'Polygon':
            vertices = []
            for i in range(int(len(source[1])/2)):
                vertices.append(np.radians([-source[1][i*2+0], -source[1][i*2+1]+90]))
            vectors = [hp.ang2vec(theta, phi) for phi, theta in vertices]
            pix = hp.query_polygon(nside, vectors, inclusive = inclusive)
            mask[pix] = mask_type
            if return_indices:
                indices.append(pix)

        elif source[0] == 'Disc':
            Phi, Theta, R = np.radians([-source[1][0], -source[1][1] + 90, source[1][2]])
            vec = hp.ang2vec(Theta, Phi)
            pix = hp.query_disc(nside, vec, R, inclusive= inclusive)
            mask[pix] = mask_type
            if return_indices:
                indices.append(pix)

        else:
            raise NotImplementedError('Please only use "Disc", "Polygon", "Strip" or "Rectangle" as a mask type.')

    if return_indices:
        return np.concatenate(indices)
    return np.ma.masked_array(Map, mask=mask)


def ellipse_to_healpy_polygon(region, n_points=100):
    if not isinstance(region, EllipseSkyRegion):
        raise ValueError("Provided region is not an ellipse")

    c_l = region.center.l.deg
    c_b = region.center.b.deg
    a = region.width.to(u.deg).value / 2
    b = region.height.to(u.deg).value / 2
    angle = region.angle.to(u.rad).value

    theta_vals = np.linspace(0, 2 * np.pi, n_points)
    x = a * np.cos(theta_vals)
    y = b * np.sin(theta_vals)

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    x_rot = cos_angle * x - sin_angle * y
    y_rot = sin_angle * x + cos_angle * y

    l = c_l + x_rot
    b = c_b.b.deg + y_rot

    l = np.mod(l, 360)
    l = np.where(l < 180, -l, 360-l)

    return np.array([l, b]).T.flatten()


def rectangle_to_healpy_polygon(region, height=None):
    if not isinstance(region, RectangleSkyRegion):
        raise ValueError("Provided region is not a rectangle")

    center_gal = region.center
    width = region.width.to(u.deg).value / 2
    if height is None:
        height = region.height.to(u.deg).value / 2
    angle = region.angle.to(u.rad).value
    if center_gal.l.deg == 0 and center_gal.b.deg == 0 and width == 180:
        return ('Strip', (-height, height))

    corners_x = np.array([-width, width, width, -width])
    corners_y = np.array([-height, -height, height, height])

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    x_rot = cos_angle * corners_x - sin_angle * corners_y
    y_rot = sin_angle * corners_x + cos_angle * corners_y

    l = center_gal.l.deg + x_rot
    b = center_gal.b.deg + y_rot
    l = l+180
    l = np.mod(l, 360)
    l = np.where(l <= 180, -l, 360 - l)
    return ('Polygon', tuple(np.array([l, b]).T.flatten()))



def get_regions(region_file, format='ds9'):
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
            healpy_reg = ('Disc', (l, b, r))
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
            healpy_reg = rectangle_to_healpy_polygon(reg)
        Regs.append(healpy_reg)

    return Regs

def get_sources(region_file, format='ds9'):
    Sources = {'_':[]}
    SkyRegion = Regions.read(region_file, format=format)
    for reg in SkyRegion:
        if isinstance(reg, CircleSkyRegion):
            l = reg.center.l.deg
            if l < 180:
                l = -l
            elif l >= 180:
                l = 360 - l
            b = reg.center.b.deg
            r = reg.radius.to(u.deg).value
            healpy_reg = ['Disc', (l, b, r)]
        elif isinstance(reg, PolygonSkyRegion):
            vertices = []
            for v in reg.vertices:
                l = v.l.deg
                if l < 180:
                    l = -l
                elif l >= 180:
                    l = 360-l
                b = v.b.deg
                vertices.append(l)
                vertices.append(b)
            healpy_reg = ['Polygon', tuple(vertices)]
        elif isinstance(reg, EllipseSkyRegion):
            vertices = ellipse_to_healpy_polygon(reg)
            healpy_reg = ['Polygon', tuple(vertices)]
        elif isinstance(reg, RectangleSkyRegion):
            vertices = rectangle_to_healpy_polygon(reg)
            healpy_reg = ['Polygon', tuple(vertices)]
        elif isinstance(reg, PointSkyRegion):
            l = reg.center.l.deg
            if l < 180:
                l = -l
            elif l >= 180:
                l = 360-l
            b = reg.center.b.deg
            healpy_reg = ['Point', (l, b)]

        meta = reg.meta
        visual = reg.visual
        if 'edgecolor' in visual:
            healpy_reg.append(meta['edgecolor'])

        if 'text' in meta:
            name = meta['text']
            Sources[name] = healpy_reg
        else:
            name = '_'
            Sources[name].append(healpy_reg)

    return Sources


def plot_sources_on_map(plot, source_file_list):
    Source_list = []
    for file in source_file_list:
        Source_list.append(get_sources(file))
    for Sources in Source_list:
        for name, source in Sources.items():
            if source[0] == 'Point':
                plot.scatter(np.radians(source[1][0]), np.radians(source[1][1]), s=3, c=source[2], marker='x', label = name)
            elif source[0] == 'Circle':
                source_circle = Circle(np.radians(source[1][0:2]), np.radians(source[1][2]), color = source[2], fill = False, linewidth = 1)
                plot.add_patch(source_circle)
            elif source[0] == 'Polygon':
                xy = list(source[1])
                xy.append(source[1][0:2])
                x = np.array(xy).T[0]
                y = np.array(xy).T[1]
                plot.plot(x, y, marker = '', ls = '', color = source[2], label = name)

    plot.legend(loc = 'lower right', fontsize= 'xx-small', labelcolor='mfc')


def count_pixels_in_region(nside, region, mask_type=True, inclusive=True, latitude_range=None, longitude_range=None, dec_range=None):
    pixel_indices = np.arange(hp.nside2npix(nside))
    masked_indices = mask_sources(pixel_indices, region, mask_type=mask_type, inclusive=inclusive)
    longitude, latitude = hp.pix2ang(nside, masked_indices, lonlat=True)
    if longitude_range is not None:
        masked_indices = np.ma.masked_where((longitude < longitude_range[0]) | (longitude >= longitude_range[1]), masked_indices)
    if latitude_range is not None:
        masked_indices = np.ma.masked_where((latitude < latitude_range[0]) | (latitude >= latitude_range[1]), masked_indices)
    if dec_range is not None:
        ra, dec = convert_galactic_to_equatorial_coordinates(latitude, longitude)
        masked_indices = np.ma.masked_where((dec < dec_range[0]) | (dec > dec_range[1]), masked_indices)
    return masked_indices.count()


