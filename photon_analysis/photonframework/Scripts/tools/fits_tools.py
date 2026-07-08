#!/usr/bin/env python3

from astropy.io import fits
from astropy.io.fits.hdu.base import _BaseHDU
import numpy as np

def number_of_extensions(filename):
    with fits.open(filename) as fits_file:
        nExtentions = len(fits_file)
    return nExtentions

def has_extension(filename, extension):
    with fits.open(filename) as fits_file:
        result = extension in fits_file
    return result

def read_fits_file(filename, extension=None):
    fits_file = fits.open(filename)
    if extension != None:
        return fits_file[extension]
    
    primaryHeader = fits_file[0].header
    if primaryHeader['NAXIS'] == 0 and primaryHeader['EXTEND']:
        return fits_file[1]
    return fits_file[0]


