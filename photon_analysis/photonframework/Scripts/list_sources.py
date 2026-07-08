#!/usr/bin/env python3

import os
from xml.etree import ElementTree

from astropy import units
from astropy.coordinates import SkyCoord

def load_sources(catalogue_filename, longitude_range=None, latitude_range=None, threshold=None):
    lon_min, lon_max = 0, 360
    lat_min, lat_max = -90, 90
    npred_min = 0
    if longitude_range is not None:
        lon_min, lon_max = longitude_range
    if latitude_range is not None:
        lat_min, lat_max = latitude_range
    if threshold is not None:
        npred_min = threshold

    sources = []

    xml_tree = ElementTree.parse(catalogue_filename)
    root = xml_tree.getroot()
    for source in root.iter("source"):
        npred = float(source.attrib["Npred"])
        ra = None
        dec = None
        for spatial in source.iter("spatialModel"):
            for param in spatial.iter("parameter"):
                if param.attrib["name"] == "RA":
                    ra = float(param.attrib["value"])
                elif param.attrib["name"] == "DEC":
                    dec = float(param.attrib["value"])
            break
        if ra is None or dec is None:
            continue
        coord = SkyCoord(ra, dec, unit=units.deg, frame="fk5")
        lonlat = coord.galactic
        lon = lonlat.l.value
        lat = lonlat.b.value
        if lon >= lon_min and lon <= lon_max and lat >= lat_min and lat <= lat_max and npred >= npred_min:
            sources.append((source.attrib["name"], lon, lat, ra, dec, npred))

    return sorted(sources, key=lambda t: -t[5])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("catalogue", help="Path to the source catalogue XML file.")
    parser.add_argument("--longitude", type=float, nargs=2, help="Minimum and maximum longitude.")
    parser.add_argument("--latitude", type=float, nargs=2, help="Minimum and maximum latitude.")
    parser.add_argument("--npred", type=float, help="Npred threshold")

    args = parser.parse_args()

    sources = load_sources(args.catalogue, longitude_range=args.longitude, latitude_range=args.latitude, threshold=args.npred)

    for name, lon, lat, ra, dec, npred in sources:
        print(f"{name:<20} {lon:>9.4f} {lat:>8.4f}  {ra:>9.4f} {dec:>8.4f} {npred:>10.2f}")


if __name__ == "__main__":
    main()
