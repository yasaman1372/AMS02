#!/usr/bin/env python3

import astropy as ap
from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np
from regions import Regions

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--region-file', required=True, help='The region file with to coordinates that schould be transformed')
    parser.add_argument("--format", default='ds9', choices=["ds9", 'fits', 'crtf'], help='The Format of the old and ny regions file')
    parser.add_argument('--frame', default='galactic', choices=("fk4", "fk5", "icrs", "galactic"), help="The Frame to convert into")
    parser.add_argument('--new-region-file', help="Name of the new file, if None the old file will be replaced")

    args = parser.parse_args()

    SkyRegions = Regions.read(args.region_file, format=args.format)

    for region in SkyRegions:
        region.center = region.center.transform_to(args.frame)

    if args.new_region_file is not None:
        SkyRegions.write(f"{args.new_region_file}", format=args.format ,overwrite=True)
    else:
        SkyRegions.write(f"{args.region_file}", format=args.format ,overwrite=True)

if __name__ == "__main__":
    main()