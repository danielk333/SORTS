#!/usr/bin/env python

"""
Loading a TLE catalog
=============================================
"""

import pathlib
import argparse

import matplotlib.pyplot as plt
import numpy as np
from sorts import plotting
from sorts.population import tle_catalog

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    default=None,
    help="Path to TLE catalog, otherwise use predefined one",
)
args = parser.parse_args()

if args.path is None:
    l1 = "1     5U 58002B   20251.29381767 +.00000045 +00000-0 +68424-4 0  9990"
    l2 = "2     5 034.2510 336.1746 1845948 000.5952 359.6376 10.84867629214144"
    pop = tle_catalog([(l1, l2)], save_mean_elements=True)
else:
    pop = tle_catalog(args.path, save_mean_elements=True)

first_elemts = pop.print(
        row_indecies=0,
        fields=["id", "a", "e", "i", "epoch", "B"],
    )
print(first_elemts)

fig, ax =plotting.kepler_orbit(
    pop.get_orbit(),
)
ax.set_title("Orbit distribution of tle catalog")

# look at kepler elements
orbit = pop.get_orbit(row_indecies=0)
print(f'\n Orbit of satnum: {pop.data["id"][0]} \n{str(orbit)}\n')

# we can also create a space object
obj = pop.get_object(index=0)
print(obj)

plt.show()
