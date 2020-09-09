#!/usr/bin/env python

'''
Loading a TLE catalog (e.g. spacetrack)
=============================================
'''
import pathlib

import matplotlib.pyplot as plt

from sorts import plotting
from sorts.population import tle_catalog

pth = pathlib.Path(__file__).parent / 'data' / 'space_track_tle.txt'

pop = tle_catalog(pth)

print(pop.print(n=slice(None,10), fields = ['oid','x','y','z','vx','vy','vz','mjd0', 'A', 'm', 'd', 'C_D', 'C_R', 'BSTAR']))

plotting.orbits(
    pop.get_fields(['x','y','z','vx','vy','vz'], named=False),
    title =  "Orbit distribution of Spacetrack catalog",
)

plt.show()
