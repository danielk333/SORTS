#!/usr/bin/env python

'''Minimoon population

'''
import pathlib

import numpy as np
from astropy.time import Time
import pyorb

from .population import Population


def NESCv9_minimoons(
                fname, albedo, 
                propagator = None,
                propagator_options = {},
                propagator_args = {},
            ):
    '''
    input file data format: 

    1. synthetic orbit designation
    2. orbital element type (KEP for heliocentric Keplerian)
    3. semimajor axis (au)
    4. eccentricity
    5. inclination (deg)
    6. longitude of ascending node (deg)
    7. argument of perihelion (deg)
    8. mean anomaly (deg)
    9. H-magnitude (filler value)
    10. epoch for which the input orbit is valid (modified Julian date)
    11. index (1)
    12. number of parameters (6)
    13. MOID (filler value)
    14. Code with which created (OPENORB)
    '''

    data = np.genfromtxt(fname)
    pop = Population(
        fields = ['oid', 'a', 'e', 'i', 'aop', 'raan', 'mu0', 'mjd0', 'd'],
        dtypes = ['int'] + ['float64']*8,
        space_object_fields = ['d'],
        state_fields = ['a', 'e', 'i', 'aop', 'raan', 'mu0'],
        epoch_field = {'field': 'mjd0', 'format': 'mjd', 'scale': 'utc'},
        propagator = propagator,
        propagator_options = propagator_options,
        propagator_args = propagator_args,
    )

    pop.allocate(data.shape[0])

    pop['oid'] = np.arange(len(pop))
    pop['a'] = data[:, 2]*pyorb.AU
    pop['e'] = data[:, 3]
    pop['i'] = data[:, 4]
    pop['raan'] = data[:, 5]
    pop['aop'] = data[:, 6]
    pop['mu0'] = data[:, 7]
    pop['mjd0'] = data[:, 9]
    pop['d'] = 10.0**(3.1236 - 0.5*np.log10(albedo) - 0.2*data[:, 8])

    return pop
