#!/usr/bin/env python

'''TLE list loading to population

'''
import pathlib

import numpy as np
from astropy.time import Time
import pyorb

from ..propagator import SGP4
from .population import Population

def tle_catalog(
        tles,
        sgp4_propagation = True, 
        propagator = None,
        propagator_options = {},
        propagator_args = {},
    ):
    '''Reads a TLE-snapshot file and converts the TLE's to orbits in a TEME frame and creates a population file. 
    A snapshot generally contains several TLE's for the same object thus will this population also contain duplicate objects.
    The BSTAR parameter is saved in field BSTAR :code:`BSTAR`. 

    *Numerical propagator assumptions:*
    To propagate with a numerical propagator one needs to make assumptions.
       * Density is :math:`5\cdot 10^3 \;\frac{kg}{m^3}`.
       * Object is a sphere
       * Drag coefficient is 2.3.

    :param str/list tles: Path to the input TLE snapshot file. Or the TLE-set can be given directly as a list of two lines that can be unpacked in a loop, e.g. :code:`[(tle1_l1, tle1_l2), (tle2_l1, tle2_l2)]`.

    :return: TLE snapshot as a Population
    :rtype: sorts.Population
    '''
    if isinstance(tles, str) or isinstance(tles, pathlib.Path):
        tle_raw = [line.rstrip('\n') for line in open(tles)]
        if len(tle_raw) % 2 != 0:
            raise Exception('Not even number of lines [not TLE compatible]')

        tle_size = len(tle_raw)//2

        tles = zip(tle_raw[0::2], tle_raw[1::2])

    prop = SGP4(
        settings = dict(
            out_frame='TEME',
            tle_input=True,
        ),
    )

    if sgp4_propagation:
        pop = Population(
            fields = ['oid','x','y','z','vx','vy','vz','mjd0', 'A', 'm', 'd', 'C_D', 'C_R', 'BSTAR', 'line1', 'line2'],
            dtypes = ['int'] + ['float64']*13 + ['U70']*2,
            space_object_fields = [],
            state_fields = ['line1', 'line2'],
            epoch_field = {'field': 'mjd0', 'format': 'mjd', 'scale': 'utc'},
            propagator = SGP4,
            propagator_options = dict(
                settings = dict(
                    out_frame='TEME',
                    tle_input=True,
                ),
            ),
        )
    else:
        pop = Population(
            fields = ['oid','x','y','z','vx','vy','vz','mjd0', 'A', 'm', 'd', 'C_D', 'C_R', 'BSTAR'],
            dtypes = ['int'] + ['float64']*13,
            space_object_fields = ['A', 'm', 'd', 'C_D', 'C_R', 'BSTAR'],
            state_fields = ['x','y','z','vx','vy','vz'],
            epoch_field = {'field': 'mjd0', 'format': 'mjd', 'scale': 'utc'},
            propagator = propagator,
            propagator_options = propagator_options,
            propagator_args = propagator_args,
        )
    
    pop.allocate(tle_size)

    for line_id, lines in enumerate(tles):
        line1, line2 = lines

        params = SGP4.get_TLE_parameters(line1, line2)
        bstar = params['bstar']
        epoch = Time(params['jdsatepoch'] + params['jdsatepochF'], format='jd', scale='utc').mjd
        oid = params['satnum']

        state_TEME = prop.propagate([0.0], lines)

        pop.data[line_id]['x'] = state_TEME[0]
        pop.data[line_id]['y'] = state_TEME[1]
        pop.data[line_id]['z'] = state_TEME[2]
        pop.data[line_id]['vx'] = state_TEME[4]
        pop.data[line_id]['vy'] = state_TEME[3]
        pop.data[line_id]['vz'] = state_TEME[5]
        
        pop.data[line_id]['oid'] = oid
        pop.data[line_id]['mjd0'] = epoch

        if sgp4_propagation:
            pop.data[line_id]['line1'] = line1
            pop.data[line_id]['line2'] = line2
    
        bstar = bstar/(prop.grav_model.radiusearthkm*1000.0)
        B = bstar*2.0/prop.rho0
        if B < 1e-9:
            rho = 500.0
            C_D = 0.0
            r = 0.1
            A = np.pi*r**2
            m = rho*4.0/3.0*np.pi*r**3
        else:
            C_D = 2.3
            rho = 5.0
            r = (3.0*C_D)/(B*rho)
            A = np.pi*r**2
            m = rho*4.0/3.0*np.pi*r**3

        pop.data[line_id]['A'] = A
        pop.data[line_id]['m'] = m
        pop.data[line_id]['d'] = r*2.0
        pop.data[line_id]['C_D'] = C_D
        pop.data[line_id]['C_R'] = 1.0

    return pop

