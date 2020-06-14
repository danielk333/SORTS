#!/usr/bin/env python

'''rapper for the SGP4 propagator

'''

#Third party import
import scipy.constants


R_earth = 6371e3
'''float: Radius of the Earth
'''

MU_earth = 398600.8*1e9
'''float: Standard gravitational parameter of the Earth using the WGS72 convention.
'''

M_earth = MU_earth/scipy.constants.G
'''float: Mass of the Earth using the WGS72 convention.
'''


