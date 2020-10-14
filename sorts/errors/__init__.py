#!/usr/bin/env python

'''Contains functions related to error calculation and a generalized method of generating random noisy data.

'''

from .errors import Errors

from .linearized_coded import LinearizedCodedIonospheric
from .linearized_coded import LinearizedCoded

from .ionospheric_ray_trace import calculate_delay, ray_trace, ray_trace_error, ionospheric_error
from .ionospheric_ray_trace import IonosphericRayTrace


from . import atmospheric_drag

from .linearized_orbit_determination import covariance_propagation, orbit_determination_covariance