#!/usr/bin/env python

''' SORTS package

'''

from .version import __version__

import os
import ctypes

__sortspath__ = os.path.dirname(__file__)
__libpath__ = __sortspath__ + "/../libsorts.so"

clibsorts = ctypes.cdll.LoadLibrary(__libpath__)

# -----------------------------------------------------------------
#                           Radar
# -----------------------------------------------------------------
# Schedule
from .radar import scheduler
from .radar import controllers

from .radar.scheduler import ObservedParameters, StaticList, Scheduler, PointingSchedule, Tracking, PriorityTracking
from .radar.controllers import RadarController, Scanner, Static, Tracker

# System
from .radar.system import RX, TX, Station
from .radar.system import Radar

#from .radar.system import station

# Radar instances
from .radar.system import instances as radars

# Other radar imports
from .radar import signals
from .radar import scans
from .radar import passes
from .radar import measurement_errors

from .radar.measurement_errors import IonosphericRayTrace, LinearizedCoded, LinearizedCodedIonospheric, Errors

from .radar.passes import Pass

from .radar.passes import equidistant_sampling
from .radar.passes import find_passes, find_simultaneous_passes
from .radar.signals import hard_target_snr

# -----------------------------------------------------------------
#                               IO 
# -----------------------------------------------------------------
from . import io

from .io import ccsds
from .io import terminal

# -----------------------------------------------------------------
#                            Plotting 
# -----------------------------------------------------------------
from . import plotting

# -----------------------------------------------------------------
#                            Transformations 
# -----------------------------------------------------------------
from .transformations import frames
from .transformations import dates

# -----------------------------------------------------------------
#                            Correlator 
# -----------------------------------------------------------------
from . import correlator

from .correlator import correlate

# -----------------------------------------------------------------
#                            Simulation 
# -----------------------------------------------------------------
from . import simulation

from .simulation import Simulation
from .simulation import MPI_single_process, MPI_action, iterable_step, store_step, cached_step, iterable_cache


# -----------------------------------------------------------------
#                            Common 
# -----------------------------------------------------------------
from . import common

from .common import constants
from .common import functions
from .common import interpolation 
from .common import profiling

from .common import Profiler

# -----------------------------------------------------------------
#                           Targets
# -----------------------------------------------------------------
from .targets import SpaceObject, Population, Propagator

from .targets import propagator
from .targets import space_object
from .targets import propagation_errors

from .targets.population import base
from .targets.population import master
from .targets.population import tles

# -----------------------------------------------------------------
#                            Others 
# -----------------------------------------------------------------
from . import linearized_orbit_determination

