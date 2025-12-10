#!/usr/bin/env python

"""SORTS package"""
import ctypes
import pathlib
import sysconfig
import logging
from .version import __version__

# Get and config module-level logger
logger = logging.getLogger(__name__)

# Find suffix
# suffix = sysconfig.get_config_var("EXT_SUFFIX")
# if suffix is None:
#     suffix = ".so"
#
# __sortspath__ = pathlib.Path(__file__).resolve().parent
# __libpath__ = __sortspath__ / ("clibsorts" + suffix)
#
# clibsorts = ctypes.cdll.LoadLibrary(str(__libpath__))

##
# v2 imports
##
from . import types
from . import utils
from . import scheduling
from . import controller
from . import simulation

from .scheduling import Schedule, ExperimentDetail
from .controller import TrackerController, FenceScanController
from .simulation import StxMrxSimulation
from .mpi_queued_execution import MpiQueuedExecution


##
# v1 imports
##
# TODO: clean up these imports

# modules
from .radar import scans
from . import radar
from . import functions
from . import constants
from . import frames
from . import dates
from . import plotting

# from . import controller_v1
from . import scheduler
from . import passes
from . import errors
from . import io
from . import interpolation

# from . import simulation_v1
from . import signals
from . import correlator
from . import propagator

# classes
from .space_object import SpaceObject
from .population import Population
from .propagator import Propagator
from .radar import Scan
from .radar import Station, TX, RX
from .controller_v1 import RadarController
from .scheduler import Scheduler
from .passes import Pass
from .errors import Errors
from .simulation_v1 import Simulation

# Functions
from .radar import get_radar, list_radars
from .correlator import correlate
from .passes import equidistant_sampling
# from .passes import find_passes, find_simultaneous_passes, group_passes
from .signals import hard_target_snr
from .simulation_v1 import (
    MPI_single_process,
    MPI_action,
    iterable_step,
    store_step,
    cached_step,
    iterable_cache,
    pre_post_actions,
)
from .pre_encounter import propagate_pre_encounter, distance_termination
