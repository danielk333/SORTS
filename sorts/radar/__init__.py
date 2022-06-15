#!/usr/bin/env python

'''Defines the required properties of a radar system and its components, including predefined instances.

'''

# -----------------------------------------------------------------
#                           Radar
# -----------------------------------------------------------------
# Schedule
from . import scheduler
from . import controllers

from .scheduler import Scheduler, TrackingScheduler
from .controllers import RadarController, Scanner, Static, Tracker

# Control manager
from .controls_manager import base
from .controls_manager import simple_manager

from .controls_manager.base import RadarControlManagerBase
from .controls_manager.simple_manager import SimpleRadarControl

# System
from .system import RX, TX, Station
from .system import Radar

from .system import radar
from .system import station

# Radar instances
from .system import instances as radars

# Other radar imports
from . import signals
from . import scans
from . import passes
from . import measurement_errors

from .measurement_errors import IonosphericRayTrace, LinearizedCoded, LinearizedCodedIonospheric, Errors

from .passes import Pass

from .passes import equidistant_sampling
from .passes import find_passes, find_simultaneous_passes
from .signals import hard_target_snr