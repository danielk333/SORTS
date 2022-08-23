#!/usr/bin/env python

'''Defines the required properties of a radar system and its components, including predefined instances.

'''

# -----------------------------------------------------------------
#                           Radar
# -----------------------------------------------------------------
# Controls
from . import controllers
from . import radar_controls

from .radar_controls import RadarControls
from .controllers import RadarController, Scanner, Static, Tracker, SpaceObjectTracker

# Control manager
from .scheduler import base
from .scheduler import static_priority_scheduler

from .scheduler.base import RadarSchedulerBase
from .scheduler.static_priority_scheduler import StaticPriorityScheduler

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

# measurements
from .measurements import measurement
from .measurements import Measurement