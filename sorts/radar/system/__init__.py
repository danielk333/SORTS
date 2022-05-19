#!/usr/bin/env python

'''Defines the required properties of a radar system and its components, including predefined instances.

'''

from .station import Station, TX, RX
from .radar import Radar

from .instances import RadarSystemsGetter

instances = RadarSystemsGetter()