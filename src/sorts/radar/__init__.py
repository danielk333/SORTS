#!/usr/bin/env python

"""Defines the required properties of a radar system and its components, including predefined instances.

"""

from .scans import Scan
from .tx_rx import Station, TX, RX

from .instances import RadarSystemsGetter

instances = RadarSystemsGetter()
