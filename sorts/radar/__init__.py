#!/usr/bin/env python

'''

'''

from .scans import Scan
from .tx_rx import Station, TX, RX

from . import instances

def __getattr__(name):
    if name in instances.radar_instances:
        return getattr(instances, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
