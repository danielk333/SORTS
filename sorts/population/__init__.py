#!/usr/bin/env python

'''

'''

from .population import Population

from .master import master_catalog
from .master import master_catalog_factor

def __getattr__(name):
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
