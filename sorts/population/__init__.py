#!/usr/bin/env python

'''Package that defines populations and methods for loading them from different data sources.

'''

from .population import Population

from .master import master_catalog
from .master import master_catalog_factor

from .tles import tle_catalog

from .minimoon import NESCv9_minimoons