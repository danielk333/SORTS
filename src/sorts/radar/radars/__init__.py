"""Defines all the pre-configured radar instances

"""

from . import eiscat_3d
from . import tsdr
from . import eiscat_uhf
from . import eiscat_esr
from . import nostra
from . import mock

from .radars import get_radar, list_radars
