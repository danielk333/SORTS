"""Defines all the pre-configured radar instances"""

# from . import eiscat_3d
# from . import tsdr
# from . import eiscat_esr
from . import (
    nostra as nostra,
    eiscat_uhf as eiscat_uhf,
)

# from . import mock

from .radars import (
    get_radar as get_radar,
    list_radars as list_radars,
)
