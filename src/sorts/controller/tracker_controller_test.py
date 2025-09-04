import numpy as np
import numpy.typing as npt
from astropy.time import Time
from sorts.propagator import SGP4
from sorts.space_object import SpaceObject
from sorts.radar.radars import get_radar
from sorts.types import Datetime64_us, Timedelta64_us, Float64_as_sec
from sorts.controller import tracker_controller


def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()


# TODO: maybe a test for fn `generate_from_state`
# TODO: should test station min_elevation are taken into account
