import numpy as np
import numpy.typing as npt
from astropy.time import Time
from sorts.radar.radars import get_radar
from sorts.controller.fence_scan_controller import FenceScanController
from sorts.schedule import ScheduleOld


def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()


# TODO: maybe a test for fn `generate_from_state`, `pointing_patterns.fence_pointing`
# TODO: should test station min_elevation are taken into account
# TODO: should test if tx entry is masked, all corresponding rx entries are masked
