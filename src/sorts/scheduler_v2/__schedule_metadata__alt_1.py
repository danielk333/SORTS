"""
NOTE: this is tmp kept as a reference alternative implementation of 'schedule_metadata.py'

TODO: remove?
"""

import enum
from dataclasses import dataclass
import numpy as np
from numpy.typing import DTypeLike


class CoordinateSystem(enum.IntEnum):
    AZELR = enum.auto()
    NED = enum.auto()
    ENU = enum.auto()


@dataclass(frozen=True)
class ColumnMetadata:
    """column metadata dataclass"""

    colname: str
    dtype: DTypeLike


class ScheduleColumn(enum.Enum):
    STT_TSTMP = ColumnMetadata("stt_tstmp", np.int8)
    "timestamp, the start time of the schedule slice"

    coordinate_system = ColumnMetadata("coordinate_system", np.int8)
    coh_int_bandwidth = ColumnMetadata("coh_int_bandwidth", np.float64)
    pointing = ColumnMetadata("pointing", np.float64)
    ipp = ColumnMetadata("ipp", np.float64)
    pulse_length = ColumnMetadata("pulse_length", np.float64)
