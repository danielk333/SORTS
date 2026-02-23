"""A module of shared types"""

import logging, enum
from dataclasses import dataclass
from sorts import types, radar


logger = logging.getLogger(__name__)

SimultaneousNum = int
"""An int16 that corresponds to the order in simultaneous pointings"""

ExperimentId = int
"""A unique int16 that identifies an experiment"""


@dataclass(kw_only=True)
class ExperimentDetail:
    id: ExperimentId

    coh_int_bandwidth: float
    ipp: float
    pulse_length: float
    power: float
    bandwidth: float
    duty_cycle: float
    noise_temp: float

    slice_duration: types.Timedelta64_us
    "Duration of a control slice, in micro-second"


ExperimentDetailMap = dict[ExperimentId, ExperimentDetail]


ExperimentIdStationIdPairsMap = dict[ExperimentId, list[tuple[radar.StationId, radar.StationId]]]


class ScheduleKey(enum.StrEnum):
    index = "index"  # type: ignore ; seems type checker might confuse this with the `index` method from `str`
    exp_num = "exp_num"
    stn_num = "stn_num"
    simult_num = "simult_num"
    start_time = "start_time"
    end_time = "end_time"
    pointing_e = "pointing_e"
    pointing_n = "pointing_n"
    pointing_u = "pointing_u"


class ScheduleValidationError(Exception):
    pass
