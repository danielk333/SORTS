from dataclasses import dataclass
from sorts.types import Timedelta64_us


# TODO: rename to sth like `ControlSliceDetail`?
@dataclass(kw_only=True)
class ExperimentDetail:
    coh_int_bandwidth: float
    ipp: float
    pulse_length: float
    power: float
    bandwidth: float
    duty_cycle: float
    noise_temp: float

    slice_duration: Timedelta64_us
    "Duration of a control slice, in micro-second"
