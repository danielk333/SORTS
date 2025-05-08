from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


# TODO: this is a tmp soution
# TODO: maybe need to support cases where some of them varies by time?
@dataclass
class ExperimentDetail:
    coh_int_bandwidth: float
    ipp: float
    pulse_length: float
    power: float
    bandwidth: float
    duty_cycle: float
    noise_temp: float


__all__ = ["ExperimentDetail"]
