import typing as t
from datetime import datetime
import numpy as np
import numpy.typing as npt


Datetime64_us = np.datetime64
Timedelta64_us = np.timedelta64

Float_64_as_sec = np.float64
"float64 as second"

Float64_as_m = np.float64
"float64 as meter"

EcefStates = npt.NDArray[np.float64]
"ECEF states, a `(6,n)` ndarray of `float64`, Usually used for space objects"


__all__ = ["Datetime64_us", "Timedelta64_us", "Float_64_as_sec", "Float64_as_m", "EcefStates"]
