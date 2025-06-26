import typing as t
from datetime import datetime
import numpy as np
import numpy.typing as npt


Datetime64_us = np.datetime64
Timedelta64_us = np.timedelta64

Float64_as_sec = np.float64
"float64 as second"

Float64_as_m = np.float64
"float64 as meter"

Float64_as_deg = np.float64
"float64 as angle in degree"

Float64_as_rad = np.float64
"float64 as angle in radian"

NDArray_1d = npt.NDArray
"(1,) shaped ndarray"

NDArray_2d = npt.NDArray
"(2,) shaped ndarray"

NDArray_3d = npt.NDArray
"(3,) shaped ndarray"

EcefStates = npt.NDArray[np.float64]
"ECEF states, a `(6,n)` ndarray of `float64`, usually used for space objects"
