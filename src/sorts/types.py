import typing as t
from datetime import datetime
import numpy as np
import numpy.typing as npt


Datetime64_us = np.datetime64
"`numpy` `datetime64` in `us` resolution"

Timedelta64_us = np.timedelta64
"`numpy` `timedelta64` in `us` resolution"

Float_as_sec = np.float64
"`float` as second"

Float_as_deg = float
"`float` as angle in degree"

Float64_as_sec = np.float64
"`numpy` `float64` as second"

Float64_as_m = np.float64
"`numpy` `float64` as meter"

Float64_as_deg = np.float64
"`numpy` `float64` as angle in degree"

Float64_as_rad = np.float64
"`numpy` `float64` as angle in radian"

NDArray_1d1 = npt.NDArray
"(1,) shaped ndarray (i.e. a single 1D vector)"

NDArray_2d1 = npt.NDArray
"(2,) shaped ndarray (i.e. a single 2D vector)"

NDArray_3d1 = npt.NDArray
"(3,) shaped ndarray (i.e. a single 3D vector)"

NDArray_6d1 = npt.NDArray
"(6,) shaped ndarray (i.e. a single 6D vector)"

NDArray_1dn = npt.NDArray
"(1,n) shaped ndarray (i.e. an array of n 1D vectors)"

NDArray_2dn = npt.NDArray
"(2,n) shaped ndarray (i.e. an array of n 2D vectors)"

NDArray_3dn = npt.NDArray
"(3,n) shaped ndarray (i.e. an array of n 3D vectors)"

NDArray_6dn = npt.NDArray
"(6,n) shaped ndarray (i.e. an array of n 6D vectors)"

EnuCoordinate = NDArray_3dn[np.float64]
"ENU cartesian coordinate, a (3,) shaped ndarray of `float64` (i.e. a single 3D vector)"

EnuCoordinates = NDArray_3dn[np.float64]
"ENU cartesian coordinates, a `(3,n)` ndarray of `float64`"

EcefCoordinate = NDArray_3d1[np.float64]
"ECEF cartesian coordinate, a (3,) shaped ndarray of `float64` (i.e. a single 3D vector)"

EcefCoordinates = NDArray_3dn[np.float64]
"ECEF cartesian coordinates, a `(3,n)` ndarray of `float64`"

EcefState = NDArray_6dn[np.float64]
"ECEF state in cartesian coordinate, a (6,) shaped ndarray of `float64`, usually used for space objects (i.e. a single 6D vector)"

EcefStates = NDArray_6dn[np.float64]
"ECEF states in cartesian coordinate, a `(6,n)` ndarray of `float64`, usually used for space objects"
