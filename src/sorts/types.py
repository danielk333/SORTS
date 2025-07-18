import numpy as np
import numpy.typing as npt
from datetime import datetime
from astropy.time import Time

Datetime64_us = np.datetime64
"`numpy` `datetime64` in `us` resolution"

Timedelta64_us = np.timedelta64
"`numpy` `timedelta64` in `us` resolution"

Datetime_like = datetime | Time | Datetime64_us | str
"""
One of:
- python builtin `datetime`,
- astropy `Time`
- numpy `timedelta64`
- ISO 8601 date or datetime string supported by numpy
"""

Float_as_sec = float
"`float` as second"

Float_as_deg = float
"`float` as angle in degrees"

Float64_as_sec = np.float64
"`numpy` `float64` as seconds"

Float64_as_m = np.float64
"`numpy` `float64` as meters"

Float64_as_deg = np.float64
"`numpy` `float64` as angle in degrees"

Float64_as_rad = np.float64
"`numpy` `float64` as angle in radians"

NDArray_3x1 = npt.NDArray
"(3,) shaped ndarray (i.e. a single 3D vector)"

NDArray_6x1 = npt.NDArray
"(6,) shaped ndarray (i.e. a single 6D vector)"

NDArray_3xN = npt.NDArray
"(3,n) shaped ndarray (i.e. `3` 1D vectors of length `n`)"

NDArray_6xN = npt.NDArray
"(6,n) shaped ndarray (i.e. `6` 1D vectors of length `n`)"

NDArray_Nx3 = npt.NDArray
"(n, 3) shaped ndarray (i.e. `n` 3D vectors)"

NDArray_Nx6 = npt.NDArray
"(n, 6) shaped ndarray (i.e. `n` 6D vectors)"

EnuCoordinate = NDArray_3x1[np.float64]
"ENU cartesian coordinate, a (3,) shaped ndarray of `float64` (i.e. a single 3D vector)"

EnuCoordinates = NDArray_3xN[np.float64]
"ENU cartesian coordinates, a `(3,n)` ndarray of `float64`"

AzelrCoordinates_DegM = NDArray_3xN[np.float64]
"""
`(Azimuth, Elevation, Range)` spherical coordinates in degrees and meters; a `(3,n)` ndarray of `float64`

- Azimuth should be in [-180, 180)
- Elevation should be in [0, 90]
"""

GeodeticCoordinates_DegM = NDArray_3xN[np.float64]
"""
`(Latitude, Longitude, Height)` geodetic coordinates in degrees and meters; a `(3,n)` ndarray of `float64`

- Latitude should be in [-90, 90]
- Longitude should be in [-180, 180)
"""

EcefCoordinates = NDArray_3xN[np.float64]
"ECEF cartesian coordinates, a `(3,n)` ndarray of `float64`"

EcefStates = NDArray_6xN[np.float64]
"ECEF states in cartesian coordinate, a `(6,n)` ndarray of `float64`, usually used for space objects"
