"""
Shared types in this package.

(Types might live in their own module instead of here if it improves readability,
and the imports can be worked around, e.g, by `if t.TYPE_CHECKING`)
"""
from copy import deepcopy
from dataclasses import dataclass, fields
import typing as t
import numpy as np
import numpy.typing as npt
from datetime import datetime, timedelta
from astropy.time import Time, TimeDelta


Datetime64_us = np.datetime64
"`numpy` `datetime64` in `us` resolution"

Timedelta64_us = np.timedelta64
"`numpy` `timedelta64` in `us` resolution"

Datetime_Like = datetime | Time | Datetime64_us | str
"""
One of:
- python builtin `datetime`,
- astropy `Time`
- numpy `timedelta64`
- ISO 8601 date or datetime string supported by numpy
"""

Timedelta_Like = timedelta | TimeDelta | Timedelta64_us | int
"""
One of:
- python builtin `timedelta`,
- astropy `TimeDelta`
- numpy `timedelta64`
- `int`, the length of the duration in `us` resolution
"""

TimeRange_us = tuple[Datetime64_us, Datetime64_us]
"""The start time and end time of the passage, a right-open interval"""

Float_as_sec = float
"`float` as second"

Float_as_m = float
"`float` as meters"

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

# TODO: make sure these comments make sense
NDArray_3x1 = npt.NDArray
"(3,) shaped ndarray (i.e. a single 3D vector)"

NDArray_6x1 = npt.NDArray
"(6,) shaped ndarray (i.e. a single 6D vector)"

NDArray_N = npt.NDArray
"(n,) shaped ndarray (i.e. a single `n`D vector)"

NDArray_6 = npt.NDArray
"(6,) shaped ndarray (i.e. a single `6`D vector)"

NDArray_3xN = npt.NDArray
"(3,n) shaped ndarray (i.e. `3` rows and `n` columns)"

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

AzelCoordinates_DegM = NDArray_3xN[np.float64]
"""
`(Azimuth, Elevation)` spherical coordinates in degrees; a `(2,n)` ndarray of `float64`

- Azimuth should be in [-180, 180)
- Elevation should be in [0, 90]
"""

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


class TxRxTuple[TxType, RxType](t.NamedTuple):
    tx: TxType
    rx: RxType


type Tuple_3[T] = tuple[T, T, T]
type Tuple_6[T] = tuple[T, T, T, T, T, T]
type Tuple_7[T] = tuple[T, T, T, T, T, T, T]
type SpaceObjectId = int

IndexLike = int | list[int] | tuple[int] | NDArray_N | slice | np.integer

S = t.TypeVar("S", bound="Settings")

Frames = t.Literal[
    "TEME",
    "ITRS",
    "ITRF",
    "ICRS",
    "ICRF",
    "GCRS",
    "GCRF",
    "HCRS",
    "HCRF",
]
GravModels = t.Literal["WGS84"]
StateType = t.Literal["kepler", "cartesian"]
AnomalyType = t.Literal["mean", "eccentric", "true"]


@dataclass
class Settings:
    def copy(self: S) -> S:
        kwargs = {key: deepcopy(getattr(self, key)) for key in self.keys}
        return self.__class__(**kwargs)

    @property
    def keys(self) -> list[str]:
        return [key.name for key in fields(self)]
