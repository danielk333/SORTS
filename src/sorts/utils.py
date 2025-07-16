import typing as t
import numpy as np
import numpy.typing as npt
from astropy.time import Time
from sorts.types import Datetime64_us, Float64_as_deg


def astropy_time_to_datetime64_us(time: Time) -> Datetime64_us:
    return t.cast(np.datetime64, time.to_value("datetime64")).astype("datetime64[us]")


def wrap_latitudes_longitudes(
    lat: npt.NDArray[Float64_as_deg], lon: npt.NDArray[Float64_as_deg]
) -> tuple[npt.NDArray[Float64_as_deg], npt.NDArray[Float64_as_deg]]:
    """
    Wrap latitudes and longitudes so that they stay within [-90, 90] and [-180, 180)

    (When a latitude wraps, the corresponding longitude value is flipped (added 180deg).)

    Returns `(wrapped_latitudes, wrapped_longitudes)` tuple
    """

    lat_wrapped = ((lat + 90) % 180) - 90
    flips = (lat + 90) // 180
    lon_wrapped = (((lon + 180) + (flips * 180)) % 360) - 180

    return (lat_wrapped, lon_wrapped)


def wrap_azimuths_elevations(
    az: npt.NDArray[Float64_as_deg], el: npt.NDArray[Float64_as_deg]
) -> tuple[npt.NDArray[Float64_as_deg], npt.NDArray[Float64_as_deg]]:
    """
    Wrap azimuths and elevations so that they stay within [-180, 180) and [0, 90]

    (When an elevation wraps, the corresponding azimuth value is flipped (added 180deg).)

    Returns `(wrapped_azimuths, wrapped_elevations)` tuple
    """

    el_wrapped = (el) % 90
    flips = (el) // 90
    az_wrapped = (((az + 180) + (flips * 180)) % 360) - 180

    return (az_wrapped, el_wrapped)
