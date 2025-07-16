import typing as t
import numpy as np
import numpy.typing as npt
from astropy.time import Time
from sorts.types import Datetime64_us, Float64_as_deg


def astropy_time_to_datetime64_us(time: Time) -> Datetime64_us:
    return t.cast(np.datetime64, time.to_value("datetime64")).astype("datetime64[us]")


def wrapped_lat_lon(
    lat: npt.NDArray[Float64_as_deg], lon: npt.NDArray[Float64_as_deg]
) -> tuple[npt.NDArray[Float64_as_deg], npt.NDArray[Float64_as_deg]]:
    """
    Wrap latitudes and longitudes so that they stay within [-90, 90] and [-180, 180)

    (When a latitude wraps, the corresponding longitude value is flipped (added 180deg).)
    """

    lat_wrapped = ((lat + 90) % 180) - 90
    flips = (lat + 90) // 180
    lon_wrapped = (((lon + 180) + (flips * 180)) % 360) - 180

    return (lat_wrapped, lon_wrapped)
