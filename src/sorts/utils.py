import typing as t
from datetime import datetime
import numpy as np
import numpy.typing as npt
from astropy.time import Time
from sorts.types import Datetime64_us, Float64_as_deg, Datetime_Like


def wrap_latitudes_longitudes(
    lat: npt.NDArray[Float64_as_deg], lon: npt.NDArray[Float64_as_deg]
) -> tuple[npt.NDArray[Float64_as_deg], npt.NDArray[Float64_as_deg]]:
    """
    Wrap latitudes and longitudes so that they stay within [-90, 90] and [-180, 180)

    Returns `(wrapped_latitudes, wrapped_longitudes)` tuple

    - When a latitude wraps, the corresponding longitude value is flipped (added 180deg)
    """

    lat_wrapped = ((lat + 90) % 180) - 90
    flips = (lat + 90) // 180
    lon_wrapped = (((lon + 180) + (flips * 180)) % 360) - 180

    return (lat_wrapped, lon_wrapped)


def wrap_azimuths_elevations(
    az: npt.NDArray[Float64_as_deg],
    el: npt.NDArray[Float64_as_deg],
    el_limit: t.Literal[90] | t.Literal[180] = 90,
    allow_negative_elevations=False,
) -> tuple[npt.NDArray[Float64_as_deg], npt.NDArray[Float64_as_deg]]:
    """
    Wrap azimuths and elevations so that they stay within [-180, 180) and [0, 90] (or [0, 180])

    Returns `(wrapped_azimuths, wrapped_elevations)` tuple

    - When an elevation wraps, the corresponding azimuth value is flipped (added 180deg)
    - `el_limit` controls whether elevations are wrapped to [0, 90] or [0, 180]
    - Negative elevations will trigger exception by default, controlled by `allow_negative_elevations`
    """

    # throw exception if there are negative elevation(s)
    if not allow_negative_elevations:
        neg_el_mask = el < 0
        first_neg_el_idx = np.argmax(neg_el_mask) if np.any(neg_el_mask) else None
        if first_neg_el_idx is not None:
            raise RuntimeError(
                f"There are negative elevation(s), which is invalid. e.g. at [{first_neg_el_idx}]: {el[first_neg_el_idx]}"
            )

    el = el % 180

    if el_limit == 180:
        return (az, el)

    else:
        el_wrapped = (el) % 90
        flips = (el) // 90
        az_wrapped = (((az + 180) + (flips * 180)) % 360) - 180

        return (az_wrapped, el_wrapped)


def to_datetime64_us(d: Datetime_Like) -> Datetime64_us:
    match d:
        case np.datetime64():
            return d
        case str() | datetime():
            return np.datetime64(d, "us")
        case Time():
            return t.cast(np.datetime64, d.to_value("datetime64")).astype("datetime64[us]")
        case _:
            raise RuntimeError(f"Convertion from {type(d)} to `datetime64[us]` is not supported.")


def to_pydatetime(d: Datetime_Like) -> datetime:
    match d:
        case datetime():
            return d
        case str():
            return np.datetime64(d, "us").astype(datetime)
        case np.datetime64():
            return d.astype(datetime)
        case Time():
            return t.cast(datetime, d.to_datetime())
        case _:
            raise RuntimeError(f"Convertion from {type(d)} to `datetime64[us]` is not supported.")
