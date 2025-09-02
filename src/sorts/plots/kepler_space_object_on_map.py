import numpy as np
import numpy.typing as npt
from sorts.space_object import SpaceObject
from sorts.types import Datetime_Like, Datetime64_us, Timedelta64_us, Float64_as_sec
from sorts.utils import to_datetime64_us
from .ecef_states_positions_plot import ecef_states_positions_plot


def kepler_space_object_on_map(
    space_object: SpaceObject,
    epoch: Datetime_Like,
    num_points=500,
    start_time: Datetime_Like | None = None,
    end_time: Datetime_Like | None = None,
):
    """Plot a space object with keplerian orbit on a map in mercator projection"""

    # TODO: check if space_object.orbit.period can be adj to always return a float
    _period = space_object.orbit.period
    period: np.timedelta64
    match _period:
        case np.ndarray():
            period = _period[0].astype("timedelta64[s]")
        case int() | float():
            period = np.timedelta64(int(_period), "s")
        case _:
            raise RuntimeError(
                f"`space_object.orbit.period` have to be of type float or ndarray of float64, but is in type {type(space_object.orbit.period)}"
            )

    period = np.timedelta64(period, "s")
    epoch = to_datetime64_us(epoch)
    start_time = to_datetime64_us(start_time) if start_time is not None else epoch
    end_time = to_datetime64_us(end_time) if end_time is not None else start_time + period

    time_arr: npt.NDArray[Datetime64_us] = np.linspace(
        start_time.astype(np.float64),
        end_time.astype(np.float64),
        num_points,
    ).astype("datetime64[us]")

    dt_arr: npt.NDArray[Timedelta64_us] = time_arr - to_datetime64_us(epoch)
    dsec_arr: npt.NDArray[Float64_as_sec] = dt_arr.astype(np.float64) / 1e6  # type: ignore

    ecefs = space_object.get_state(dsec_arr)
    plot = ecef_states_positions_plot(ecefs)

    return plot
