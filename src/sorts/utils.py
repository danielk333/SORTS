import typing as t
import numpy as np
from astropy.time import Time
from sorts.types import Datetime64_us


def astropy_time_to_datetime64_us(time: Time) -> Datetime64_us:
    return t.cast(np.datetime64, time.to_value("datetime64")).astype("datetime64[us]")
