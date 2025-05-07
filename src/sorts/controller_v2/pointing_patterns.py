import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


def fence_pointing(
    azimuth_deg: float,
    min_elevation_deg: float,
    dwell_s: float,
    num: int,
    start_time: datetime,
    end_time: datetime,
):
    """Return an `azelr` of type `NDArray[float64]`, shape `(3, num)`"""

    interval_s = (end_time - start_time).total_seconds() / num
    if interval_s < dwell_s:
        raise RuntimeError(
            f"The available dwell duration will be too small with the give `start_time`, `end_time` and `num`: "
            + f"{interval_s} < {dwell_s}"
        )

    el = np.linspace(min_elevation_deg, 180.0 - min_elevation_deg, num=num, dtype=np.float64)
    el_over_90deg_mask = el > 90.0
    # make 0 <= el < 90
    el[el_over_90deg_mask] = 180.0 - el[el_over_90deg_mask]

    az = np.full(num, azimuth_deg, dtype=np.float64)
    # wrap around az for those with el > 90 deg
    az[el_over_90deg_mask] = np.mod(az[el_over_90deg_mask] + 180.0, 360.0)

    azelr = np.stack(
        [
            az,
            el,
            np.full(num, 1.0, dtype=np.float64),
        ],
    )

    return azelr


__all__ = ["fence_pointing"]
