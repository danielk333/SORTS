import logging
import numpy as np
from sorts.types import Float_as_deg, AzelrCoordinates_DegM

logger = logging.getLogger(__name__)


def fence_pattern(
    azimuth: Float_as_deg,
    min_elevation: Float_as_deg,
    pointings_per_cycle: int,
) -> AzelrCoordinates_DegM:
    """
    Generate radar pointings that evenly sweep over a symmetrical elevation range, at the given `azimuth`

    Note that the sweeping always strokes in the same direction (not back and forth).
    """

    el = np.linspace(
        min_elevation, 180.0 - min_elevation, num=pointings_per_cycle, dtype=np.float64
    )
    az = np.full(pointings_per_cycle, azimuth, dtype=np.float64)

    # make 0 <= el < 90
    el_over_90deg_mask = el > 90.0
    el[el_over_90deg_mask] = 180.0 - el[el_over_90deg_mask]

    # wrap around az for those with el > 90 deg
    az[el_over_90deg_mask] = np.mod(az[el_over_90deg_mask] + 180.0, 360.0)

    azelr = np.stack(
        [
            az,
            el,
            np.full(pointings_per_cycle, 1.0, dtype=np.float64),
        ],
    )

    return azelr
