import logging
import numpy as np
import numpy.typing as npt
import spacecoords
from sorts.types import Float_as_deg, AzelrCoordinates_DegM, EnuCoordinates

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


def create_mask_by_min_elevation(
    pointings: EnuCoordinates, min_elevation: float
) -> npt.NDArray[np.bool]:
    loc_zenith = np.array([0, 0, 1], dtype=np.float64)

    _pointings_zenith_ang = spacecoords.linalg.vector_angle(loc_zenith, pointings, degrees=True)
    match _pointings_zenith_ang:
        case np.ndarray():
            pointings_zenith_ang = _pointings_zenith_ang
        case float():
            pointings_zenith_ang = np.array([_pointings_zenith_ang], dtype=np.float64)
        case _:
            raise RuntimeError(
                f"unexpected type of `_pointings_zenith_ang`: {type(_pointings_zenith_ang)}"
            )

    el_in_range_mask = pointings_zenith_ang <= 90.0 - min_elevation

    return el_in_range_mask
