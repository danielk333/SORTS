import logging, typing as t
import numpy as np
from .controller_protocol import ControllerProtocol
from ..scheduler_v2.schedule_metadata import (
    CoordinateSystem,
    schedule_column_dtypes,
    ScheduleColumnName,
    schedule_column_names,
)

logger = logging.getLogger(__name__)


class ScannerController(ControllerProtocol):
    pass
    # def close(self) -> None:
    #     pass


# TODO: `dwell` not needed here?
def random_uniform_scan_points(min_elevation=30.0, dwell=0.1, npoints=10000):
    """
    create a generator that output uniform randomly distributed points in the FOV.

    Parameters
    ----------
    min_elevation : float
        min elevation angle in degree
    dwell : float
        radar dwell time in seconds
    npoints : int
        number of points to sample

    TODO: numpy structured array vs pandas dataframe?

    ASK: does dwell includes ipp and other gaps?
    """

    min_z = np.sin(np.radians(min_elevation))

    theta = 2 * np.pi * np.random.rand(npoints)
    phi = np.arccos(np.random.rand(npoints) * (1 - min_z) + min_z)

    k = np.empty((3, npoints), dtype=np.float64)
    k[0, :] = np.cos(theta) * np.sin(phi)
    k[1, :] = np.sin(theta) * np.sin(phi)
    k[2, :] = np.cos(phi)

    results = {
        schedule_column_names["coordinate_system"]: "enu",
        schedule_column_names["pointing"]: k,
    }

    return results


# TODO: kept as tmp reference, should be removed when done
def __random_uniform_scan_points(min_elevation=30.0, dwell=0.1, npoints=10000):
    """
    create a generator that output uniform randomly distributed points in the FOV.

    Parameters
    ----------
    min_elevation : float
        min elevation angle in degree
    dwell : float
        radar dwell time in seconds
    npoints : int
        number of points to sample

    TODO: do we need to include `coordinates="enu"`?
    TODO: numpy structured array vs pandas dataframe?

    ASK: does dwell includes ipp and other gaps?
    """

    min_z = np.sin(np.radians(min_elevation))

    theta = 2 * np.pi * np.random.rand()
    phi = np.arccos(np.random.rand() * (1 - min_z) + min_z)

    record_cols: list[ScheduleColumnName] = [
        "coordinate_system",
        "pointing",
    ]

    def generator():
        for _ in range(npoints):
            ctrl_slice = np.record(
                (
                    CoordinateSystem.ENU,
                    (
                        np.cos(theta) * np.sin(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(phi),
                    ),
                ),
                dtype=[(k, schedule_column_dtypes[k]) for k in record_cols],
            )

            yield ctrl_slice

    return generator()
