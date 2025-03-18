import logging, typing as t, math
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from .. import scheduler_v2 as schr
from .. import controller_v2 as ctrlr

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class RandomUniformScansController(ctrlr.ControllerProtocol):
    """
    a controller that generate random uniform scans

    TODO: numpy structured array vs pandas dataframe?

    ASK: is dwell calculated by `pulse_length` and does it includes ipp and other gaps?
    """

    min_elevation_deg: float = 30.0
    dwell_ns: float = 1e6  # 1ms
    npoints: int = 10_000

    coordinate_system: schr.CoordinateSystem = schr.CoordinateSystem.ENU
    coh_int_bandwidth: float = 1.0
    ipp: float = 1.0
    pulse_length: float = 1.0

    def generate(
        self,
        stt_tstmp: datetime = datetime.now(timezone.utc),
        end_tstmp: datetime = datetime.now(timezone.utc) + timedelta(hours=24),
        res_ns: int = int(1e6),
    ) -> pd.DataFrame:
        """
        Parameters
        ---

        stt_tstmp
            start timestamp, irrelevant in this controller
        end_time
            end timestamp, irrelevant in this controller

        Returns
        ---
        a DataFrame with these columns:

        |index     |coordinate_system|coh_int_bandwidth|pointing                 |ipp    |pulse_length|
        |:-        |:-               |:-               |:-                       |:-     |:-          |
        |datetime64|int8             |float64          |(float64,float64,float64)|float64|float64     |
        """
        ...

        max_points_by_res_ns = math.floor(
            (end_tstmp - stt_tstmp) / timedelta(microseconds=res_ns / 1000)
        )
        if max_points_by_res_ns < self.npoints:
            raise RuntimeError(
                f"npoints: {self.npoints} larger than res_ns allows: {max_points_by_res_ns}"
            )

        max_points_by_dwell_ns = math.floor(
            (end_tstmp - stt_tstmp) / timedelta(microseconds=self.dwell_ns / 1000)
        )
        if max_points_by_dwell_ns < self.npoints:
            raise RuntimeError(
                f"npoints: {self.npoints} larger than dwell_ns allows: {max_points_by_dwell_ns}"
            )

        min_z = np.sin(np.radians(self.min_elevation_deg))
        theta = 2 * np.pi * np.random.rand(self.npoints)
        phi = np.arccos(np.random.rand(self.npoints) * (1 - min_z) + min_z)

        arr = np.recarray((self.npoints,), dtype=schr.schedule_ndarray_dtype)
        cn = schr.schedule_column_names

        arr[cn["stt_tstmp"]] = pd.date_range(
            start=stt_tstmp, end=end_tstmp, periods=self.npoints
        ).values

        arr[cn["coordinate_system"]].fill(self.coordinate_system)
        arr[cn["coh_int_bandwidth"]].fill(self.coh_int_bandwidth)

        arr[cn["pointing_p1"]] = np.cos(theta) * np.sin(phi)
        arr[cn["pointing_p2"]] = np.sin(theta) * np.sin(phi)
        arr[cn["pointing_p3"]] = np.cos(phi)

        arr[cn["ipp"]].fill(self.ipp)
        arr[cn["ipp"]].fill(self.pulse_length)

        ret_df = pd.DataFrame(
            arr,
            columns=[*schr.schedule_column_names.values()],
        )

        # align the rows to res_ns
        ret_df[cn["stt_tstmp"]] = t.cast(pd.Series, ret_df[cn["stt_tstmp"]]).dt.floor(f"{res_ns}ns")

        ret_df.set_index(cn["stt_tstmp"], inplace=True)

        return ret_df

    # TODO: remove?
    # NOTE: kept for tmp reference; a ver of generate that return ndarray
    def generate_ndarray(
        self,
        stt_tstmp: datetime = datetime.now(timezone.utc),
        end_tstmp: datetime = datetime.now(timezone.utc) + timedelta(hours=24),
        res_ns: int = int(1e6),
    ) -> np.ndarray:
        """
        Parameters
        ---

        stt_tstmp
            start timestamp, irrelevant in this controller
        end_time
            end timestamp, irrelevant in this controller

        Returns
        ---
        2d ndarray with these columns:

        |stt_tstmp |coordinate_system|coh_int_bandwidth|pointing                 |ipp    |pulse_length|
        |:-        |:-               |:-               |:-                       |:-     |:-          |
        |datetime64|int8             |float64          |(float64,float64,float64)|float64|float64     |
        """
        ...

        min_z = np.sin(np.radians(self.min_elevation_deg))
        theta = 2 * np.pi * np.random.rand(self.npoints)
        phi = np.arccos(np.random.rand(self.npoints) * (1 - min_z) + min_z)

        ret = np.recarray((self.npoints,), dtype=schr.schedule_ndarray_dtype)
        cn = schr.schedule_column_names

        ret[cn["stt_tstmp"]] = pd.date_range(
            start=stt_tstmp, end=end_tstmp, periods=self.npoints
        ).values

        ret[cn["coordinate_system"]].fill(self.coordinate_system)
        ret[cn["coh_int_bandwidth"]].fill(self.coh_int_bandwidth)

        ret[cn["pointing"]]["p1"] = np.cos(theta) * np.sin(phi)
        ret[cn["pointing"]]["p2"] = np.sin(theta) * np.sin(phi)
        ret[cn["pointing"]]["p3"] = np.cos(phi)

        ret[cn["ipp"]].fill(self.ipp)
        ret[cn["ipp"]].fill(self.pulse_length)

        return ret


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

    record_cols: list[schr.ScheduleColumnName] = [
        "coordinate_system",
        "pointing",
    ]

    def generator():
        for _ in range(npoints):
            ctrl_slice = np.record(
                (
                    schr.CoordinateSystem.ENU,
                    (
                        np.cos(theta) * np.sin(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(phi),
                    ),
                ),
                dtype=[(k, schr.schedule_column_dtypes[k]) for k in record_cols],
            )

            yield ctrl_slice

    return generator()
