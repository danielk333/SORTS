from __future__ import annotations
import logging, math, typing as t
import numpy as np
import numpy.typing as npt
from sorts.radar import Station
from sorts.frames import enu_to_ecef, ecef_to_enu, sph_to_cart
from sorts.types import (
    Float_as_deg,
    Datetime64_us,
    Float64_as_m,
    EcefCoordinates,
    EnuCoordinates,
    Datetime_Like,
)
from sorts.utils import to_datetime64_us
from sorts.schedule import Schedule, ExperimentDetail
from sorts.controller import pointing_funcs

logger = logging.getLogger(__name__)


class Spec(t.TypedDict):
    """A TypedDict of params"""

    tx_station: Station
    rx_stations: t.Sequence[Station]
    azimuth: Float_as_deg
    min_elevation: Float_as_deg
    pointings_per_cycle: int
    scan_range: npt.NDArray[Float64_as_m]
    exp_detail: ExperimentDetail


# TODO: can be dissolved?
class State(t.TypedDict):
    """A TypedDict of params"""

    start_time: Datetime64_us
    end_time: Datetime64_us
    tx_schedule_size: int
    tx_pointings_of_a_cycle: EnuCoordinates
    """NOTE: It may contain out of range pointings"""


class Output(t.NamedTuple):
    """A named tuple of `(tx_schedule, [rx_schedule, ...])`"""

    tx_schedule: Schedule
    rx_schedules: t.Sequence[Schedule]


# TODO: should we generate tx pointings at the specified ranges instead of normalized to 1?
def generate_from_state(spec: Spec, state: State) -> Output:
    # The logic of this function:
    # 1. repeat the cycle of tx pointings from state to form the tx schedule
    # 2. from the single cycle of tx pointings, we convert it into ECEF location coord and extend them by the `scan_range`
    # 3. using the resultant location coords from previous step,
    #    we convert them to rx station pointings of a cycle in ECEF coord,
    #    and then further back to pointings in ENU coord,
    #    and finally repeat them to form a rx schedule, for each rx station

    pointings_per_cycle = state["tx_pointings_of_a_cycle"].shape[1]

    # NOTE: for `np.arange` 'stop param,
    #   - we subtract 'slice_duration' so that only full slice are included
    #   - and add `+1` so that slice with time range `('end_time - 'slice_duration', 'end_time')` is included
    tx_slice_start_time: npt.NDArray[Datetime64_us] = np.arange(
        state["start_time"],
        state["end_time"] - spec["exp_detail"]["slice_duration"] + 1,
        spec["exp_detail"]["slice_duration"],
    )

    # TODO: `tx_schedule_size` is a bit of a mismisnomer, as out-of-range entries might later be removed
    # repeat a cycle of pointings until it is at least the size of `tx_schedule_size`,
    # then trim to exactly `tx_schedule_size` long
    tx_pointing: EnuCoordinates = np.tile(
        state["tx_pointings_of_a_cycle"],
        (state["tx_schedule_size"] + pointings_per_cycle - 1) // pointings_per_cycle,
    )[:, : state["tx_schedule_size"]]

    # mask tx values by min_elevation requirement
    tx_mask = pointing_funcs.create_mask_by_min_elevation(
        tx_pointing, spec["tx_station"].min_elevation
    )
    tx_slice_start_time_masked = tx_slice_start_time[tx_mask]
    tx_pointing_masked = tx_pointing[:, tx_mask]

    tx_stn_num = next(
        (k for k, v in spec["exp_detail"]["stn_num_map"].items() if v == spec["tx_station"].uid)
    )

    tx_schedule = Schedule.from_ndarrays(
        {
            "stn_id": spec["tx_station"].uid,
            "exp_detail_map": {spec["exp_detail"]["id"]: spec["exp_detail"]},
            "start_time": tx_slice_start_time_masked,
            "end_time": tx_slice_start_time_masked + spec["exp_detail"]["slice_duration"],
            "exp_num": np.full(
                len(tx_slice_start_time_masked), spec["exp_detail"]["id"], dtype=np.int16
            ),
            "stn_num": np.full(len(tx_slice_start_time_masked), tx_stn_num, dtype=np.int16),
            "simult_num": np.full(len(tx_slice_start_time_masked), 0, dtype=np.int16),
            "pointing": tx_pointing_masked,
        }
    )

    # TODO: `rx_schedule_size` is a bit of a mismisnomer, as out-of-range entries might later be removed
    rx_slice_start_time = tx_slice_start_time.repeat(len(spec["scan_range"]))
    rx_schedule_size = state["tx_schedule_size"] * len(spec["scan_range"])
    rx_schedules: list[Schedule] = []
    tx_pointings_of_a_cycle_without_translation_ecef: EcefCoordinates = enu_to_ecef(
        lat=spec["tx_station"].ecef_lat,
        lon=spec["tx_station"].ecef_lon,
        alt=spec["tx_station"].ecef_alt,
        enu=state["tx_pointings_of_a_cycle"],
        degrees=True,
    )
    rx_pointing_of_a_cycle_ecef: EcefCoordinates = (
        tx_pointings_of_a_cycle_without_translation_ecef[:, :, np.newaxis]
        * spec["scan_range"][np.newaxis, np.newaxis, :]
        + spec["tx_station"].ecef[:, np.newaxis, np.newaxis]
    ).reshape((3, -1))

    for rx_station in spec["rx_stations"]:
        rx_pointings_of_a_cycle_without_translation_ecef: EcefCoordinates = (
            rx_pointing_of_a_cycle_ecef - rx_station.ecef[:, np.newaxis]
        )
        rx_pointings_of_a_cycle_enu: EnuCoordinates = ecef_to_enu(
            lat=rx_station.ecef_lat,
            lon=rx_station.ecef_lon,
            alt=rx_station.ecef_alt,
            ecef=rx_pointings_of_a_cycle_without_translation_ecef,
            degrees=True,
        )

        # repeat a cycle of pointings until it is at least the size of `rx_schedule_size`
        # then trim to exactly `rx_schedule_size` long
        rx_pointings_enu: EnuCoordinates = np.tile(
            rx_pointings_of_a_cycle_enu,
            (rx_schedule_size + pointings_per_cycle - 1) // pointings_per_cycle,
        )[:, :rx_schedule_size]
        rx_pointings_simult_num = np.arange(rx_schedule_size) % len(spec["scan_range"])

        # mask rx values by min_elevation requirement, and has a corresponding tx value
        rx_mask_by_min_elevation = pointing_funcs.create_mask_by_min_elevation(
            rx_pointings_enu, rx_station.min_elevation
        )
        rx_mask_by_tx_mask = np.isin(rx_slice_start_time, tx_slice_start_time_masked)
        rx_mask = np.logical_and(rx_mask_by_min_elevation, rx_mask_by_tx_mask)
        rx_slice_start_time_masked = rx_slice_start_time[rx_mask]
        rx_pointing_masked = rx_pointings_enu[:, rx_mask]
        rx_pointings_simult_num_masked = rx_pointings_simult_num[rx_mask]

        rx_stn_num = next(
            (k for k, v in spec["exp_detail"]["stn_num_map"].items() if v == rx_station.uid)
        )

        rx_schedule = Schedule.from_ndarrays(
            {
                "stn_id": rx_station.uid,
                "exp_detail_map": {spec["exp_detail"]["id"]: spec["exp_detail"]},
                "start_time": rx_slice_start_time_masked,
                "end_time": rx_slice_start_time_masked + spec["exp_detail"]["slice_duration"],
                "exp_num": np.full(
                    len(rx_slice_start_time_masked), spec["exp_detail"]["id"], dtype=np.int16
                ),
                "stn_num": np.full(len(rx_slice_start_time_masked), rx_stn_num, dtype=np.int16),
                "simult_num": rx_pointings_simult_num_masked,
                "pointing": rx_pointing_masked,
            }
        )

        rx_schedules.append(rx_schedule)

    return Output(tx_schedule, rx_schedules)


class FenceScanController:
    """
    Generate schedule for a fence scaning pattern

    - The preferred way to create instances of this class is via its class methods (e.g. `TrackerController.from_space_object`).
    - This class serve as a frontend to the `State` type in this module
    """

    # TODO: the radar station computation capacity poses limit on the size of simutaneous `scan_range`, we should check/validate against it

    def __init__(self, spec: Spec, state: State | None):
        self.spec: Spec = spec
        self.state: State | None = state

        self._cached_output: Output | None = None

    @classmethod
    def from_scan_spec(
        cls,
        tx_station: Station,
        rx_stations: t.Sequence[Station],
        azimuth: Float_as_deg,
        min_elevation: Float_as_deg,
        pointings_per_cycle: int,
        scan_range: npt.NDArray[Float64_as_m],
        exp_detail: ExperimentDetail,
    ) -> FenceScanController:
        # TODO: update/adapt or remove?
        # self._total_duration_s = (self.end_time - self.start_time).total_seconds()
        # if self._total_duration_s < self.dwell_s:
        #     raise RuntimeError(
        #         f"The specified time range ({self.start_time.isoformat()} to {self.end_time.isoformat()}) "
        #         + f"cannot be smaller than the dwell ({self.dwell_s} sec)."
        #     )

        ctrl = FenceScanController(
            spec={
                "tx_station": tx_station,
                "rx_stations": rx_stations,
                "azimuth": azimuth,
                "min_elevation": min_elevation,
                "pointings_per_cycle": pointings_per_cycle,
                "scan_range": scan_range,
                "exp_detail": exp_detail,
            },
            state=None,
        )

        return ctrl

    def compute_single_cycle_pointings(self, start_time: Datetime_Like, end_time: Datetime_Like):
        """Do the computation then update the `state` property and return `self`."""

        exp_detail = self.spec["exp_detail"]

        start_time_np = to_datetime64_us(start_time)
        end_time_np = to_datetime64_us(end_time)
        tx_schedule_size = math.floor((end_time_np - start_time_np) / exp_detail["slice_duration"])

        tx_pointings_of_a_cycle = sph_to_cart(
            pointing_funcs.fence_pattern(
                azimuth=self.spec["azimuth"],
                min_elevation=self.spec["min_elevation"],
                pointings_per_cycle=self.spec["pointings_per_cycle"],
            ),
            degrees=True,
        )

        self.state = {
            "start_time": start_time_np,
            "end_time": end_time_np,
            "tx_schedule_size": tx_schedule_size,
            "tx_pointings_of_a_cycle": tx_pointings_of_a_cycle,
        }

        return self

    def generate(self, start_time: Datetime_Like, end_time: Datetime_Like) -> Output:
        self.compute_single_cycle_pointings(start_time, end_time)
        state = t.cast(State, self.state)

        output = generate_from_state(spec=self.spec, state=state)
        self._cached_output = output

        return output
