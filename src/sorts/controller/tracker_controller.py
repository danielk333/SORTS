from __future__ import annotations
import logging, typing as t
import numpy as np
import numpy.typing as npt
import xarray as xr
import bokeh.layouts as bokeh_layouts
import pyant
from sorts.space_object import SpaceObject
from sorts.radar import Station
from sorts.types import (
    EcefStates,
    Datetime64_us,
    Timedelta64_us,
    Float64_as_sec,
    EnuCoordinates,
    Datetime_Like,
    Timedelta_Like,
)
from sorts.utils import to_datetime64_us, to_timedelta64_us
from sorts import plots
from sorts.schedule import Schedule, ScheduleData, ExperimentDetail, schedule_data_funcs

logger = logging.getLogger(__name__)


class Spec(t.TypedDict):
    """A TypedDict of params"""

    tx_station: Station
    rx_stations: t.Sequence[Station]
    exp_detail: ExperimentDetail
    spobj: t.NotRequired[SpaceObject]
    epoch: t.NotRequired[Datetime_Like]


class State(t.TypedDict):
    """A TypedDict of params"""

    spobj_time: npt.NDArray[Datetime64_us]
    spobj_states: EcefStates


def generate_from_state(spec: Spec, state: State) -> Schedule:
    loc_zenith = np.array([0, 0, 1], dtype=np.float64)

    # generate pointings
    tx_pointings: EnuCoordinates = spec["tx_station"].enu(state["spobj_states"][:3])

    tx_pointings_zenith_ang = pyant.coordinates.vector_angle(loc_zenith, tx_pointings, degrees=True)
    tx_el_in_range_mask = tx_pointings_zenith_ang <= 90.0 - spec["tx_station"].min_elevation
    tx_pointings = tx_pointings[:, tx_el_in_range_mask]

    rxs_pointings: list[EnuCoordinates] = []
    rx_el_in_range_with_tx_masks: list[npt.NDArray[np.bool]] = []
    pure_rx_stations = [stn for stn in spec["rx_stations"] if stn.uid != spec["tx_station"].uid]
    for rx_station in pure_rx_stations:
        rx_pointings: EnuCoordinates = rx_station.enu(state["spobj_states"][:3])

        rx_pointings_zenith_ang = pyant.coordinates.vector_angle(
            loc_zenith, rx_pointings, degrees=True
        )
        rx_el_in_range_mask = rx_pointings_zenith_ang <= 90.0 - rx_station.min_elevation

        rx_el_in_range_with_tx_mask = np.logical_and(tx_el_in_range_mask, rx_el_in_range_mask)
        rx_el_in_range_with_tx_masks.append(rx_el_in_range_with_tx_mask)

        rx_pointings = rx_pointings[:, rx_el_in_range_with_tx_mask]
        rxs_pointings.append(rx_pointings)

    tx_sch_time = state["spobj_time"][tx_el_in_range_mask]
    tx_sch_len = len(tx_sch_time)

    tx_schdata = schedule_data_funcs.from_ndarrays(
        {
            "exp_detail_map": {spec["exp_detail"]["id"]: spec["exp_detail"]},
            "start_time": tx_sch_time,
            "end_time": tx_sch_time + spec["exp_detail"]["slice_duration"],
            "exp_num": np.full(tx_sch_len, spec["exp_detail"]["id"], dtype=np.int16),
            "stn_num": np.full(tx_sch_len, spec["tx_station"].uid, dtype=np.int16),
            "simult_num": np.full(tx_sch_len, 0, dtype=np.int16),
            "pointing": tx_pointings,
        }
    )

    rx_schdatas: list[ScheduleData] = []
    for rx_stn, rx_mask, rx_pointings in zip(
        pure_rx_stations, rx_el_in_range_with_tx_masks, rxs_pointings
    ):
        rx_sch_time = state["spobj_time"][rx_mask]
        rx_sch_len = len(rx_sch_time)

        rx_schdatas.append(
            schedule_data_funcs.from_ndarrays(
                {
                    "exp_detail_map": {spec["exp_detail"]["id"]: spec["exp_detail"]},
                    "start_time": rx_sch_time,
                    "end_time": rx_sch_time + spec["exp_detail"]["slice_duration"],
                    "exp_num": np.full(rx_sch_len, spec["exp_detail"]["id"], dtype=np.int16),
                    "stn_num": np.full(rx_sch_len, rx_stn.uid, dtype=np.int16),
                    "simult_num": np.full(rx_sch_len, 0, dtype=np.int16),
                    "pointing": rx_pointings,
                }
            )
        )

    resultant_schdata = xr.concat([tx_schdata, *rx_schdatas], dim=Schedule._K.multi_index)
    resultant_schdata = resultant_schdata.sortby(Schedule._K.start_time)
    output = Schedule(resultant_schdata)

    return output


# TODO: remove or adapt to ENU coord
def plot_state_and_output(state: State, rx_schedules: t.Sequence[Schedule]):
    pos_plot = plots.ecef_states_positions_plot(state["spobj_states"])
    pos_plot.title = "ecef_states_positions_plot"

    rx_skyplot_plots = []
    for idx, rx_schedule in enumerate(rx_schedules):
        rx_sch_dict = rx_schedule.to_ndarrays()
        rx_skyplot_plot = plots.azel_skyplot(
            rx_sch_dict["pointing"][0],
            rx_sch_dict["pointing"][1],
        )
        rx_skyplot_plot.title = f"rx_skyplot_plot_{idx}"
        rx_skyplot_plots.append(rx_skyplot_plot)

    plot = bokeh_layouts.layout(
        [
            [pos_plot],
            rx_skyplot_plots,
        ]  # type: ignore
    )
    return plot


class TrackerController:
    """
    Generate pointing schedule that tracks a space object.

    - The preferred way to create instances of this class is via its class methods (e.g. `TrackerController.from_space_object`).
    - This class serve as a frontend to the `State` type in this module
    """

    def __init__(self, spec: Spec, state: State | None):
        self.spec: Spec = spec
        self.state: State | None = state

        self._cached_output: Schedule | None = None
        """A cache of the latest `Output`, handy for plotting"""

    @classmethod
    def from_ecef_states(
        cls,
        time: npt.NDArray[Datetime64_us],
        space_object_states: EcefStates,
        tx_station: Station,
        rx_stations: t.Sequence[Station],
        exp_detail: ExperimentDetail,
    ) -> TrackerController:
        ctrl = TrackerController(
            spec={
                "tx_station": tx_station,
                "rx_stations": rx_stations,
                "exp_detail": exp_detail,
            },
            state={
                "spobj_time": time,
                "spobj_states": space_object_states,
            },
        )

        return ctrl

    @classmethod
    def from_space_object(
        cls,
        spobj: SpaceObject,
        epoch: Datetime_Like,
        tx_station: Station,
        rx_stations: t.Sequence[Station],
        exp_detail: ExperimentDetail,
    ) -> TrackerController:
        exp_detail.update(
            {"stn_pairs": [(tx_station.uid, rx_station.uid) for rx_station in rx_stations]}
        )

        ctrl = TrackerController(
            spec={
                "tx_station": tx_station,
                "rx_stations": rx_stations,
                "exp_detail": exp_detail,
                "spobj": spobj,
                "epoch": epoch,
            },
            state=None,
        )

        return ctrl

    def compute_ecef_states(
        self, start_time: Datetime_Like, end_time: Datetime_Like, slice_duration: Timedelta_Like
    ):
        """Do the computation then update the `state` property and return `self`."""

        if "spobj" not in self.spec:
            raise RuntimeError(
                "Cannot compute space object ECEF states without `spobj` in the `spec` prop."
            )
        if "epoch" not in self.spec:
            raise RuntimeError(
                "Cannot compute space object ECEF states without `epoch` in the `spec` prop."
            )

        exp_detail: ExperimentDetail = self.spec["exp_detail"]

        # NOTE: for `np.arange` 'stop param,
        #   - we subtract 'slice_duration' so that only full slice are included
        #   - and add `+1` so that slice with time range `('end_time - 'slice_duration', 'end_time')` is included
        time: npt.NDArray[Datetime64_us] = np.arange(
            to_datetime64_us(start_time),
            to_datetime64_us(end_time) - to_timedelta64_us(slice_duration) + 1,
            exp_detail["slice_duration"],
        )
        dt: npt.NDArray[Timedelta64_us] = time - to_datetime64_us(self.spec["epoch"])
        dsec = t.cast(npt.NDArray[Float64_as_sec], dt.astype(np.float64) / 1e6)

        ecefs = self.spec["spobj"].get_state(dsec)

        self.state = {
            "spobj_time": time,
            "spobj_states": ecefs,
        }

        return self

    def generate(
        self, start_time: Datetime_Like | None = None, end_time: Datetime_Like | None = None
    ) -> Schedule:
        """
        Generate the schedules.
        `start_time` and `end_time` should be omitted if this instance is created from `TrackerController.from_ecef_states`
        """

        if start_time is not None and end_time is not None:
            self.compute_ecef_states(
                start_time, end_time, self.spec["exp_detail"]["slice_duration"]
            )
            state = t.cast(State, self.state)
        elif self.state is None:
            raise RuntimeError(
                "Cannot generate without valid state property."
                + " Please either provide the `start_time` and `end_time` param"
                + " or ensure it is set correctly using methods like `compute_ecef_states` or proper constructors."
            )
        else:
            state = self.state

        output = generate_from_state(spec=self.spec, state=state)
        self._cached_output = output

        return output

    # TODO: can be removed?
    def plot(self, start_time: Datetime_Like | None = None, end_time: Datetime_Like | None = None):
        global plot_state_and_output

        if self.state is None:
            if start_time is not None and end_time is not None:
                self.compute_ecef_states(
                    start_time, end_time, self.spec["exp_detail"]["slice_duration"]
                )
                state = t.cast(State, self.state)
            else:
                raise RuntimeError(
                    "Cannot plot TrackerController without valid state property."
                    + " Please either provide the `start_time` and `end_time` param"
                    + " or ensure it is set correctly using methods like `compute_ecef_states` or proper constructors."
                )
        else:
            state = self.state

        if self._cached_output is None:
            if start_time is not None and end_time is not None:
                cached_output = self.generate(start_time, end_time)
                self._cached_output = cached_output
            else:
                raise RuntimeError(
                    "Cannot plot TrackerController without valid output cache."
                    + " Please either provide the `start_time` and `end_time` param"
                    + " or ensure it is set correctly using methods like `compute_ecef_states` or proper constructors."
                )
        else:
            cached_output = self._cached_output

        p = plot_state_and_output(state, cached_output)
        return p
