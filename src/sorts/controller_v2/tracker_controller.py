from __future__ import annotations
import logging, typing as t
import numpy as np
import numpy.typing as npt
import bokeh.layouts as bokeh_layouts
from sorts.frames import cart_to_sph
from sorts.space_object import SpaceObject
from sorts.radar.tx_rx import Station
from sorts.types import (
    EcefStates,
    Datetime64_us,
    Timedelta64_us,
    Float64_as_sec,
    AzelrCoordinates_DegM,
    Datetime_Like,
)
from sorts.utils import wrap_azimuths_elevations, to_datetime64_us
from sorts import plots
from sorts import schedule_v2 as schedule
from sorts.schedule_v2 import Schedule, ExperimentDetail

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


class Output(t.NamedTuple):
    """tuple of `(tx_schedule, [rx_schedule, ...])`"""

    tx_schedule: Schedule
    rx_schedules: t.Sequence[Schedule]


def generate_from_state(spec: Spec, state: State) -> Output:
    # generate pointings
    tx_pointings: AzelrCoordinates_DegM = cart_to_sph(
        spec["tx_station"].enu(state["spobj_states"][:3]),
        degrees=True,
    )
    rxs_pointings: list[AzelrCoordinates_DegM] = [
        cart_to_sph(rx_station.enu(state["spobj_states"][:3]), degrees=True)
        for rx_station in spec["rx_stations"]
    ]

    # filter out invalid values
    is_out_of_tx_el_range_mask = tx_pointings[1] < spec["tx_station"].min_elevation
    is_out_of_rxs_el_range_mask = [
        ((rx_pointings[1] < rx_station.min_elevation))
        for rx_station, rx_pointings in zip(spec["rx_stations"], rxs_pointings)
    ]
    is_out_of_el_range_mask = np.logical_and.reduce(
        [is_out_of_tx_el_range_mask, *is_out_of_rxs_el_range_mask]
    )

    tx_pointings = tx_pointings[:, ~is_out_of_el_range_mask]

    for idx, rx_pointings in enumerate(rxs_pointings):
        rxs_pointings[idx] = rx_pointings[:, ~is_out_of_el_range_mask]

    # apply wrapping
    tx_pointings[0], tx_pointings[1] = wrap_azimuths_elevations(tx_pointings[0], tx_pointings[1])

    for rx_pointings in rxs_pointings:
        rx_pointings[0], rx_pointings[1] = wrap_azimuths_elevations(
            rx_pointings[0], rx_pointings[1]
        )

    sch_time = state["spobj_time"][~is_out_of_el_range_mask]
    sch_len = len(sch_time)

    tx_sch = Schedule(
        exp_detail_map={spec["exp_detail"]["id"]: spec["exp_detail"]},
        start_time=sch_time,
        exp_num=np.full(sch_len, spec["exp_detail"]["id"], dtype=np.int64),
        pointing_az=tx_pointings[0],
        pointing_el=tx_pointings[1],
    )
    schedule.validate_schedule_length(tx_sch)

    rx_schs = [
        schedule.validate_schedule_length(
            Schedule(
                exp_detail_map={spec["exp_detail"]["id"]: spec["exp_detail"]},
                start_time=sch_time,
                exp_num=np.full(sch_len, spec["exp_detail"]["id"], dtype=np.int64),
                pointing_az=rx_pointings[0],
                pointing_el=rx_pointings[1],
            )
        )
        for rx_pointings in rxs_pointings
    ]

    output = Output(tx_sch, rx_schs)
    return output


def plot_state_and_output(
    state: State,
    output: Output,
):
    pos_plot = plots.ecef_states_positions_plot(state["spobj_states"])
    pos_plot.title = "ecef_states_positions_plot"

    rx_skyplot_plots = []
    for idx, rx_schedule in enumerate(output.rx_schedules):
        rx_skyplot_plot = plots.azel_skyplot(
            rx_schedule["pointing_az"],
            rx_schedule["pointing_el"],
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

        self._cached_output: Output | None = None
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

    def compute_ecef_states(self, start_time: Datetime_Like, end_time: Datetime_Like):
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

        time: npt.NDArray[Datetime64_us] = np.arange(
            to_datetime64_us(start_time),
            to_datetime64_us(end_time),
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
    ) -> Output:
        """
        Generate the schedules.
        `start_time` and `end_time` should be omitted if this instance is created from `TrackerController.from_ecef_states`
        """

        global generate_from_state

        if start_time is not None and end_time is not None:
            self.compute_ecef_states(start_time, end_time)
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

    def plot(self, start_time: Datetime_Like | None = None, end_time: Datetime_Like | None = None):
        global plot_state_and_output

        if self.state is None:
            if start_time is not None and end_time is not None:
                self.compute_ecef_states(start_time, end_time)
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
