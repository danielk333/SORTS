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
    Datetime_like,
)
from sorts.utils import wrap_azimuths_elevations, to_datetime64_us
from sorts import plots
from sorts.schedule_v2 import Schedule, ExperimentDetail

logger = logging.getLogger(__name__)


class State(t.TypedDict):
    tx_station: Station
    rx_stations: t.Sequence[Station]
    exp_detail: ExperimentDetail
    spobj_time: npt.NDArray[Datetime64_us]
    spobj_states: EcefStates


# a workaround to get reference to TypedDict keys
StateKey = t.Literal["tx_station", "rx_stations", "exp_detail", "spobj_time", "spobj_states"]
assert set(t.get_args(StateKey)) == State.__annotations__.keys()


class Output(t.NamedTuple):
    """tuple of `(tx_schedule, [rx_schedule, ...])`"""

    tx_schedule: Schedule
    rx_schedules: t.Sequence[Schedule]


def generate_from_state(state: State) -> Output:
    # generate pointings
    tx_pointings: AzelrCoordinates_DegM = cart_to_sph(
        state["tx_station"].enu(state["spobj_states"][:3]),
        degrees=True,
    )
    rxs_pointings: list[AzelrCoordinates_DegM] = [
        cart_to_sph(rx_station.enu(state["spobj_states"][:3]), degrees=True)
        for rx_station in state["rx_stations"]
    ]

    # filter out invalid values
    is_out_of_tx_el_range_mask = tx_pointings[1] < state["tx_station"].min_elevation
    is_out_of_rxs_el_range_mask = [
        ((rx_pointings[1] < rx_station.min_elevation))
        for rx_station, rx_pointings in zip(state["rx_stations"], rxs_pointings)
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
        meta={state["exp_detail"]["id"]: state["exp_detail"]},
        start_time=sch_time,
        exp_num=np.full(sch_len, state["exp_detail"]["id"], dtype=np.int64),
        pointing_az=tx_pointings[0],
        pointing_el=tx_pointings[1],
    )

    rx_schs = [
        Schedule(
            meta={state["exp_detail"]["id"]: state["exp_detail"]},
            start_time=sch_time,
            exp_num=np.full(sch_len, state["exp_detail"]["id"], dtype=np.int64),
            pointing_az=rx_pointings[0],
            pointing_el=rx_pointings[1],
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
            rx_schedule.pointing_az,
            rx_schedule.pointing_el,
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

    The preferred way to create instances of this class is via its class methods (e.g. `TrackerController.from_space_object`).
    """

    def __init__(self, state: State | None = None):
        self.state = state

        self._partial_state: dict[t.Union[StateKey, str], t.Any] = {}
        """A partial `self.state` with potentially extra fields for internal manipulations"""

        self._cached_output: Output | None = None

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
            {
                "tx_station": tx_station,
                "rx_stations": rx_stations,
                "exp_detail": exp_detail,
                "spobj_time": time,
                "spobj_states": space_object_states,
            }
        )

        return ctrl

    @classmethod
    def from_space_object(
        cls,
        spobj: SpaceObject,
        epoch: Datetime_like,
        tx_station: Station,
        rx_stations: t.Sequence[Station],
        exp_detail: ExperimentDetail,
    ) -> TrackerController:
        ctrl = TrackerController()

        ctrl._partial_state["spobj"] = spobj
        ctrl._partial_state["epoch"] = epoch
        ctrl._partial_state["tx_station"] = tx_station
        ctrl._partial_state["rx_stations"] = rx_stations
        ctrl._partial_state["exp_detail"] = exp_detail

        return ctrl

    def compute_state_from_time_range(
        self, start_time: Datetime_like, end_time: Datetime_like, epoch: Datetime_like
    ):
        """Update the `state` and return `self`."""

        exp_detail: ExperimentDetail = self._partial_state["exp_detail"]

        time: npt.NDArray[Datetime64_us] = np.arange(
            to_datetime64_us(start_time),
            to_datetime64_us(end_time),
            exp_detail["slice_duration"],
        )
        dt: npt.NDArray[Timedelta64_us] = time - to_datetime64_us(epoch)
        dsec = t.cast(npt.NDArray[Float64_as_sec], dt.astype(np.float64) / 1e6)
        ecefs = self._partial_state["spobj"].get_state(dsec)

        state: State = {
            "tx_station": self._partial_state["tx_station"],
            "rx_stations": self._partial_state["rx_stations"],
            "exp_detail": self._partial_state["exp_detail"],
            "spobj_time": time,
            "spobj_states": ecefs,
        }

        self.state = state

        return self

    def generate(self, start_time: Datetime_like, end_time: Datetime_like) -> Output:
        global generate_from_state

        if self.state is None:
            epoch: Datetime_like = self._partial_state["epoch"]
            self.compute_state_from_time_range(start_time, end_time, epoch)
            state = t.cast(State, self.state)
        else:
            state = self.state

        output = generate_from_state(state)
        self._cached_output = output

        return output

    def plot(self, start_time: Datetime_like | None = None, end_time: Datetime_like | None = None):
        global plot_state_and_output

        if self.state is None:
            if start_time is not None and end_time is not None:
                epoch: Datetime_like = self._partial_state["epoch"]
                self.compute_state_from_time_range(start_time, end_time, epoch)
                state = t.cast(State, self.state)
            else:
                raise RuntimeError(
                    "Cannot plot TrackerController without valid state property."
                    + " Please either call method `compute_state_from_time_range` beforehand"
                    + " or provide the `start_time` and `end_time` param"
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
                    + " Please either call method `generate` beforehand"
                    + " or provide the `start_time` and `end_time` param"
                )
        else:
            cached_output = self._cached_output

        p = plot_state_and_output(state, cached_output)
        return p
