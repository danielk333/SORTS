import logging, typing as t
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import bokeh.layouts as bokeh_layouts
from sorts.frames import ecef_to_enu, cart_to_sph
from sorts.radar.tx_rx import Station
from sorts.types import (
    EcefStates,
    Float_as_deg,
    Datetime64_us,
    EcefCoordinates,
    EnuCoordinates,
    AzelrCoordinates_DegM,
    Datetime_like,
)
from sorts.utils import wrap_azimuths_elevations, to_datetime64_us
from sorts import plots
from sorts.schedule_v2 import Schedule, ExperimentDetail

logger = logging.getLogger(__name__)


class TrackerControllerOutput(t.NamedTuple):
    tx_schedule: Schedule
    rx_schedules: t.Sequence[Schedule]


# TODO: add a version of `TrackerController`
#   that takes a space object as input, or, a classmethod `from_space_object`?
@dataclass(kw_only=True)
class TrackerController:
    # TODO: this is WIP
    """
    Generate pointing schedule that tracks a space object.

    Note:
    - `azimuth` and `elevation` are measured in degree and are centered on the radar station

    """

    tx_station: Station
    rx_stations: t.Sequence[Station]
    exp_detail: ExperimentDetail

    time: npt.NDArray[Datetime64_us]
    space_object_states: EcefStates
    min_elevation: Float_as_deg = 0.0

    def __post_init__(self):
        self._cached_output: TrackerControllerOutput | None = None

    def generate(self) -> TrackerControllerOutput:
        # generate pointings
        tx_pointings: AzelrCoordinates_DegM = cart_to_sph(
            self.tx_station.enu(self.space_object_states[:3]),
            degrees=True,
        )
        rxs_pointings: list[AzelrCoordinates_DegM] = [
            cart_to_sph(rx_station.enu(self.space_object_states[:3]), degrees=True)
            for rx_station in self.rx_stations
        ]

        # filter out invalid values
        is_out_of_tx_el_range_mask = tx_pointings[1] < self.min_elevation
        is_out_of_rxs_el_range_mask = [
            ((rx_pointings[1] < self.min_elevation)) for rx_pointings in rxs_pointings
        ]
        is_out_of_el_range_mask = np.logical_and.reduce(
            [is_out_of_tx_el_range_mask, *is_out_of_rxs_el_range_mask]
        )

        tx_pointings = tx_pointings[:, ~is_out_of_el_range_mask]

        for idx, rx_pointings in enumerate(rxs_pointings):
            rxs_pointings[idx] = rx_pointings[:, ~is_out_of_el_range_mask]

        # apply wrapping
        tx_pointings[0], tx_pointings[1] = wrap_azimuths_elevations(
            tx_pointings[0], tx_pointings[1]
        )

        for rx_pointings in rxs_pointings:
            rx_pointings[0], rx_pointings[1] = wrap_azimuths_elevations(
                rx_pointings[0], rx_pointings[1]
            )

        sch_time = self.time[~is_out_of_el_range_mask]
        sch_len = len(sch_time)

        tx_sch = Schedule(
            meta={self.exp_detail.id: self.exp_detail},
            start_time=sch_time,
            exp_num=np.full(sch_len, self.exp_detail.id, dtype=np.int64),
            pointing_az=tx_pointings[0],
            pointing_el=tx_pointings[1],
        )

        rx_schs = [
            Schedule(
                meta={self.exp_detail.id: self.exp_detail},
                start_time=sch_time,
                exp_num=np.full(sch_len, self.exp_detail.id, dtype=np.int64),
                pointing_az=rx_pointings[0],
                pointing_el=rx_pointings[1],
            )
            for rx_pointings in rxs_pointings
        ]

        self._cached_output = TrackerControllerOutput(tx_sch, rx_schs)
        return self._cached_output

    # TODO: implement `start_time`, `end_time` param
    def plot(self, start_time: Datetime_like, end_time: Datetime_like):
        # start_time = to_datetime64_us(start_time)
        # end_time = to_datetime64_us(end_time)
        if self._cached_output is None:
            # TODO: type checker couldn't pickup `self.generate()` `_cached_output` so an assigned is used here
            self._cached_output = self.generate()

        pos_plot = plots.ecef_states_positions_plot(self.space_object_states)
        pos_plot.title = "ecef_states_positions_plot"

        rx_skyplot_plots = []
        for idx, rx_schedule in enumerate(self._cached_output.rx_schedules):
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


def point_ecef(station: Station, point: EcefCoordinates) -> EnuCoordinates:
    """Generate ENU coordinates relative to the radar station from points in ECEF coordinate."""
    # NOTE: based on `src/sorts/radar/tx_rx.py` `L133` `Station.point_ecef`

    k: EnuCoordinates = ecef_to_enu(
        station.ecef_lat,
        station.ecef_lon,
        station.ecef_alt,
        point,
        degrees=True,
    )
    k_norm = np.linalg.norm(k, axis=0)

    k = k / k_norm

    return k
