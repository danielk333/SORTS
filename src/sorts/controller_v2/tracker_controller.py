import logging, typing as t
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from astropy.time import Time
from sorts.radar.tx_rx import Station
from sorts.frames import ecef_to_enu
from sorts.types import EcefStates, Float_as_deg, Datetime64_us, EcefCoordinates, EnuCoordinates
from sorts.radar.tx_rx import Station
from sorts.schedule_v2 import Schedule, ExperimentDetail
from sorts.controller_v2.controller_protocol import ControllerProtocol
from sorts.controller_v2 import pointing_patterns

logger = logging.getLogger(__name__)


class TrackerControllerOutput(t.NamedTuple):
    tx_schedule: Schedule
    rx_schedules: t.Sequence[Schedule]


# TODO: add a version of `TrackerController`
#   that takes a space object as input, or, a classmethod `from_space_object`?
@dataclass(kw_only=True)
class TrackerController:
    """
    Generate pointing schedule that tracks a space object.

    Note:
    - `azimuth` and `elevation` are measured in degree and are centered on the radar station
    - `azimuth_range` is a right-open interval `[min, max)`; defaults to `(0.0, 360.0)`
    - `elevation_range` is a closed interval `[min, max]` ; defaults to `(0.0, 90.0)`

    TODO: this is WIP
    """

    tx_station: Station
    rx_stations: t.Sequence[Station]
    exp_detail: ExperimentDetail

    time: npt.NDArray[Datetime64_us]
    space_object_states: EcefStates
    azimuth_range: tuple[Float_as_deg, Float_as_deg] = (0.0, 360.0)
    elevation_range: tuple[Float_as_deg, Float_as_deg] = (0.0, 90.0)

    def __post_init__(self):
        self._cached_output: TrackerControllerOutput | None = None

    def generate(self) -> TrackerControllerOutput:
        # TODO: remove?
        # start_time_np = t.cast(np.datetime64, start_time.to_value("datetime64"))
        # end_time_np = t.cast(np.datetime64, end_time.to_value("datetime64"))
        # start_time_np = t.cast(np.datetime64, start_time.to_value("datetime64")).astype(
        #     "datetime64[us]"
        # )
        # end_time_np = t.cast(np.datetime64, end_time.to_value("datetime64")).astype(
        #     "datetime64[us]"
        # )

        # time_arr: npt.NDArray[Datetime64_us] = np.arange(
        #     start_time_np, end_time_np, np.timedelta64(self.control_slice_duration * 1e6, "us")
        # )

        # control_slice_duration: Float_as_sec =

        tx_pointings = point_ecef(self.tx_station, self.space_object_states[:3])
        rxs_pointings = [
            point_ecef(rx_station, self.space_object_states[:3]) for rx_station in self.rx_stations
        ]

        sch_len = len(self.time)

        tx_sch = Schedule(
            meta={self.exp_detail.id: self.exp_detail},
            stt_tstmp_us=self.time,
            exp_num=np.full(sch_len, self.exp_detail.id, dtype=np.int64),
            pointing_az=tx_pointings[0],
            pointing_el=tx_pointings[1],
        )

        rx_schs = [
            Schedule(
                meta={self.exp_detail.id: self.exp_detail},
                stt_tstmp_us=self.time,
                exp_num=np.full(sch_len, self.exp_detail.id, dtype=np.int64),
                pointing_az=rx_pointings[0],
                pointing_el=rx_pointings[1],
            )
            for rx_pointings in rxs_pointings
        ]

        self._cached_output = TrackerControllerOutput(tx_sch, rx_schs)
            return self._cached_output

    # TODO: WIP
    def plot(self):
        import plotly.express as px
        import plotly.graph_objects as go

        r = self.generate()

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(values=["A Scores", "B Scores"]),
                    # cells=dict(values=[[100, 90, 80, 90], [95, 85, 75, 95]]))
                    cells=dict(values=[r.tx_schedule.stt_tstmp_us, [95, 85, 75, 95]]),
                )
            ]
        )

        return fig
        # fig.show()


def point_ecef(station: Station, point: EcefCoordinates) -> EnuCoordinates:
    """Generate ENU coordinates relative to the radar station from points in ECEF coordinate."""

    k: EnuCoordinates = ecef_to_enu(
        station.ecef_lat,
        station.ecef_lon,
        station.ecef_alt,
        point,
        degrees=True,
    )
    k_norm = np.linalg.norm(k, axis=0)

    k = k / k_norm

    # self.beam.point(k) # TODO: check if this method call is needed

    return k
