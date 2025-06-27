from __future__ import annotations
import typing as t
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import numpy.typing as npt
import sorts
from sorts.interpolation import Interpolator
from sorts.radar.radars.composite_key import RadarStationCompositeKey
from sorts.radar.tx_rx import Station
from sorts.types import Datetime64_us, EcefStates
from sorts import scheduler_v2 as scheduler
from sorts.detection_systems import ExperimentDetail, Observation


class DetectionSystemProtocol(t.Protocol):
    def find_passes_time_ranges(
        self,
        dt_s_arr: npt.NDArray[np.float64],
        space_object_states: npt.NDArray[np.float64],
        epoch: datetime,
    ) -> t.Sequence[tuple[Datetime64_us, Datetime64_us]]: ...

    def get_schedule_mask_by_time_range(
        self, time_range: tuple[Datetime64_us, Datetime64_us]
    ) -> npt.NDArray[np.bool]: ...

    def calculate_observation(
        self,
        space_object: sorts.SpaceObject,
        space_object_states_interpolator: Interpolator,
        epoch: datetime,
        schedule_mask: npt.NDArray[np.bool] | None,
    ) -> list[Observation]: ...
