import logging, typing as t
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import pyorb
import sorts
from sorts import detection_config as detection_config_

logger = logging.getLogger(__name__)


class SpaceObjectsDtSamplerS(t.Protocol):
    def __call__(
        self, orbit: pyorb.Orbit, start_time: datetime, end_time: datetime
    ) -> npt.NDArray[np.float64]: ...


# TODO: split into 'SimulationConfig' and 'Simulation'?
@dataclass(kw_only=True)
class Simulation:
    epoch: datetime
    start_time: datetime
    end_time: datetime
    detection_config: detection_config_.DetectionConfigProtocol
    space_objects: list[sorts.SpaceObject]

    # TODO: support different sampler for different obj?
    space_objects_dt_sampler_s: SpaceObjectsDtSamplerS
    """A function with signature `(start_time: Time, end_time: Time) -> npt.NDArray[np.float64]`"""

    def __post_init__(self):
        self.spobjs_smpl_dt_s_arr: list[npt.NDArray[np.float64]] = [
            self.space_objects_dt_sampler_s(
                spobj.state,
                self.start_time,
                self.end_time,
            )
            for spobj in self.space_objects
        ]

    def calculate_observations(self):
        masks: list[list[npt.NDArray[np.bool]]] = []
        obss: list[list[detection_config_.Observation]] = []

        for spobj_idx, spobj in enumerate(self.space_objects):
            spobj_smpl_dt_s_arr = self.spobjs_smpl_dt_s_arr[spobj_idx]

            time_ranges = self.detection_config.find_passes_time_ranges(
                dt_s_arr=spobj_smpl_dt_s_arr,
                space_object_states=spobj.get_state(spobj_smpl_dt_s_arr),
                epoch=self.epoch,
            )
            masks_for_spobj: list[npt.NDArray[np.bool]] = [
                self.detection_config.get_schedule_mask_by_time_range(time_range)
                for time_range in time_ranges
            ]

            obs_for_spobj: list[detection_config_.Observation] = [
                self.detection_config.calculate_observation(
                    space_object=self.space_objects[spobj_idx], epoch=self.epoch, schedule_mask=mask
                )
                for mask in masks_for_spobj
            ]

            masks.append(masks_for_spobj)
            obss.append(obs_for_spobj)

        return obss, masks  # TODO: returning `masks` is just a quick tmp workaround

    # NOTE: kept for ref until the class is stablized
    # def run(self) -> dict[RadarStationCompositeKey, dict]: ...


# TODO: implement or remove
@dataclass(kw_only=True)
class SimulationResult:
    pass


__all__ = ["Simulation", "SimulationResult", "SpaceObjectsDtSamplerS"]
