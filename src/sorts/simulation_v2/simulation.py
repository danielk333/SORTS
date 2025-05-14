import logging, typing as t
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import pyorb
import sorts
from sorts import detection_config as detection_config_
from sorts.interpolation import Interpolator
from sorts.types import Float64_as_sec, EcefStates

logger = logging.getLogger(__name__)


class SpaceObjectsDtSamplerS(t.Protocol):
    def __call__(
        self, orbit: pyorb.Orbit, start_time: datetime, end_time: datetime
    ) -> npt.NDArray[np.float64]: ...


Dcfg = t.TypeVar("Dcfg", bound=detection_config_.DetectionConfigProtocol)


# TODO: split into 'SimulationConfig' and 'Simulation'?
@dataclass(kw_only=True)
class Simulation(t.Generic[Dcfg]):
    epoch: datetime
    start_time: datetime
    end_time: datetime
    detection_config: Dcfg  # NOTE: generics is needed to retain the original type
    space_objects: list[sorts.SpaceObject]

    # TODO: support different sampler for different obj?
    # TODO: probably taking a function + a args/kwargs obj is more pythonic
    space_objects_dt_sampler_s: SpaceObjectsDtSamplerS
    """A function with signature `(start_time: Time, end_time: Time) -> npt.NDArray[np.float64]`"""

    space_objects_dt_interpolator_s: type[Interpolator]

    def propagate_and_sample_space_objects_states(self):
        """
        Use the sampler the get the delta time of space object within the simulation `start_time` and `end_time`
        """

        spobjs_smpl_dt_s_arr: list[npt.NDArray[Float64_as_sec]] = [
            self.space_objects_dt_sampler_s(
                spobj.state,
                self.start_time,
                self.end_time,
            )
            for spobj in self.space_objects
        ]

        spobjs_smpl_states: list[EcefStates] = [
            spobj.get_state(spobj_smpl_dt_s_arr)
            for spobj, spobj_smpl_dt_s_arr in zip(self.space_objects, spobjs_smpl_dt_s_arr)
        ]

        return spobjs_smpl_dt_s_arr, spobjs_smpl_states

    def calculate_observations(self):
        masks: list[list[npt.NDArray[np.bool]]] = []
        obss: list[list[detection_config_.Observation]] = []

        spobjs_smpl_dt_s_arr, spobjs_smpl_states = self.propagate_and_sample_space_objects_states()
        spobjs_states_interps = [
            create_space_object_states_interpolator(
                self.space_objects_dt_interpolator_s, spobj_smpl_states, spobj_smpl_dt_s_arr
            )
            for spobj_smpl_dt_s_arr, spobj_smpl_states in zip(
                spobjs_smpl_dt_s_arr, spobjs_smpl_states
            )
        ]

        for spobj, spobj_smpl_dt_s_arr, spobj_states_interp in zip(
            self.space_objects, spobjs_smpl_dt_s_arr, spobjs_states_interps
        ):
            time_ranges = self.detection_config.find_passes_time_ranges(
                dt_s_arr=spobj_smpl_dt_s_arr,
                space_object_states=spobj_states_interp.get_state(spobj_smpl_dt_s_arr),
                epoch=self.epoch,
            )
            masks_for_spobj: list[npt.NDArray[np.bool]] = [
                self.detection_config.get_schedule_mask_by_time_range(time_range)
                for time_range in time_ranges
            ]

            obs_for_spobj: list[detection_config_.Observation] = [
                self.detection_config.calculate_observation(
                    space_object=spobj, epoch=self.epoch, schedule_mask=mask
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


def create_space_object_states_interpolator(
    interpolator: type[Interpolator],
    states: EcefStates,
    sample_dt_s_arr: npt.NDArray[Float64_as_sec],
):
    states_interp = interpolator(states, sample_dt_s_arr)
    return states_interp


__all__ = ["Simulation", "SimulationResult", "SpaceObjectsDtSamplerS"]
