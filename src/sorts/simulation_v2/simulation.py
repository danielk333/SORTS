import logging, typing as t
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import pyorb
import sorts
from sorts import detection_systems
from sorts.interpolation import Interpolator
from sorts.types import Float64_as_sec, EcefStates

logger = logging.getLogger(__name__)


class SpaceObjectsDtSamplerS(t.Protocol):
    def __call__(
        self, orbit: pyorb.Orbit, start_time: datetime, end_time: datetime
    ) -> npt.NDArray[np.float64]: ...


Dsys = t.TypeVar("Dsys", bound=detection_systems.DetectionSystemProtocol)


# TODO: split into 'SimulationConfig' and 'Simulation'?
#   - `Simulation` is evolving into more than just a pure data class
#   - we can move the pure data part into `SimulationConfig` and retain the top level api as `Simulation`
@dataclass(kw_only=True)
class Simulation(t.Generic[Dsys]):
    epoch: datetime
    start_time: datetime
    end_time: datetime
    detection_system: Dsys  # NOTE: generics is needed to retain the original type
    space_objects: list[sorts.SpaceObject]

    # TODO: support different sampler for different obj?
    # TODO: probably taking a function + a args/kwargs obj is more pythonic
    space_objects_dt_sampler_s: SpaceObjectsDtSamplerS
    """A function with signature `(start_time: Time, end_time: Time) -> npt.NDArray[np.float64]`"""

    # TODO: rename to `space_objects_dt_s_interpolator`
    space_objects_dt_interpolator_s: type[Interpolator]

    def __post_init__(self):
        # TODO: these are short cuts to access internal states of `Simulation` (e.g. for plotting)
        #   need to be removed or exposed more properly
        self._spobjs_states_interps: list[Interpolator] = []

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

    def calculate_observations(self) -> list[list[detection_systems.Observation]]:
        obss: list[list[detection_systems.Observation]] = []

        spobjs_smpl_dt_s_arr, spobjs_smpl_states = self.propagate_and_sample_space_objects_states()
        spobjs_states_interps = [
            create_space_object_states_interpolator(
                self.space_objects_dt_interpolator_s, spobj_smpl_states, spobj_smpl_dt_s_arr
            )
            for spobj_smpl_dt_s_arr, spobj_smpl_states in zip(
                spobjs_smpl_dt_s_arr, spobjs_smpl_states
            )
        ]
        self._spobjs_states_interps = spobjs_states_interps

        for spobj, spobj_smpl_dt_s_arr, spobj_smpl_states, spobj_states_interp in zip(
            self.space_objects, spobjs_smpl_dt_s_arr, spobjs_smpl_states, spobjs_states_interps
        ):
            time_ranges = self.detection_system.find_passes_time_ranges(
                dt_s_arr=spobj_smpl_dt_s_arr,
                space_object_states=spobj_smpl_states,
                epoch=self.epoch,
            )
            masks_for_spobj: list[npt.NDArray[np.bool]] = [
                self.detection_system.get_schedule_mask_by_time_range(time_range)
                for time_range in time_ranges
            ]

            # TODO: improvements needed; this is only works for StxSrx case, where calculate_observation gives out 1 element list
            # TODO: use for-loop + mutation instead of nested for-comprehension for better readability
            obs_for_spobj: list[detection_systems.Observation] = [
                obs
                # TODO: remove enumerate; it was used as tmp replacement for `for mask in masks_for_spobj`
                for time_range_idx, time_range in enumerate(time_ranges)
                for obs in self.detection_system.calculate_observation(
                    space_object=spobj,
                    space_object_states_interpolator=spobj_states_interp,
                    epoch=self.epoch,
                    schedule_mask=masks_for_spobj[time_range_idx],
                    time_range=time_range,
                )
            ]

            obss.append(obs_for_spobj)

        return obss

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
