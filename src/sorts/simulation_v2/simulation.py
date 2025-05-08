import logging, typing as t
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import pyorb
import sorts
from sorts import passes_v2 as passes
from sorts import scheduler_v2 as scheduler
from sorts import detection_config
from sorts.simulation_v2.experiment_detail import ExperimentDetail

logger = logging.getLogger(__name__)


class SpaceObjectsDtSamplerS(t.Protocol):
    def __call__(
        self, orbit: pyorb.Orbit, start_time: datetime, end_time: datetime
    ) -> npt.NDArray[np.float64]: ...


Dcfg = t.TypeVar("Dcfg", bound=detection_config.DetectionConfig)


# TODO: split into 'SimulationConfig' and 'Simulation'?
@dataclass(kw_only=True)
class Simulation(t.Generic[Dcfg]):
    epoch: datetime
    start_time: datetime
    end_time: datetime
    detection_config: Dcfg
    space_objects: list[sorts.SpaceObject]  # TODO: 'spobjs' vs 'space_objects'?

    # TODO: support different sampler for different obj?
    space_objects_dt_sampler_s: SpaceObjectsDtSamplerS
    """A function with signature `(start_time: Time, end_time: Time) -> npt.NDArray[np.float64]`"""

    exp_num_map: dict[int, ExperimentDetail]

    def __post_init__(self):
        self.spobjs_smpl_dt_s_arr: list[npt.NDArray[np.float64]] = [
            self.space_objects_dt_sampler_s(
                spobj.state,
                self.start_time,
                self.end_time,
            )
            for spobj in self.space_objects
        ]

    def find_passes(self):
        spobjs_smpl_states = [
            spobj.get_state(spobj_smpl_dt_s_arr)
            for (spobj_smpl_dt_s_arr, spobj) in zip(self.spobjs_smpl_dt_s_arr, self.space_objects)
        ]

        passes_foreach_spobj = [
            self.detection_config.find_passes(
                dt_s_arr=spobj_smpl_dt_s_arr,
                space_object_states=spobj_smpl_states,
                epoch=self.epoch,
            )
            for (spobj_smpl_dt_s_arr, spobj_smpl_states) in zip(
                self.spobjs_smpl_dt_s_arr, spobjs_smpl_states
            )
        ]

        return passes_foreach_spobj

    def calculate_observations(self):
        passes = self.find_passes()
        mask = get_schedule_mask_by_pass(self.detection_config.tx_schedule, passes[0][0])

        obs = self.detection_config.calculate_observations(
            space_objects=self.space_objects, epoch=self.epoch, schedule_mask=mask
        )
        return obs, mask  # TODO: returning `mask` is just a quick tmp workaround

    # NOTE: kept for ref until the class is stablized
    # def run(self) -> dict[RadarStationCompositeKey, dict]: ...


# TODO: implement or remove
@dataclass(kw_only=True)
class SimulationResult:
    pass


def get_schedule_mask_by_pass(schedule: scheduler.Schedule, pass_obj: passes.Pass):
    # TODO: might need to check for `end_tstmp_us` when it is added to `Schedule` type
    sch_dt_s_arr_pass_mask: npt.NDArray[np.bool] = np.logical_and(
        schedule.stt_tstmp_us >= pass_obj.t[0],
        schedule.stt_tstmp_us <= pass_obj.t[-1],
    )

    return sch_dt_s_arr_pass_mask


__all__ = ["Simulation", "SimulationResult", "SpaceObjectsDtSamplerS"]
