from __future__ import annotations
import logging, typing as t, pickle, traceback, time
from pathlib import Path
from dataclasses import dataclass
import numpy.typing as npt
import pyorb
import sorts
from tqdm import tqdm
from mpi4py import MPI
from sorts import radar, schedule, controller
from sorts.interpolation import Interpolator
from sorts.utils import to_datetime64_us
from sorts.types import Datetime_Like, Float64_as_sec
from sorts.schedule import Schedule, ExperimentDetailMap
from sorts.simulation.stx_mrx_simulation.simulation_unit import (
    SimulationUnit,
    FromPassagesOverTxRxStationPairParam,
    Observation,
)
from . import funcs

logger = logging.getLogger(__name__)


@dataclass
class WorkerJobAssignment:
    param: FromPassagesOverTxRxStationPairParam
    persist_dir: Path


class _MpiMsg:
    exit_ok: t.Final = "exit_ok"
    terminate: t.Final = "terminate"
    worker_job_assignment = WorkerJobAssignment


type SimulationEnvironment = t.Mapping[str, t.Any]
"""
A mapping of `str` to `Any`, with at least these items:
```
{
    "spec_by_controllers": SpecByControllers,
}
```
"""

sim_unit_fname_tpl = "sim_unit.{id}.pickle"


class SpaceObjectDsecSampler(t.Protocol):
    def __call__(
        self, orbit: pyorb.Orbit, start_time: Datetime_Like, end_time: Datetime_Like
    ) -> npt.NDArray[Float64_as_sec]: ...


class Spec(t.TypedDict):
    """A TypedDict of params"""

    station_map: dict[radar.StationId, radar.Station]
    station_id_pairs: list[tuple[radar.StationId, radar.StationId]]
    schedule: Schedule
    exp_detail_map: ExperimentDetailMap
    epoch: Datetime_Like
    start_time: Datetime_Like
    end_time: Datetime_Like
    space_objects: t.Sequence[sorts.SpaceObject]
    dsec_sampler: SpaceObjectDsecSampler  # TODO: support different sampler for different obj?
    # TODO: we need to implement falback mechanism,
    #   e.g. a `Legendre8` `Interpolator` requires >=8 points, but sometime it might get less than that
    interpolator_class: type[Interpolator]


class SpecByControllers(t.TypedDict):
    """A TypedDict of params"""

    controllers: t.Sequence[controller.ControllerBase]
    schedule: Schedule
    epoch: Datetime_Like
    start_time: Datetime_Like
    end_time: Datetime_Like
    space_objects: t.Sequence[sorts.SpaceObject]
    dsec_sampler: SpaceObjectDsecSampler  # TODO: support different sampler for different obj?
    interpolator_class: type[Interpolator]


def prepare_simulation_unit_params(spec: Spec) -> list[FromPassagesOverTxRxStationPairParam]:
    spobjs_smpl_dsec, spobjs_smpl_states = funcs.sample_and_propagate_space_objects_states(
        sampler=spec["dsec_sampler"],
        spobjs=spec["space_objects"],
        start_time=to_datetime64_us(spec["start_time"]),
        end_time=to_datetime64_us(spec["end_time"]),
    )
    logger.debug("sample and propagate done")

    spobjs_interpolators = [
        spec["interpolator_class"](spobj_smpl_states, spobj_smpl_dsec)
        for spobj_smpl_dsec, spobj_smpl_states in zip(spobjs_smpl_dsec, spobjs_smpl_states)
    ]
    logger.debug("interpolators done")

    passages_lists = funcs.find_passages(
        spec=spec,
        spobjs_smpl_dsec=spobjs_smpl_dsec,
        spobjs_smpl_states=spobjs_smpl_states,
    )
    logger.debug("find_passages done")

    sim_units_param = funcs.derive_simulation_unit_params(
        spec=spec,
        passages_lists=passages_lists,
        spobjs_interpolators=spobjs_interpolators,
    )
    # filter away param with empty schedule
    sim_units_param = [
        p for p in sim_units_param if len(p["schedule"]._data[Schedule._K.multi_index]) > 0
    ]
    logger.info(f"prepare_simulation_unit_params done")

    return sim_units_param


def mpi_master_proc_loop(
    comm: MPI.Intracomm, master_proc_rank: int, spec: Spec, rank_size: int, persist_dir: Path
) -> None:
    logger.debug(f"running in mpi with rank: {rank_size}")
    logger.info(f"master: {master_proc_rank} | simulation preparation start")

    sim_units_param = prepare_simulation_unit_params(spec)

    ##
    # parallization section
    ##
    logger.info(f"master: {master_proc_rank} | parallel processing of SimulationUnit start")

    num_workers = comm.Get_size() - 1
    is_worker_idle_list = [True for _ in range(num_workers)]
    processed_sim_unit_cnt = 0
    next_sim_unit_param_idx = 0

    while processed_sim_unit_cnt < len(sim_units_param):

        # TODO: would be more robust to check ids in sim_units_param than looping next_sim_unit_idx
        # send sim_unit if there is idle worker
        if any(is_worker_idle_list) and next_sim_unit_param_idx < len(sim_units_param):

            idle_worker_idx = is_worker_idle_list.index(True)
            idle_worker_rank = idle_worker_idx + 1
            # TODO: check if the  (full ScheduleData + indexer for SimulationUnit) or (just the relevant slices of ScheduleData) are sent
            comm.send(
                WorkerJobAssignment(
                    param=sim_units_param[next_sim_unit_param_idx], persist_dir=persist_dir
                ),
                dest=idle_worker_rank,
            )

            logger.debug(
                f"master: {master_proc_rank} | sent `SimulationUnit`"
                + f" <{sim_units_param[next_sim_unit_param_idx]['id']}>"
                + f" ({next_sim_unit_param_idx+1}/{len(sim_units_param)}) to worker {idle_worker_rank}"
            )

            is_worker_idle_list[idle_worker_idx] = False
            next_sim_unit_param_idx += 1

        else:
            # otherwise, wait for result
            logger.debug(f"master: {master_proc_rank} | awaiting results ...")

            status = MPI.Status()
            recv_obss_cnt: int = comm.recv(status=status)
            worker_rank = status.Get_source()
            logger.debug(
                f"master: {master_proc_rank} | received observation count: {recv_obss_cnt}, from worker: {worker_rank}"
            )
            processed_sim_unit_cnt += 1

            is_worker_idle_list[worker_rank - 1] = True

    logger.info(f"master: {master_proc_rank} | parallel processing of SimulationUnit done")

    # TODO: maybe we can use `comm.bcast` here?
    #   but worker also need to call `comm.bcast` for listening,
    #   not sure if mpi allows listening to both `bcast` and `recv`
    for r in range(1, num_workers + 1):
        logger.debug(f"master: {master_proc_rank} | terminating worker: {r} ...")
        comm.send(_MpiMsg.terminate, dest=r)
        comm.recv(source=r)  # wait for an ack

    # TODO: re-eval if we should implement automatic result gathering
    logger.warning(
        f"master: {master_proc_rank} | observations are not gathered by mpi master process automatically at the moment."
    )
    logger.info(f"master: {master_proc_rank} | master main loop done,  returning...")


def mpi_worker_proc_loop(comm: MPI.Intracomm, master_proc_rank: int, worker_proc_rank: int) -> None:
    while True:
        logger.info(f"worker: {worker_proc_rank} | waiting for msg...")
        msg = comm.recv(source=master_proc_rank)

        match msg:
            case _MpiMsg.terminate:
                # exit if `_MpiMsg.terminate` is received
                logger.info(f"worker: {worker_proc_rank} | exiting...")
                comm.send(_MpiMsg.exit_ok, dest=master_proc_rank)  # reply an ack to master
                exit()

            case _MpiMsg.worker_job_assignment():
                param = msg.param
                persist_fpath = msg.persist_dir / sim_unit_fname_tpl.format(id=param["id"])

                try:
                    if persist_fpath.exists():
                        logger.info(
                            f"worker: {worker_proc_rank} | SimulationUnit: {param["id"]} already completed, will load from the saved file instead"
                        )

                        with open(persist_fpath, "rb") as f:
                            sim_unit = pickle.load(f)

                    else:
                        # NOTE:
                        #   As a simple way to reduce risk of corrupted files,
                        #   we write to an tmp file first then rename that file
                        #
                        #   sim_unit is saved 2 times, 1 before running `simulate` and 1 after

                        sim_unit = SimulationUnit.from_passages_over_tx_rx_station_pair(**param)

                        persist_fpath_tmp = persist_fpath.with_suffix(persist_fpath.suffix + ".tmp")
                        with open(persist_fpath_tmp, "wb") as f:
                            pickle.dump(sim_unit, f)
                            persist_fpath_tmp.rename(persist_fpath)

                        logger.info(f"worker: {worker_proc_rank} | `SimulationUnit.simulate` start")
                        sim_unit.simulate()

                        persist_fpath_tmp = persist_fpath.with_suffix(persist_fpath.suffix + ".tmp")
                        with open(persist_fpath_tmp, "wb") as f:
                            pickle.dump(sim_unit, f)
                            # delete the file we saved earlier, then rename the new dump file
                            persist_fpath.unlink(missing_ok=True)
                            persist_fpath_tmp.rename(persist_fpath)

                except Exception as err:
                    raise RuntimeError(
                        f"Runtime fail in worker: {worker_proc_rank} | SimulationUnit: {param["id"]}"
                    ) from err

                obss = sim_unit.observations

                comm.send(len(obss), dest=master_proc_rank)
                logger.info(
                    f"worker: {worker_proc_rank} | SimulationUnit:{sim_unit.id} done with {len(obss)} observations"
                )

            case _:
                # throw for unexpected msg
                raise RuntimeError(f"worker: {worker_proc_rank} | received unexcepted msg: {msg}")


def iter_mpi_simulation_results(save_dir: Path):
    for fpath in save_dir.glob(sim_unit_fname_tpl.format(id="*")):
        with open(fpath, "rb") as f:
            sim_unit: SimulationUnit = pickle.load(f)
            yield sim_unit


# TODO: we need to enforce each station to has a unique id (`.uid` prop)
#   either in the simulation class or in related station getter like `get_radar`
class StxMrxSimulation:
    """
    NOTE: This is intended as an internal constructor, please use the constructor methods to create instances.
    """

    def __init__(self, spec: Spec):
        self.spec: Spec = spec
        self.sim_units: list[SimulationUnit] = []
        self.obss: list[Observation] = []

    @classmethod
    def from_controllers(cls, spec: SpecByControllers):
        """A constructor method"""

        stn_map: dict[radar.StationId, radar.Station] = {}
        stn_id_pairs_set: set[tuple[radar.StationId, radar.StationId]] = set()
        exp_detail_map: schedule.ExperimentDetailMap = {}

        for ctrl in spec["controllers"]:
            stn_map.update(ctrl.get_station_map())

            for pairs in ctrl.get_experiment_id_station_id_pairs_map().values():
                stn_id_pairs_set.update(pairs)

            exp_detail = ctrl.get_experiment_detail()
            exp_detail_map[exp_detail["id"]] = exp_detail

        return cls(
            spec={
                "station_map": stn_map,
                "station_id_pairs": list(stn_id_pairs_set),
                "schedule": spec["schedule"],
                "exp_detail_map": exp_detail_map,
                "epoch": spec["epoch"],
                "start_time": spec["start_time"],
                "end_time": spec["end_time"],
                "space_objects": spec["space_objects"],
                "dsec_sampler": spec["dsec_sampler"],
                "interpolator_class": spec["interpolator_class"],
            }
        )

    # TODO: is there better way to capture env for working with mpi than using a `SimulationEnvironment`?
    @classmethod
    def mpi_run(
        cls,
        persist_dpath: str | Path,
        prep_sim_env_fn: t.Callable[[Path], SimulationEnvironment],
        result_analysis_fn: t.Callable[[Path, t.Self], None],
    ) -> None:
        try:
            master_proc_rank: t.Final = 0
            comm = MPI.COMM_WORLD
            r = comm.Get_rank()
            rank_size = comm.Get_size()

            if r == master_proc_rank:  # master
                ###
                # simulation
                ###
                persist_dir = Path(persist_dpath)
                calc_start_time = time.perf_counter()

                if not persist_dir.exists():
                    persist_dir.mkdir(parents=True)
                assert persist_dir.exists()
                assert persist_dir.is_dir()

                sim_env = prep_sim_env_fn(persist_dir)

                sim = cls.from_controllers(sim_env["spec_by_controllers"])
                mpi_master_proc_loop(
                    comm=comm,
                    master_proc_rank=r,
                    spec=sim.spec,
                    rank_size=rank_size,
                    persist_dir=persist_dir,
                )

                calc_time = time.perf_counter() - calc_start_time
                logger.info(f"master: {master_proc_rank} | simulation took {calc_time} sec")

                ###
                # result analysis
                ###
                calc_start_time = time.perf_counter()

                logger.info(f"master: {master_proc_rank} | result_analysis_fn start")
                result_analysis_fn(persist_dir, sim)

                calc_time = time.perf_counter() - calc_start_time
                logger.info(f"master: {master_proc_rank} | result_analysis_fn took {calc_time} sec")

                return

            else:  # workers
                mpi_worker_proc_loop(
                    comm=comm, master_proc_rank=master_proc_rank, worker_proc_rank=r
                )

                return

        except Exception as err:
            comm = MPI.COMM_WORLD
            r = comm.Get_rank()

            logger.error(
                f"terminating mpi proc due to exception occured in rank: {r}, error:\n"
                + "\n".join(traceback.format_exception(err))
            )
            comm.Abort(1)

    def run(self) -> tuple[list[Observation], list[SimulationUnit]]:
        logger.debug("starting stx mrx sim")

        self.sim_units = []
        self.obss = []

        sim_units_param = prepare_simulation_unit_params(self.spec)

        pbar = tqdm(desc="simulating", total=len(sim_units_param))

        for param in sim_units_param:
            sim_unit = SimulationUnit.from_passages_over_tx_rx_station_pair(**param)
            self.sim_units.append(sim_unit)

            sim_unit.simulate()
            self.obss.extend(sim_unit.observations)
            pbar.update(1)
        logger.debug("simulation done")

        pbar.close()

        return self.obss, self.sim_units
