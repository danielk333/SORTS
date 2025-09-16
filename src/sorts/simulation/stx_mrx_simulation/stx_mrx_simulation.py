from __future__ import annotations
import logging, typing as t
from pathlib import Path
import numpy.typing as npt
import pyorb
import sorts
from tqdm import tqdm
from mpi4py import MPI
from sorts.interpolation import Interpolator
from sorts.utils import to_datetime64_us
from sorts.types import Datetime_Like, Float64_as_sec
from sorts.schedule import Schedule, ExperimentDetail
from sorts.simulation.stx_mrx_simulation.observation import Observation
from sorts.simulation.stx_mrx_simulation.simulation_unit import SimulationUnit
from . import funcs

logger = logging.getLogger(__name__)


class _MpiMsg:
    exit_ok: t.Final = "exit_ok"
    terminate: t.Final = "terminate"


class SpaceObjectDsecSampler(t.Protocol):
    def __call__(
        self, orbit: pyorb.Orbit, start_time: Datetime_Like, end_time: Datetime_Like
    ) -> npt.NDArray[Float64_as_sec]: ...


class Spec(t.TypedDict):
    """A TypedDict of params"""

    tx_schedule: Schedule
    rx_schedules: t.Sequence[Schedule]
    exp_detail_map: dict[int, ExperimentDetail]
    epoch: Datetime_Like
    start_time: Datetime_Like
    end_time: Datetime_Like
    space_objects: t.Sequence[sorts.SpaceObject]
    dsec_sampler: SpaceObjectDsecSampler  # TODO: support different sampler for different obj?
    # TODO: we need to implement falback mechanism,
    #   e.g. a `Legendre8` `Interpolator` requires >=8 points, but sometime it might get less than that
    interpolator_class: type[Interpolator]


# TODO: we need to enforce each station to has a unique id (`.uid` prop)
#   either in the simulation class or in related station getter like `get_radar`
class StxMrxSimulation:
    def __init__(self, spec: Spec):
        self.spec: Spec = spec
        self.sim_units: list[SimulationUnit] = []
        self.obss: list[Observation] = []

    def run(self) -> tuple[list[Observation], list[SimulationUnit]]:
        logger.debug("starting stx mrx sim")

        spobjs_smpl_dsec, spobjs_smpl_states = funcs.sample_and_propagate_space_objects_states(
            sampler=self.spec["dsec_sampler"],
            spobjs=self.spec["space_objects"],
            start_time=to_datetime64_us(self.spec["start_time"]),
            end_time=to_datetime64_us(self.spec["end_time"]),
        )
        logger.debug("sample and propagate done")

        spobjs_interpolators = [
            self.spec["interpolator_class"](spobj_smpl_states, spobj_smpl_dsec)
            for spobj_smpl_dsec, spobj_smpl_states in zip(spobjs_smpl_dsec, spobjs_smpl_states)
        ]
        logger.debug("interpolators done")

        passages_lists = funcs.find_passages(
            spec=self.spec,
            spobjs_smpl_dsec=spobjs_smpl_dsec,
            spobjs_smpl_states=spobjs_smpl_states,
        )
        logger.debug("find_passages done")

        sim_units = funcs.derive_simulation_units(
            spec=self.spec,
            passages_lists=passages_lists,
            spobjs_interpolators=spobjs_interpolators,
        )
        logger.debug("derive_simulation_units done")

        pbar = tqdm(desc="simulating", total=len(sim_units))

        for sim_unit in sim_units:
            sim_unit.simulate()
            self.obss.extend(funcs.derive_observations(sim_unit))
            pbar.update(1)
        logger.debug("simulation done")

        pbar.close()

        return self.obss, sim_units

    def mpi_run(
        self, persistence_dir_path: str | Path
    ) -> tuple[list[Observation], list[SimulationUnit]]:
        persist_dir = Path(persistence_dir_path)
        assert persist_dir.exists()
        assert persist_dir.is_dir()

        master_proc_rank: t.Final = 0

        comm = MPI.COMM_WORLD
        r = comm.Get_rank()
        rank_size = comm.Get_size()

        if r == master_proc_rank:  # master
            logger.debug(f"running in mpi with rank: {rank_size}")
            logger.info(f"master: {master_proc_rank} | simulation preparation start")

            spobjs_smpl_dsec, spobjs_smpl_states = funcs.sample_and_propagate_space_objects_states(
                sampler=self.spec["dsec_sampler"],
                spobjs=self.spec["space_objects"],
                start_time=to_datetime64_us(self.spec["start_time"]),
                end_time=to_datetime64_us(self.spec["end_time"]),
            )
            logger.debug("sample and propagate done")

            spobjs_interpolators = [
                self.spec["interpolator_class"](spobj_smpl_states, spobj_smpl_dsec)
                for spobj_smpl_dsec, spobj_smpl_states in zip(spobjs_smpl_dsec, spobjs_smpl_states)
            ]
            logger.debug("interpolators done")

            passages_lists = funcs.find_passages(
                spec=self.spec,
                spobjs_smpl_dsec=spobjs_smpl_dsec,
                spobjs_smpl_states=spobjs_smpl_states,
            )
            logger.debug("find_passages done")

            sim_units = funcs.derive_simulation_units(
                spec=self.spec,
                passages_lists=passages_lists,
                spobjs_interpolators=spobjs_interpolators,
            )
            self.sim_units = sim_units

            logger.info(f"master: {master_proc_rank} | simulation preparation done")

            ##
            # MPI section
            ##
            logger.info(f"master: {master_proc_rank} | parallel processing of SimulationUnit start")

            obss: list[Observation] = []

            num_workers = comm.Get_size() - 1
            is_worker_idle_list = [True for _ in range(num_workers)]
            processed_sim_unit_cnt = 0
            next_sim_unit_idx = 0

            while processed_sim_unit_cnt < len(sim_units):

                # send sim_unit if there is idle worker
                if any(is_worker_idle_list) and next_sim_unit_idx < len(sim_units):

                    idle_worker_idx = is_worker_idle_list.index(True)
                    idle_worker_rank = idle_worker_idx + 1
                    comm.send(sim_units[next_sim_unit_idx], dest=idle_worker_rank)

                    logger.debug(
                        f"master: {master_proc_rank} | sent `SimulationUnit` {next_sim_unit_idx+1} of {len(sim_units)} to worker {idle_worker_rank}"
                    )

                    is_worker_idle_list[idle_worker_idx] = False
                    next_sim_unit_idx += 1

                else:
                    # otherwise, wait for result
                    logger.debug(f"master: {master_proc_rank} | awaiting results ...")

                    status = MPI.Status()
                    recv_obss = comm.recv(status=status)
                    worker_rank = status.Get_source()
                    logger.debug(
                        f"master: {master_proc_rank} | received {len(recv_obss)} observations from worker: {worker_rank}"
                    )
                    processed_sim_unit_cnt += 1

                    is_worker_idle_list[worker_rank - 1] = True
                    obss.extend(recv_obss)

            self.obss = obss
            logger.info(f"master: {master_proc_rank} | parallel processing of SimulationUnit done")

            # TODO: maybe we can use `comm.bcast` here?
            #   but worker also need to call `comm.bcast` for listening,
            #   not sure if mpi allows listening to both `bcast` and `recv`
            for r in range(1, num_workers + 1):
                logger.debug(f"master: {master_proc_rank} | terminating worker: {r} ...")
                comm.send(_MpiMsg.terminate, dest=r)
                comm.recv(source=r)  # wait for an ack

            logger.info(f"master: {master_proc_rank} | `mpi_run` done,  returning...")
            return self.obss, self.sim_units

        else:  # workers
            while True:
                logger.info(f"worker: {r} | waiting for msg...")
                msg = comm.recv(source=master_proc_rank)

                # exit if `_MpiMsg.terminate` is received
                if msg == _MpiMsg.terminate:
                    logger.info(f"worker: {r} | exiting...")
                    comm.send(_MpiMsg.exit_ok, dest=master_proc_rank)  # reply an ack to master
                    exit()

                # throw if the msg is not a `SimulationUnit`
                if not isinstance(msg, SimulationUnit):
                    raise RuntimeError(f"worker: {r} | received unexcepted msg: {msg}")

                sim_unit = msg
                logger.info(f"worker: {r} | `SimulationUnit.simulate` start")

                sim_unit.simulate()
                obss = funcs.derive_observations(sim_unit)

                comm.send(obss, dest=master_proc_rank)
                logger.info(f"worker: {r} | `SimulationUnit.simulate` done")
