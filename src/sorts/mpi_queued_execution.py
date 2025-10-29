from __future__ import annotations
import logging, typing as t, pickle, traceback, time, abc
from pathlib import Path
from dataclasses import dataclass
from mpi4py import MPI
from sorts import StxMrxSimulation  #


logger = logging.getLogger(__name__)


class _MpiK:
    """Internal helper for accessing string constant consistently"""

    exit_ok: t.Final = "exit_ok"
    terminate: t.Final = "terminate"


# NOTE: Resorted to using a rather loosely typed dict instead of setting a generic param of `MpiQueuedExecution`
#       because it seems python does not infer generic param based on method signatures of subclasses.
#       Which mean if the generic param are not provided when subclassing, they are considered as Any/Unknown.
#
#       Althought using a loosely typed dict will not provide a strict type correctness,
#       it at least provide the following values:
#       - a loosely typed dict is better than Unknown/Any
#       - user can cast it to TypedDict for enhanced type safety will minimal friction
#       - code are easier to read without generics, especially when the behaviour of subclassing with generics is not so clear
#
#       An alternative solution is to provide default values for generic param.
#       Such feature is recently added in python 3.13, which is released on October 7, 2024.
#       But at the moment of writing, we are on October 29, 2025 only
#       and 3.13 is too new to be adopted as the baseline python version for a lib.
MasterProcessEnvironment = dict[str, t.Any]
"""A dict of `[str, Any]`. Cast to `TypedDict` for enhanced type safety."""


class WorkerJobAssignment(t.TypedDict):
    param: t.Any
    persist_dpath: Path


class MpiQueuedExecution(abc.ABC):
    master_proc_rank = 0

    def __init__(self, sim_unit_fname_tpl: str):
        # TODO: is there a better way to pass and store this `sim_unit_fname_tpl`?
        self.sim_unit_fname_tpl = sim_unit_fname_tpl

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

    @abc.abstractmethod
    def prepare_master_process_environment(self) -> MasterProcessEnvironment:
        """
        Prepare the "master_process_environment" and returns it.

        The "master_process_environment" will be passed to subsequence master functions.

        NOTE:
            - This method will run in the master process.
            - The "master_process_environment" is automatically peristed and must be serializable by pickle.
        """
        ...

    @abc.abstractmethod
    def create_simulation(self, menv: MasterProcessEnvironment) -> StxMrxSimulation:
        """NOTE: This method will run in the master process."""
        ...

    @abc.abstractmethod
    def run_worker_job(self, persist_dpath: Path, jab_param: WorkerJobAssignment) -> None:
        """NOTE: This method will run in the worker process."""
        ...

    @abc.abstractmethod
    def analyze_result(
        self, menv: MasterProcessEnvironment, sim: StxMrxSimulation, persist_dpath: Path
    ) -> None:
        """NOTE: This method will run in the master process."""
        ...

    def mpi_master_proc_loop(
        self,
        persist_dpath: Path,
        menv: MasterProcessEnvironment,
        sim: StxMrxSimulation,
    ) -> None:
        rank_size = self.comm.Get_size()

        logger.debug(f"running in mpi with rank: {rank_size}")
        logger.info(f"master: {self.master_proc_rank} | simulation preparation start")

        sim_units_param = sim.prepare_simulation_unit_params()

        ##
        # parallization section
        ##
        logger.info(
            f"master: {self.master_proc_rank} | parallel processing of SimulationUnit start"
        )

        num_workers = self.comm.Get_size() - 1
        is_worker_idle_list = [True for _ in range(num_workers)]
        next_sim_unit_param_idx = 0
        processed_sim_unit_cnt = 0

        while processed_sim_unit_cnt < len(sim_units_param):

            # TODO: would be more robust to check ids in sim_units_param than looping next_sim_unit_idx
            # send sim_unit if there is idle worker
            if any(is_worker_idle_list) and next_sim_unit_param_idx < len(sim_units_param):
                idle_worker_idx = is_worker_idle_list.index(True)
                idle_worker_rank = idle_worker_idx + 1

                job_param: WorkerJobAssignment = {
                    "param": sim_units_param[next_sim_unit_param_idx],
                    "persist_dpath": persist_dpath,
                }
                self.comm.send(job_param, dest=idle_worker_rank)

                logger.debug(
                    f"master: {self.master_proc_rank} | sent `SimulationUnit`"
                    + f" <{sim_units_param[next_sim_unit_param_idx].id}>"
                    + f" ({next_sim_unit_param_idx+1}/{len(sim_units_param)}) to worker {idle_worker_rank}"
                )

                is_worker_idle_list[idle_worker_idx] = False
                next_sim_unit_param_idx += 1

            else:
                # otherwise, wait for result
                logger.debug(f"master: {self.master_proc_rank} | awaiting results ...")

                status = MPI.Status()
                recv_obss_cnt: int = self.comm.recv(status=status)
                worker_rank = status.Get_source()
                logger.debug(
                    f"master: {self.master_proc_rank} | received observation count: {recv_obss_cnt}, from worker: {worker_rank}"
                )
                processed_sim_unit_cnt += 1

                is_worker_idle_list[worker_rank - 1] = True

        logger.info(f"master: {self.master_proc_rank} | parallel processing of SimulationUnit done")

        # TODO: maybe we can use `comm.bcast` here?
        #   but worker also need to call `comm.bcast` for listening,
        #   not sure if mpi allows listening to both `bcast` and `recv`
        for r in range(1, num_workers + 1):
            logger.debug(f"master: {self.master_proc_rank} | terminating worker: {r} ...")
            self.comm.send(_MpiK.terminate, dest=r)
            self.comm.recv(source=r)  # wait for an ack

        # TODO: re-eval if we should implement automatic result gathering
        logger.warning(
            f"master: {self.master_proc_rank} | observations are not gathered by mpi master process automatically at the moment."
        )
        logger.info(f"master: {self.master_proc_rank} | master main loop done,  returning...")

    def mpi_worker_proc_loop(self, persist_dpath: Path) -> None:
        worker_proc_rank = self.rank

        while True:
            logger.info(f"worker: {worker_proc_rank} | waiting for msg...")
            msg = self.comm.recv(source=self.master_proc_rank)

            if msg == _MpiK.terminate:
                # exit if `_MpiK.terminate` is received
                logger.info(f"worker: {worker_proc_rank} | exiting...")
                self.comm.send(_MpiK.exit_ok, dest=self.master_proc_rank)  # reply an ack to master
                exit()

            # TODO: would be nice to get better type checking then just dict here
            elif isinstance(msg, dict):
                job_param = t.cast(WorkerJobAssignment, msg)
                self.run_worker_job(persist_dpath, job_param)

            else:
                # throw for unexpected msg
                raise RuntimeError(f"worker: {worker_proc_rank} | received unexcepted msg: {msg}")

    def run_with_mpi(self, persist_dpath: str | Path):
        try:
            comm = MPI.COMM_WORLD
            r = comm.Get_rank()
            persist_dpath = Path(persist_dpath)

            if r == self.master_proc_rank:  # master
                ###
                # simulation
                ###
                calc_start_time = time.perf_counter()

                if not persist_dpath.exists():
                    persist_dpath.mkdir(parents=True)
                assert persist_dpath.exists()
                assert persist_dpath.is_dir()

                menv = self.prepare_master_process_environment()

                # saving sim env
                persist_fpath = persist_dpath / f"menv.pickle"
                persist_fpath_tmp = persist_fpath.with_suffix(persist_fpath.suffix + ".tmp")
                with open(persist_fpath_tmp, "wb") as f:
                    pickle.dump(menv, f)
                persist_fpath_tmp.rename(persist_fpath)

                sim = self.create_simulation(menv)

                self.mpi_master_proc_loop(persist_dpath=persist_dpath, menv=menv, sim=sim)

                calc_time = time.perf_counter() - calc_start_time
                logger.info(f"master: {self.master_proc_rank} | simulation took {calc_time} sec")

                ###
                # result analysis
                ###
                calc_start_time = time.perf_counter()

                logger.info(f"master: {self.master_proc_rank} | result_analysis_fn start")
                self.analyze_result(menv=menv, sim=sim, persist_dpath=persist_dpath)

                calc_time = time.perf_counter() - calc_start_time
                logger.info(
                    f"master: {self.master_proc_rank} | result_analysis_fn took {calc_time} sec"
                )

                return

            else:  # workers
                self.mpi_worker_proc_loop(persist_dpath=persist_dpath)

                return

        except Exception as err:
            comm = MPI.COMM_WORLD
            r = comm.Get_rank()

            logger.error(
                f"terminating mpi proc due to exception occured in rank: {r}, error:\n"
                + "\n".join(traceback.format_exception(err))
            )
            comm.Abort(1)

    def run_without_mpi(self):
        """Run the execution without MPI, mostly useful for debugging."""

        menv = self.prepare_master_process_environment()
        sim = self.create_simulation(menv)

        calc_start_time = time.perf_counter()
        obss, sim_units = sim.run()
        calc_time = time.perf_counter() - calc_start_time
        logger.info(f"result_analysis_fn took {calc_time} sec")

        print(f"len(obss): {len(obss)}")

    def run(self, persist_dpath: str | Path, is_run_with_mpi=True):
        if is_run_with_mpi:
            self.run_with_mpi(persist_dpath)
        else:
            self.run_without_mpi()
