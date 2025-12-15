from __future__ import annotations
from tqdm import tqdm
import logging, typing as t, traceback, time, abc
import sys
from mpi4py import MPI


logger = logging.getLogger(__name__)


class WorkerError(RuntimeError):
    pass


class _MpiK:
    """Internal helper for accessing string constant consistently"""

    exit_ok: t.Final = "exit_ok"
    terminate: t.Final = "terminate"


# TODO:
# The architecture of this class could use a nice diagram, its really general purpose which is super
# nice! But to avoid tracing out the control flow manually, a diagram could shortcut someone looking
# at this for the first time

# NOTE: Resorted to using a loose `Mapping` type instead of setting a generic param of `MpiQueuedExecution`
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
WorkerJobParam = t.Mapping
"""Alias of `Mapping`. Cast to `TypedDict` for enhanced type safety."""


class MpiQueuedExecution(abc.ABC):
    """
    NOTE: In typical usage, an instance of this class will be created on each MPI process, which means:
        - Each instance will init it's attributes/properties independently at different time
        - Attributes/Properties of the same name can end up having different value
          (e.g. a timestamp attribute will have different value on each instance.)

    """

    master_proc_rank = 0

    def __init__(self, is_run_with_mpi=True, progress=False):
        self.is_run_with_mpi = is_run_with_mpi

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_workers = self.comm.Get_size() - 1
        self.progress = progress

    # TODO: maybe this can take kwargs that are passed from run?
    @abc.abstractmethod
    def master_process(self) -> None:
        """
        The code that only ran on the master rank process.
        - Must invoke the method `mpi_master_proc_loop` at least once to start dispatching job to workers.
        """

    @abc.abstractmethod
    def worker_process(self, job_param: WorkerJobParam) -> None:
        """The code that only ran on the worker rank processes."""
        ...

    def _mpi_master_proc_loop_with_mpi(self, work_job_params: t.Sequence[WorkerJobParam]) -> None:
        rank_size = self.comm.Get_size()

        logger.debug(f"running in mpi with rank: {rank_size}")
        logger.info(f"master: {self.master_proc_rank} | simulation preparation start")

        ##
        # parallization section
        ##
        logger.info(
            f"master: {self.master_proc_rank} | parallel processing of SimulationUnit start"
        )

        is_worker_idle_list = [True for _ in range(self.num_workers)]
        next_work_job_param_idx = 0
        processed_work_job_cnt = 0
        if self.progress:
            pbar = tqdm("MPI worker progress", total=len(work_job_params), file=sys.stdout)

        while processed_work_job_cnt < len(work_job_params):
            # send next work_job_param if there is idle worker
            if any(is_worker_idle_list) and next_work_job_param_idx < len(work_job_params):
                idle_worker_idx = is_worker_idle_list.index(True)
                idle_worker_rank = idle_worker_idx + 1

                job_param = work_job_params[next_work_job_param_idx]
                self.comm.send(job_param, dest=idle_worker_rank)

                logger.debug(
                    f"master: {self.master_proc_rank} | sent `SimulationUnit`"
                    + f" ({next_work_job_param_idx+1}/{len(work_job_params)}) to worker {idle_worker_rank}"
                )

                is_worker_idle_list[idle_worker_idx] = False
                next_work_job_param_idx += 1

            # otherwise, wait for result
            else:
                logger.debug(f"master: {self.master_proc_rank} | awaiting results ...")

                status = MPI.Status()
                self.comm.recv(status=status)
                worker_rank = status.Get_source()

                processed_work_job_cnt += 1
                if self.progress:
                    pbar.update(1)

                is_worker_idle_list[worker_rank - 1] = True

        if self.progress:
            pbar.close()
        logger.info(f"master: {self.master_proc_rank} | master proc loop done, returning...")

    def _mpi_master_proc_loop_without_mpi(
        self, work_job_params: t.Sequence[WorkerJobParam]
    ) -> None:
        """This will be ran instead of `mpi_master_proc_loop` when `is_run_with_mpi` is `False`"""

        if self.progress:
            pbar = tqdm("Worker progress", total=len(work_job_params), file=sys.stdout)
        for work_job_param in work_job_params:
            self.worker_process(work_job_param)
            if self.progress:
                pbar.update(1)
        if self.progress:
            pbar.close()
        logger.info("master proc loop done, returning...")

    def mpi_master_proc_loop(self, work_job_params: t.Sequence[WorkerJobParam]) -> None:
        if self.is_run_with_mpi:
            return self._mpi_master_proc_loop_with_mpi(work_job_params)
        else:
            return self._mpi_master_proc_loop_without_mpi(work_job_params)

    def mpi_worker_proc_loop(self) -> None:
        worker_proc_rank = self.rank

        while True:
            logger.info(f"worker: {worker_proc_rank} | waiting for msg...")
            msg = self.comm.recv(source=self.master_proc_rank)

            if msg == _MpiK.terminate:
                # exit if `_MpiK.terminate` is received
                logger.info(f"worker: {worker_proc_rank} | exiting...")
                self.comm.send(_MpiK.exit_ok, dest=self.master_proc_rank)  # reply an ack to master

                # break here to allow for further execution of workers later
                break

            elif isinstance(msg, t.Mapping):
                job_param = t.cast(WorkerJobParam, msg)
                try:
                    self.worker_process(job_param)
                except BaseException as exc:
                    raise WorkerError(
                        "Error during job:\n "
                        + "\n".join([f"{key}: {val}" for key, val in job_param.items()])
                    ) from exc

            else:
                # throw for unexpected msg
                raise RuntimeError(f"worker: {worker_proc_rank} | received unexcepted msg: {msg}")

    def _run_with_mpi(self):
        try:
            # master
            if self.rank == self.master_proc_rank:
                calc_start_time = time.perf_counter()

                self.master_process()

                # TODO: maybe we can use `comm.bcast` here?
                #   but worker also need to call `comm.bcast` for listening,
                #   not sure if mpi allows listening to both `bcast` and `recv`
                for r in range(1, self.num_workers + 1):
                    logger.debug(f"master: {self.master_proc_rank} | terminating worker: {r} ...")
                    self.comm.send(_MpiK.terminate, dest=r)
                    self.comm.recv(source=r)  # wait for an ack

                calc_time = time.perf_counter() - calc_start_time
                logger.info(f"master_process took {calc_time} sec")

            # workers
            else:
                self.mpi_worker_proc_loop()

        except Exception as err:
            comm = MPI.COMM_WORLD
            r = comm.Get_rank()

            logger.error(
                f"terminating mpi proc due to exception occured in rank: {r}, error:\n"
                + "\n".join(traceback.format_exception(err))
            )
            comm.Abort(1)

    def _run_without_mpi(self):
        """Run the execution without MPI, mostly useful for debugging."""

        calc_start_time = time.perf_counter()

        self.master_process()

        calc_time = time.perf_counter() - calc_start_time
        logger.info(f"master_process took {calc_time} sec")

    def run(self):
        if self.is_run_with_mpi:
            self._run_with_mpi()
        else:
            self._run_without_mpi()
