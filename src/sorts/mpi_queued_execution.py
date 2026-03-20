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
    worker_process_return_ok: t.Final = "worker_process_return_ok"


class MpiQueuedExecution[WorkerJobParams](abc.ABC):
    """
    `MpiQueuedExecution` is a helper abstract class for using MPI.

    It will allocate one MPI process as master process (rank 0) and allocate the reset as worker process (rank >0),
    and automatically dispatch jobs to workers, until the list of jobs is exhausted.
    A job is just a python mapping object.

    The param `is_run_with_mpi` is mostly for debugging. It adjusts code routing so no MPI is used/referenced
    during execution, so we can run it in a single process (without MPI) and attach debugger easily.

    Diagram:
    ```
    +---------------------------------------------------------------------------------+
    |    When executed with MPI:                                                      |
    |                                                                                 |
    |    +------------------+  +------------------+  +------------------+  +-+        |
    |    |MPI process rank 0|  |MPI process rank 1|  |MPI process rank 2|  | |        |
    |    |                  |  |                  |  |                  |  | |        |
    |    |      master      |  |      worker      |  |      worker      |  | |        |
    |    |      [jobs]      |  |      job_0       |  |      job_1       |  | |        |
    |    +------------------+  +------------------+  +------------------+  +-+ ...    |
    |                                                                                 |
    +---------------------------------------------------------------------------------+
    ```

    ## Usage:
    Subclass `MpiQueuedExecution`.

    Implement `master_process` and invoke `self.mpi_master_proc_loop` inside it exactly once.
    It takes a sequence of mapping object as param, and will distribute one item from the list
    to a idle worker process until the list is exhausted.

    Implement `worker_process`. It will receive one of the mapping object from master process as param.

    Instantiate the subclass and invoke its `run` method

    NOTE:
        In typical usage, an instance of this class will be created on each MPI process, which means:
        - Each instance will init it's attributes/properties independently at different time
        - Attributes/Properties of the same name can end up having different value
          (e.g. a timestamp attribute will have different value on each instance.)

    NOTE:
        Provide a generic param of for `MpiQueuedExecution` when subclassing for enhanced type safety,
        python defaults it to `Unknown` if not provided
    """

    # NOTE: Provide default values for generic param is preferred but
    #       such feature only is recently added in python 3.13, which is released on October 7, 2024.
    #       At the moment of writing, we are on 29 October, 2025 only
    #       and 3.13 is too new to be adopted as the baseline python version for a lib.

    # TODO: spawn MPI process from this class instead of using relying on external `mpiexec`.

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
    def worker_process(self, worker_job_params: WorkerJobParams) -> None:
        """The code that only ran on the worker rank processes."""
        ...

    def _mpi_master_proc_loop_with_mpi(
        self, worker_job_params: t.Sequence[WorkerJobParams]
    ) -> None:
        rank_size = self.comm.Get_size()

        logger.debug(f"running in mpi with rank: {rank_size}")
        logger.info(f"master: {self.master_proc_rank} | simulation preparation start")

        ##
        # parallization section
        ##
        logger.info(f"master: {self.master_proc_rank} | parallel processing of worker jobs start")

        is_worker_idle_list = [True for _ in range(self.num_workers)]
        next_work_job_param_idx = 0
        processed_work_job_cnt = 0

        pbar = None
        if self.progress:
            pbar = tqdm("MPI worker progress", total=len(worker_job_params), file=sys.stdout)

        while processed_work_job_cnt < len(worker_job_params):
            # send next work_job_param if there is idle worker
            if any(is_worker_idle_list) and next_work_job_param_idx < len(worker_job_params):
                idle_worker_idx = is_worker_idle_list.index(True)
                idle_worker_rank = idle_worker_idx + 1

                job_param = worker_job_params[next_work_job_param_idx]
                self.comm.send(job_param, dest=idle_worker_rank)

                logger.debug(
                    f"master: {self.master_proc_rank} | sent worker job"
                    + f" ({next_work_job_param_idx+1}/{len(worker_job_params)}) to worker {idle_worker_rank}"
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
                if self.progress and pbar is not None:
                    pbar.update(1)

                is_worker_idle_list[worker_rank - 1] = True

        if self.progress and pbar is not None:
            pbar.close()

        logger.info(f"master: {self.master_proc_rank} | master proc loop done, returning...")

    def _mpi_master_proc_loop_without_mpi(
        self, worker_job_params: t.Sequence[WorkerJobParams]
    ) -> None:
        """This will be ran instead of `mpi_master_proc_loop` when `is_run_with_mpi` is `False`"""

        pbar = None
        if self.progress:
            pbar = tqdm("Worker progress", total=len(worker_job_params), file=sys.stdout)

        for work_job_param in worker_job_params:
            self.worker_process(work_job_param)
            if self.progress and pbar is not None:
                pbar.update(1)
        if self.progress and pbar is not None:
            pbar.close()

        logger.info("master proc loop done, returning...")

    def mpi_master_proc_loop(self, worker_job_params: t.Sequence[WorkerJobParams]) -> None:
        if self.is_run_with_mpi:
            return self._mpi_master_proc_loop_with_mpi(worker_job_params)
        else:
            return self._mpi_master_proc_loop_without_mpi(worker_job_params)

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
                job_param = t.cast(WorkerJobParams, msg)
                try:
                    self.worker_process(job_param)
                    self.comm.send(_MpiK.worker_process_return_ok, dest=self.master_proc_rank)
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
